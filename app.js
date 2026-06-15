const $ = (selector) => document.querySelector(selector);
const MODEL_URL = "model.onnx";
const ORT_ASSET_ROOT = new URL("vendor/ort/", window.location.href).href;
const IMAGE_SIZE = 384;
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD = [0.229, 0.224, 0.225];

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const emptyState = $("#empty-state");
const previewState = $("#preview-state");
const previewImage = $("#preview-image");
const analyzeButton = $("#analyze-button");
const statusMessage = $("#status-message");

let selectedFile = null;
let selectedObjectUrl = null;
let sessionPromise = null;
let lastReport = null;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = ORT_ASSET_ROOT;

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function clearFile(event) {
  event?.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
  selectedObjectUrl = null;
  previewImage.removeAttribute("src");
  previewState.hidden = true;
  emptyState.hidden = false;
  analyzeButton.disabled = true;
  statusMessage.textContent = "";
}

function selectFile(file) {
  statusMessage.textContent = "";
  if (!file || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    statusMessage.textContent = "Choose a JPEG, PNG or WEBP image.";
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    statusMessage.textContent = "The image is larger than 15 MB.";
    return;
  }
  if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
  selectedFile = file;
  selectedObjectUrl = URL.createObjectURL(file);
  previewImage.src = selectedObjectUrl;
  $("#preview-name").textContent = file.name;
  $("#preview-size").textContent = formatBytes(file.size);
  emptyState.hidden = true;
  previewState.hidden = false;
  analyzeButton.disabled = false;
}

async function loadSession() {
  if (!sessionPromise) {
    statusMessage.textContent = "Loading the 83 MB vision model. The first run can take a minute...";
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }).catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

async function decodeImage(file) {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height) * (384 / 440);
  const sourceX = Math.floor((bitmap.width - side) / 2);
  const sourceY = Math.floor((bitmap.height - side) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, sourceX, sourceY, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  bitmap.close();
  return context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

function imageDataToTensor(imageData) {
  const pixels = imageData.data;
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const values = new Float32Array(plane * 3);
  for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
    const sourceIndex = pixelIndex * 4;
    values[pixelIndex] = ((pixels[sourceIndex] / 255) - IMAGE_MEAN[0]) / IMAGE_STD[0];
    values[plane + pixelIndex] = ((pixels[sourceIndex + 1] / 255) - IMAGE_MEAN[1]) / IMAGE_STD[1];
    values[(plane * 2) + pixelIndex] = ((pixels[sourceIndex + 2] / 255) - IMAGE_MEAN[2]) / IMAGE_STD[2];
  }
  return new ort.Tensor("float32", values, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function textureComplexity(imageData) {
  const pixels = imageData.data;
  let difference = 0;
  let samples = 0;
  for (let y = 0; y < IMAGE_SIZE; y += 4) {
    for (let x = 0; x < IMAGE_SIZE - 4; x += 4) {
      const current = ((y * IMAGE_SIZE) + x) * 4;
      const next = current + 16;
      difference += Math.abs(pixels[current] - pixels[next]);
      difference += Math.abs(pixels[current + 1] - pixels[next + 1]);
      difference += Math.abs(pixels[current + 2] - pixels[next + 2]);
      samples += 3;
    }
  }
  return Math.min(1, (difference / Math.max(samples, 1)) / 48);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function classify(probability) {
  if (probability >= 0.68) return { verdict: "likely_ai", label: "Likely AI-generated" };
  if (probability <= 0.32) return { verdict: "likely_real", label: "Likely camera-captured" };
  return { verdict: "inconclusive", label: "Inconclusive" };
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function renderReport(report) {
  lastReport = report;
  $("#result-panel").classList.add("has-result");
  $("#result-placeholder").hidden = true;
  $("#result-content").hidden = false;
  $("#engine-badge").textContent = "Community Forensics / ONNX Web";
  $("#scan-id").textContent = `Analysis ${report.scan_id.slice(0, 8)}`;
  $("#verdict-label").textContent = report.label;
  const score = Math.round(report.ai_probability * 100);
  const confidence = Math.round(report.confidence * 100);
  $("#score-value").textContent = `${score}%`;
  $("#score-ring").style.setProperty("--score", score);
  $("#confidence-value").textContent = `${confidence}%`;
  $("#confidence-bar").style.width = `${confidence}%`;
  $("#evidence-list").innerHTML = report.evidence.map((item) => `
    <div class="evidence-item ${item.direction}"><i></i><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div></div>
  `).join("");
  $("#signal-list").innerHTML = report.signals.map((signal) => `
    <div class="signal-row"><div><span>${escapeHtml(signal.label)}</span><b>${Math.round(signal.value * 100)}</b></div><div class="mini-track"><i style="width:${Math.round(signal.value * 100)}%"></i></div></div>
  `).join("");
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  analyzeButton.querySelector("span").textContent = "Analyzing";
  try {
    const [session, imageData] = await Promise.all([loadSession(), decodeImage(selectedFile)]);
    statusMessage.textContent = "Running the vision transformer locally in your browser...";
    const input = imageDataToTensor(imageData);
    const outputMap = await session.run({ [session.inputNames[0]]: input });
    const logit = Number(outputMap[session.outputNames[0]].data[0]);
    const probability = sigmoid(logit);
    const classification = classify(probability);
    const confidence = Math.min(0.95, 0.44 + (Math.abs(probability - 0.5) * 1.02));
    const texture = textureComplexity(imageData);
    const report = {
      scan_id: crypto.randomUUID(),
      created_at: new Date().toISOString(),
      file_name: selectedFile.name,
      verdict: classification.verdict,
      label: classification.label,
      ai_probability: probability,
      confidence,
      model: "OwensLab/commfor-model-384",
      runtime: "ONNX Runtime Web 1.22.0 / WebAssembly",
      evidence: [
        {
          direction: probability >= 0.5 ? "ai" : "real",
          title: probability >= 0.5 ? "Synthetic-pattern support" : "Camera-pattern support",
          detail: `The learned model assigned ${Math.round(probability * 100)}% probability to the synthetic-image class.`,
        },
        {
          direction: classification.verdict === "inconclusive" ? "neutral" : "real",
          title: classification.verdict === "inconclusive" ? "Decision boundary is close" : "Decision margin is meaningful",
          detail: classification.verdict === "inconclusive"
            ? "The score is too close to the model boundary for a strong attribution."
            : "The score is outside the conservative inconclusive range, but remains probabilistic evidence.",
        },
        {
          direction: "neutral",
          title: "On-device privacy",
          detail: "The image was processed in this browser and was not sent to an analysis server.",
        },
      ],
      signals: [
        { label: "AI likelihood", value: probability },
        { label: "Camera likelihood", value: 1 - probability },
        { label: "Decision confidence", value: confidence },
        { label: "Local texture complexity", value: texture },
      ],
      limitations: "Detector estimates can be affected by unseen generators, editing, screenshots, resizing and compression.",
    };
    renderReport(report);
    statusMessage.textContent = "";
  } catch (error) {
    console.error(error);
    statusMessage.textContent = "The browser model could not load. Check the connection, refresh, and try again.";
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("loading");
    analyzeButton.querySelector("span").textContent = "Analyze provenance";
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (["Enter", " "].includes(event.key)) {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
$("#remove-image").addEventListener("click", clearFile);
analyzeButton.addEventListener("click", analyze);

["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

document.addEventListener("paste", (event) => {
  const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
  const pastedFile = imageItem?.getAsFile();
  if (!pastedFile) return;
  event.preventDefault();
  const extension = pastedFile.type.split("/")[1]?.replace("jpeg", "jpg") || "png";
  selectFile(new File([pastedFile], `pasted-image.${extension}`, { type: pastedFile.type }));
});

$("#download-report").addEventListener("click", () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `doubledeep-${lastReport.scan_id}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});
