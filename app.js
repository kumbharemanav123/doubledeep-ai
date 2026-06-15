const $ = (selector) => document.querySelector(selector);
const MODEL_MANIFEST_URL = "model-manifest.json";
const ORT_ASSET_ROOT = new URL("vendor/ort/", window.location.href).href;
const IMAGE_SIZE = 224;
const IMAGE_MEAN = [0.5, 0.5, 0.5];
const IMAGE_STD = [0.5, 0.5, 0.5];

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
    sessionPromise = fetch(MODEL_MANIFEST_URL)
      .then((response) => {
        if (!response.ok) throw new Error("Model manifest could not be loaded.");
        return response.json();
      })
      .then((manifest) => {
        const totalSize = formatBytes(manifest.total_size);
        statusMessage.textContent = `Loading the ${totalSize} primary vision model. The first run can take a few minutes...`;
        return ort.InferenceSession.create(new URL(manifest.model, window.location.href).href, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
          externalData: manifest.external_data.map((file) => ({
            path: file.path,
            data: new URL(file.path, window.location.href).href,
          })),
        });
      })
      .catch((error) => {
        sessionPromise = null;
        throw error;
      });
  }
  return sessionPromise;
}

async function decodeImage(file) {
  return createImageBitmap(file);
}

function buildViewRects(bitmap) {
  const width = bitmap.width;
  const height = bitmap.height;
  const views = [{ x: 0, y: 0, width, height }];

  if (height > width * 1.1) {
    const side = width;
    [0, Math.floor((height - side) / 2), height - side].forEach((y) => {
      views.push({ x: 0, y, width: side, height: side });
    });
  } else if (width > height * 1.1) {
    const side = height;
    [0, Math.floor((width - side) / 2), width - side].forEach((x) => {
      views.push({ x, y: 0, width: side, height: side });
    });
  } else {
    const side = Math.max(32, Math.floor(Math.min(width, height) * 0.84));
    views.push(
      { x: 0, y: 0, width: side, height: side },
      { x: Math.floor((width - side) / 2), y: Math.floor((height - side) / 2), width: side, height: side },
      { x: width - side, y: height - side, width: side, height: side },
    );
  }
  return views;
}

function renderView(bitmap, view) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, view.x, view.y, view.width, view.height, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

function imageViewsToTensor(imageViews) {
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const values = new Float32Array(imageViews.length * plane * 3);
  imageViews.forEach((imageData, viewIndex) => {
    const pixels = imageData.data;
    const viewOffset = viewIndex * plane * 3;
    for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
      const sourceIndex = pixelIndex * 4;
      values[viewOffset + pixelIndex] = ((pixels[sourceIndex] / 255) - IMAGE_MEAN[0]) / IMAGE_STD[0];
      values[viewOffset + plane + pixelIndex] = ((pixels[sourceIndex + 1] / 255) - IMAGE_MEAN[1]) / IMAGE_STD[1];
      values[viewOffset + (plane * 2) + pixelIndex] = ((pixels[sourceIndex + 2] / 255) - IMAGE_MEAN[2]) / IMAGE_STD[2];
    }
  });
  return new ort.Tensor("float32", values, [imageViews.length, 3, IMAGE_SIZE, IMAGE_SIZE]);
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

function classify(probability, confidence) {
  if (confidence < 0.48 || (probability >= 0.43 && probability <= 0.57)) {
    return { verdict: "inconclusive", label: "Inconclusive" };
  }
  if (probability > 0.57) return { verdict: "likely_ai", label: "Likely AI-generated" };
  if (probability < 0.43) return { verdict: "likely_real", label: "Likely camera-captured" };
  return { verdict: "inconclusive", label: "Inconclusive" };
}

function aggregateProbabilities(probabilities) {
  const sorted = [...probabilities].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
  const mean = probabilities.reduce((sum, value) => sum + value, 0) / probabilities.length;
  const variance = probabilities.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / probabilities.length;
  const disagreement = Math.sqrt(variance);
  const aiSupport = probabilities.filter((value) => value >= 0.8).length / probabilities.length;
  const realSupport = probabilities.filter((value) => value <= 0.2).length / probabilities.length;
  const strongConflict = Math.max(...probabilities) >= 0.95 && Math.min(...probabilities) <= 0.05;

  let probability = (0.56 * median) + (0.24 * mean) + (0.20 * probabilities[0]);
  if (aiSupport > 0.5) {
    probability = Math.max(probability, 0.64 + (0.28 * aiSupport));
  } else if (strongConflict) {
    probability = 0.5 + (0.08 * (aiSupport - realSupport));
  } else if (realSupport > 0.5) {
    probability = Math.min(probability, 0.36 - (0.24 * realSupport));
  }
  probability = Math.max(0.01, Math.min(0.99, probability));

  const support = Math.max(aiSupport, realSupport);
  let confidence = 0.52 + (0.30 * Math.abs(probability - 0.5) * 2) + (0.18 * support) - (0.34 * disagreement);
  if (strongConflict && aiSupport <= 0.5) confidence = Math.min(confidence, 0.44);
  confidence = Math.max(0.25, Math.min(0.96, confidence));
  return { probability, confidence, disagreement, support, strongConflict };
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
  $("#engine-badge").textContent = "DoubleDeep SigLIP / ONNX Web";
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
    const [session, bitmap] = await Promise.all([loadSession(), decodeImage(selectedFile)]);
    statusMessage.textContent = "Running whole-image and regional checks locally in your browser...";
    const imageViews = buildViewRects(bitmap).map((view) => renderView(bitmap, view));
    bitmap.close();
    const input = imageViewsToTensor(imageViews);
    const outputMap = await session.run({ [session.inputNames[0]]: input });
    const probabilities = Array.from(outputMap[session.outputNames[0]].data, (logit) => sigmoid(Number(logit)));
    const assessment = aggregateProbabilities(probabilities);
    const probability = assessment.probability;
    const confidence = assessment.confidence;
    const classification = classify(probability, confidence);
    const texture = textureComplexity(imageViews[0]);
    const report = {
      scan_id: crypto.randomUUID(),
      created_at: new Date().toISOString(),
      file_name: selectedFile.name,
      verdict: classification.verdict,
      label: classification.label,
      ai_probability: probability,
      confidence,
      model: "Ateeqq/ai-vs-human-image-detector",
      runtime: "ONNX Runtime Web 1.22.0 / WebAssembly",
      evidence: [
        {
          direction: probability >= 0.5 ? "ai" : "real",
          title: probability >= 0.5 ? "Synthetic-pattern support" : "Camera-pattern support",
          detail: `The primary detector assigned ${Math.round(probability * 100)}% aggregated probability to the synthetic-image class.`,
        },
        {
          direction: assessment.strongConflict ? "neutral" : (probability >= 0.5 ? "ai" : "real"),
          title: assessment.strongConflict ? "Regional views disagree" : "Multi-view assessment",
          detail: assessment.strongConflict
            ? "Whole-image and crop-level predictions conflict, so confidence is reduced."
            : `${Math.round(assessment.support * 100)}% of whole-image and regional views provide strong support in the same direction.`,
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
        { label: "Model view agreement", value: assessment.support },
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
