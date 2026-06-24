const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const MODEL_MANIFEST_URL = "model-manifest.json";
const FEEDBACK_MEMORY_URL = "feedback-memory.json";
const LOCAL_FEEDBACK_KEY = "doubledeep.feedback.v2";
const ORT_ASSET_ROOT = new URL("vendor/ort/", window.location.href).href;
const IMAGE_SIZE = 224;
const IMAGE_MEAN = [0.5, 0.5, 0.5];
const IMAGE_STD = [0.5, 0.5, 0.5];
const DCT_SIZE = 32;
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const MAX_VIDEO_BYTES = 600 * 1024 * 1024;
const MAX_VIDEO_SECONDS = 5 * 60;
const VIDEO_FRAME_LIMIT = 18;

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const emptyState = $("#empty-state");
const previewState = $("#preview-state");
const previewImage = $("#preview-image");
const previewVideo = $("#preview-video");
const analyzeButton = $("#analyze-button");
const statusMessage = $("#status-message");
const urlPanel = $("#url-panel");
const videoUrlInput = $("#video-url");

let activeMode = "image";
let selectedFile = null;
let selectedObjectUrl = null;
let selectedUrl = "";
let selectedSourceType = "file";
let sessionPromise = null;
let feedbackMemoryPromise = null;
let lastReport = null;
let lastFingerprint = null;
let selectedFeedback = "";
let selectedCorrection = "";
let dctMatrix = null;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = ORT_ASSET_ROOT;

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function readLocalFeedback() {
  try {
    return JSON.parse(localStorage.getItem(LOCAL_FEEDBACK_KEY) || '{"version":2,"items":[]}');
  } catch {
    return { version: 2, items: [] };
  }
}

function writeLocalFeedback(data) {
  localStorage.setItem(LOCAL_FEEDBACK_KEY, JSON.stringify(data));
}

async function loadFeedbackMemory() {
  if (!feedbackMemoryPromise) {
    feedbackMemoryPromise = fetch(FEEDBACK_MEMORY_URL, { cache: "no-cache" })
      .then((response) => (response.ok ? response.json() : { exact: {}, visual: [] }))
      .catch(() => ({ exact: {}, visual: [] }));
  }
  const bundled = await feedbackMemoryPromise;
  const local = readLocalFeedback();
  const exact = { ...(bundled.exact || {}) };
  const visual = [...(bundled.visual || [])];
  local.items.forEach((item) => {
    if (!["ai", "real"].includes(item.corrected_label)) return;
    if (item.sha256) exact[item.sha256] = item.corrected_label;
    if (item.fingerprint?.phash && item.fingerprint?.dhash) {
      visual.push({ ...item.fingerprint, sha256: item.sha256, label: item.corrected_label });
    }
  });
  return {
    version: 2,
    threshold: bundled.threshold ?? 0.9,
    conflict_margin: bundled.conflict_margin ?? 0.025,
    exact,
    visual,
  };
}

async function sha256Hex(file) {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Text(value) {
  const data = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function getDctMatrix() {
  if (dctMatrix) return dctMatrix;
  dctMatrix = Array.from({ length: DCT_SIZE }, (_, u) => {
    const scale = Math.sqrt(2 / DCT_SIZE) * (u === 0 ? 1 / Math.sqrt(2) : 1);
    return Array.from({ length: DCT_SIZE }, (_, x) => scale * Math.cos((Math.PI * ((2 * x) + 1) * u) / (2 * DCT_SIZE)));
  });
  return dctMatrix;
}

function grayscaleSample(source, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, source.width || source.videoWidth, source.height || source.videoHeight, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let index = 0; index < gray.length; index += 1) {
    const sourceIndex = index * 4;
    gray[index] = (0.299 * pixels[sourceIndex]) + (0.587 * pixels[sourceIndex + 1]) + (0.114 * pixels[sourceIndex + 2]);
  }
  return gray;
}

function visualFingerprint(source) {
  const matrix = getDctMatrix();
  const width = source.width || source.videoWidth || 1;
  const height = source.height || source.videoHeight || 1;
  const sample = grayscaleSample(source, DCT_SIZE, DCT_SIZE);
  const lowFrequency = [];
  for (let u = 0; u < 8; u += 1) {
    for (let v = 0; v < 8; v += 1) {
      if (u === 0 && v === 0) continue;
      let value = 0;
      for (let x = 0; x < DCT_SIZE; x += 1) {
        for (let y = 0; y < DCT_SIZE; y += 1) {
          value += matrix[u][x] * sample[(y * DCT_SIZE) + x] * matrix[v][y];
        }
      }
      lowFrequency.push(value);
    }
  }
  const phashThreshold = median(lowFrequency);
  const phash = lowFrequency.map((value) => (value > phashThreshold ? "1" : "0")).join("");

  const differenceSample = grayscaleSample(source, 9, 8);
  let dhash = "";
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      dhash += differenceSample[(y * 9) + x + 1] > differenceSample[(y * 9) + x] ? "1" : "0";
    }
  }
  return { phash, dhash, aspect: width / Math.max(height, 1) };
}

function bitSimilarity(first, second) {
  const length = Math.min(first?.length || 0, second?.length || 0);
  if (!length) return 0;
  let matches = 0;
  for (let index = 0; index < length; index += 1) {
    if (first[index] === second[index]) matches += 1;
  }
  return matches / length;
}

function visualSimilarity(first, second) {
  const perceptual = bitSimilarity(first.phash, second.phash);
  const difference = bitSimilarity(first.dhash, second.dhash);
  const aspectPenalty = 0.15 * Math.min(Math.abs(Math.log(Math.max(first.aspect, 1e-6) / Math.max(second.aspect, 1e-6))), 1);
  return clamp((0.75 * perceptual) + (0.25 * difference) - aspectPenalty);
}

function exactCorrection(memory, digest) {
  const label = memory.exact?.[digest];
  return label ? { label, exact: true, similarity: 1 } : null;
}

function visualCorrection(memory, fingerprint) {
  const candidates = (memory.visual || [])
    .map((entry) => ({ ...entry, similarity: visualSimilarity(fingerprint, entry) }))
    .sort((a, b) => b.similarity - a.similarity);
  const best = candidates[0];
  const threshold = Number(memory.threshold ?? 0.90);
  const conflictMargin = Number(memory.conflict_margin ?? 0.025);
  if (!best || best.similarity < threshold) return null;
  const conflict = candidates.some((entry) => entry.label !== best.label && entry.similarity >= best.similarity - conflictMargin);
  if (conflict) return null;
  return { label: best.label, exact: false, similarity: best.similarity };
}

function setMode(mode) {
  activeMode = mode;
  clearFile();
  selectedUrl = "";
  videoUrlInput.value = "";
  $$(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  urlPanel.hidden = mode !== "url";
  dropzone.hidden = mode === "url";
  const isVideo = mode === "video";
  fileInput.accept = isVideo ? "video/mp4,video/webm,video/quicktime,video/x-m4v" : "image/jpeg,image/png,image/webp";
  $("#drop-title").textContent = isVideo ? "Drop a video here" : "Drop an image here";
  $("#drop-copy").textContent = isVideo ? "click to browse or drop" : "click to browse, drop, or paste";
  $("#drop-limit").textContent = isVideo ? "MP4, MOV or WEBM - maximum 5 minutes" : "JPEG, PNG or WEBP - maximum 15 MB";
  analyzeButton.disabled = true;
}

function clearFile(event) {
  event?.stopPropagation();
  selectedFile = null;
  selectedSourceType = activeMode === "url" ? "url" : "file";
  fileInput.value = "";
  if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
  selectedObjectUrl = null;
  previewImage.removeAttribute("src");
  previewVideo.removeAttribute("src");
  previewImage.hidden = true;
  previewVideo.hidden = true;
  previewState.hidden = true;
  emptyState.hidden = false;
  analyzeButton.disabled = true;
  statusMessage.textContent = "";
}

function selectFile(file) {
  statusMessage.textContent = "";
  if (!file) return;
  const isImage = ["image/jpeg", "image/png", "image/webp"].includes(file.type);
  const isVideo = file.type.startsWith("video/") || /\.(mp4|m4v|mov|webm)$/i.test(file.name);
  if (activeMode === "image" && !isImage) {
    statusMessage.textContent = "Choose a JPEG, PNG or WEBP image.";
    return;
  }
  if (activeMode === "video" && !isVideo) {
    statusMessage.textContent = "Choose an MP4, MOV or WEBM video.";
    return;
  }
  if (isImage && file.size > MAX_IMAGE_BYTES) {
    statusMessage.textContent = "The image is larger than 15 MB.";
    return;
  }
  if (isVideo && file.size > MAX_VIDEO_BYTES) {
    statusMessage.textContent = `The video is larger than ${formatBytes(MAX_VIDEO_BYTES)}.`;
    return;
  }
  if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
  selectedFile = file;
  selectedUrl = "";
  selectedSourceType = "file";
  selectedObjectUrl = URL.createObjectURL(file);
  previewImage.hidden = !isImage;
  previewVideo.hidden = !isVideo;
  if (isImage) previewImage.src = selectedObjectUrl;
  if (isVideo) {
    previewVideo.src = selectedObjectUrl;
    previewVideo.load();
  }
  $("#preview-name").textContent = file.name;
  $("#preview-size").textContent = formatBytes(file.size);
  emptyState.hidden = true;
  previewState.hidden = false;
  analyzeButton.disabled = false;
}

function loadDirectUrl() {
  const value = videoUrlInput.value.trim();
  statusMessage.textContent = "";
  if (!/^https?:\/\//i.test(value)) {
    statusMessage.textContent = "Enter a full http or https video URL.";
    return;
  }
  selectedUrl = value;
  selectedFile = null;
  selectedSourceType = "url";
  analyzeButton.disabled = false;
  statusMessage.textContent = "Video link loaded. Analysis will work if the host allows browser frame access.";
}

async function loadSession() {
  if (!sessionPromise) {
    sessionPromise = fetch(MODEL_MANIFEST_URL)
      .then((response) => {
        if (!response.ok) throw new Error("Model manifest could not be loaded.");
        return response.json();
      })
      .then((manifest) => {
        statusMessage.textContent = `Loading the ${formatBytes(manifest.total_size)} vision model. The first run can take a few minutes...`;
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

function buildViewRects(width, height) {
  const views = [{ x: 0, y: 0, width, height }];
  if (height > width * 1.1) {
    const side = width;
    [0, Math.floor((height - side) / 2), height - side].forEach((y) => views.push({ x: 0, y, width: side, height: side }));
  } else if (width > height * 1.1) {
    const side = height;
    [0, Math.floor((width - side) / 2), width - side].forEach((x) => views.push({ x, y: 0, width: side, height: side }));
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

function renderView(source, view = null) {
  const sourceWidth = source.width || source.videoWidth;
  const sourceHeight = source.height || source.videoHeight;
  const rect = view || { x: 0, y: 0, width: sourceWidth, height: sourceHeight };
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, rect.x, rect.y, rect.width, rect.height, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
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
  return clamp((difference / Math.max(samples, 1)) / 48);
}

function frameDifference(first, second) {
  if (!first || !second) return 0;
  const a = first.data;
  const b = second.data;
  let total = 0;
  let samples = 0;
  for (let index = 0; index < a.length; index += 16) {
    total += Math.abs(a[index] - b[index]);
    total += Math.abs(a[index + 1] - b[index + 1]);
    total += Math.abs(a[index + 2] - b[index + 2]);
    samples += 3;
  }
  return clamp((total / Math.max(samples, 1)) / 70);
}

function aggregateProbabilities(probabilities) {
  const sorted = [...probabilities].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const medianValue = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const mean = average(probabilities);
  const variance = average(probabilities.map((value) => (value - mean) ** 2));
  const disagreement = Math.sqrt(variance);
  const aiSupport = probabilities.filter((value) => value >= 0.8).length / probabilities.length;
  const realSupport = probabilities.filter((value) => value <= 0.2).length / probabilities.length;
  const strongConflict = Math.max(...probabilities) >= 0.95 && Math.min(...probabilities) <= 0.05;

  let probability = (0.56 * medianValue) + (0.24 * mean) + (0.20 * probabilities[0]);
  if (aiSupport > 0.5) probability = Math.max(probability, 0.64 + (0.28 * aiSupport));
  else if (strongConflict) probability = 0.5 + (0.08 * (aiSupport - realSupport));
  else if (realSupport > 0.5) probability = Math.min(probability, 0.36 - (0.24 * realSupport));
  probability = clamp(probability, 0.01, 0.99);

  const support = Math.max(aiSupport, realSupport);
  let confidence = 0.52 + (0.30 * Math.abs(probability - 0.5) * 2) + (0.18 * support) - (0.34 * disagreement);
  if (strongConflict && aiSupport <= 0.5) confidence = Math.min(confidence, 0.44);
  confidence = clamp(confidence, 0.25, 0.96);
  return { probability, confidence, disagreement, support, strongConflict, median: medianValue, mean };
}

function classify(probability, confidence) {
  if (confidence < 0.48 || (probability >= 0.43 && probability <= 0.57)) return { verdict: "inconclusive", label: "Inconclusive" };
  if (probability > 0.57) return { verdict: "likely_ai", label: "Likely AI-generated / deepfake" };
  if (probability < 0.43) return { verdict: "likely_real", label: "Likely camera-captured" };
  return { verdict: "inconclusive", label: "Inconclusive" };
}

async function runModelOnSource(source, includeCrops = true) {
  const session = await loadSession();
  const width = source.width || source.videoWidth;
  const height = source.height || source.videoHeight;
  const views = includeCrops ? buildViewRects(width, height).map((view) => renderView(source, view)) : [renderView(source)];
  const input = imageViewsToTensor(views);
  const outputMap = await session.run({ [session.inputNames[0]]: input });
  const probabilities = Array.from(outputMap[session.outputNames[0]].data, (logit) => sigmoid(Number(logit)));
  return { probabilities, views };
}

function buildSearchLinks(report) {
  if (!$("#web-check").checked) return null;
  const terms = [
    report.source_url || report.file_name || "uploaded media",
    report.media_type === "video" ? "deepfake video original source" : "AI generated image source",
    report.label,
  ].filter(Boolean).join(" ");
  const encoded = encodeURIComponent(terms);
  return {
    title: "Web trace kit",
    detail: "Static browser apps cannot crawl the web automatically or bypass reverse-search engines. Use these links to check whether the same media appears elsewhere.",
    queries: [
      { label: "Google", url: `https://www.google.com/search?q=${encoded}` },
      { label: "Bing", url: `https://www.bing.com/search?q=${encoded}` },
      { label: "Yandex", url: `https://yandex.com/search/?text=${encoded}` },
      { label: "TinEye", url: "https://tineye.com/" },
    ],
  };
}

function renderWebCard(report) {
  const web = report.web_trace;
  const card = $("#web-card");
  if (!web) {
    card.hidden = true;
    card.innerHTML = "";
    return;
  }
  card.hidden = false;
  card.innerHTML = `
    <div class="verification-heading"><span>Open web check</span><h4>${escapeHtml(web.title)}</h4></div>
    <p class="verification-status">${escapeHtml(web.detail)}</p>
    <div class="link-grid">
      ${web.queries.map((query) => `<a href="${escapeHtml(query.url)}" target="_blank" rel="noreferrer">${escapeHtml(query.label)}</a>`).join("")}
    </div>
  `;
}

function renderFrameStrip(frames = []) {
  const strip = $("#frame-strip");
  if (!frames.length) {
    strip.hidden = true;
    strip.innerHTML = "";
    return;
  }
  strip.hidden = false;
  strip.innerHTML = frames.slice(0, 8).map((frame) => `
    <div><img src="${frame.thumbnail}" alt="Video frame at ${frame.time.toFixed(1)} seconds"><span>${frame.time.toFixed(1)}s &middot; ${Math.round(frame.probability * 100)}%</span></div>
  `).join("");
}

function resetFeedbackUi() {
  selectedFeedback = "";
  selectedCorrection = "";
  $("#feedback-card").hidden = !lastReport;
  $("#correction-fields").hidden = true;
  $("#feedback-submit").disabled = true;
  $("#feedback-status").textContent = "";
  $("#feedback-status").classList.remove("error");
  $("#feedback-notes").value = "";
  $$("[data-feedback], [data-correction]").forEach((button) => button.classList.remove("selected"));
}

function renderReport(report) {
  lastReport = report;
  $("#result-panel").classList.add("has-result");
  $("#result-placeholder").hidden = true;
  $("#result-content").hidden = false;
  $("#engine-badge").textContent = report.engine || "DoubleDeep SigLIP / ONNX Web";
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
  renderWebCard(report);
  renderFrameStrip(report.frames || []);
  resetFeedbackUi();
}

function renderCorrectionReport(match, digest, sourceName, mediaType) {
  const isAi = match.label === "ai";
  const confidence = match.exact ? 0.99 : Math.min(0.96, 0.82 + (0.14 * match.similarity));
  const probability = isAi ? (match.exact ? 0.99 : 0.94) : (match.exact ? 0.01 : 0.06);
  const report = {
    scan_id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
    media_type: mediaType,
    file_name: sourceName,
    verdict: isAi ? "likely_ai" : "likely_real",
    label: isAi ? "Likely AI-generated / deepfake" : "Likely camera-captured",
    ai_probability: probability,
    confidence,
    engine: "DoubleDeep correction memory",
    model: "DoubleDeep correction memory",
    runtime: "Browser SHA-256 and perceptual fingerprint matching",
    evidence: [
      {
        direction: isAi ? "ai" : "real",
        title: match.exact ? "Exact correction memory" : "Visual correction memory",
        detail: match.exact
          ? "This exact media item was previously corrected and saved in the feedback ledger."
          : `This media closely matches a previously corrected reference (${Math.round(match.similarity * 100)}% perceptual match).`,
      },
      { direction: "neutral", title: "Fast path used", detail: "The corrected label was applied before loading the large neural model." },
      { direction: "neutral", title: "On-device privacy", detail: "The media was checked in this browser and was not sent to an analysis server." },
    ],
    signals: [
      { label: "AI likelihood", value: probability },
      { label: "Camera likelihood", value: 1 - probability },
      { label: "Correction confidence", value: confidence },
      { label: "Visual match", value: match.similarity },
    ],
    image_sha256: digest,
    limitations: "Correction memory applies only to exact or very close visual matches.",
  };
  report.web_trace = buildSearchLinks(report);
  renderReport(report);
}

async function analyzeImageFile(file) {
  statusMessage.textContent = "Checking correction memory...";
  const [memory, digest] = await Promise.all([loadFeedbackMemory(), sha256Hex(file)]);
  let bitmap = await createImageBitmap(file);
  try {
    lastFingerprint = visualFingerprint(bitmap);
    const exactMatch = exactCorrection(memory, digest);
    if (exactMatch) {
      renderCorrectionReport(exactMatch, digest, file.name, "image");
      return;
    }
    const visualMatch = visualCorrection(memory, lastFingerprint);
    if (visualMatch) {
      renderCorrectionReport(visualMatch, digest, file.name, "image");
      return;
    }
    statusMessage.textContent = "Running whole-image and regional checks locally in your browser...";
    const model = await runModelOnSource(bitmap, true);
    const assessment = aggregateProbabilities(model.probabilities);
    const classification = classify(assessment.probability, assessment.confidence);
    const texture = textureComplexity(model.views[0]);
    const report = {
      scan_id: crypto.randomUUID(),
      created_at: new Date().toISOString(),
      media_type: "image",
      file_name: file.name,
      verdict: classification.verdict,
      label: classification.label,
      ai_probability: assessment.probability,
      confidence: assessment.confidence,
      engine: "DoubleDeep SigLIP / ONNX Web",
      model: "Ateeqq/ai-vs-human-image-detector",
      runtime: "ONNX Runtime Web / WebAssembly",
      evidence: [
        {
          direction: assessment.probability >= 0.5 ? "ai" : "real",
          title: assessment.probability >= 0.5 ? "Synthetic-pattern support" : "Camera-pattern support",
          detail: `The primary detector assigned ${Math.round(assessment.probability * 100)}% aggregated probability to the synthetic-image class.`,
        },
        {
          direction: assessment.strongConflict ? "neutral" : (assessment.probability >= 0.5 ? "ai" : "real"),
          title: assessment.strongConflict ? "Regional views disagree" : "Multi-view assessment",
          detail: assessment.strongConflict
            ? "Whole-image and crop-level predictions conflict, so confidence is reduced."
            : `${Math.round(assessment.support * 100)}% of whole-image and regional views provide strong support in the same direction.`,
        },
        { direction: "neutral", title: "On-device privacy", detail: "The image was processed in this browser and was not sent to an analysis server." },
      ],
      signals: [
        { label: "AI likelihood", value: assessment.probability },
        { label: "Camera likelihood", value: 1 - assessment.probability },
        { label: "Decision confidence", value: assessment.confidence },
        { label: "Model view agreement", value: assessment.support },
        { label: "Local texture complexity", value: texture },
      ],
      sha256: digest,
      fingerprint: lastFingerprint,
      limitations: "Detector estimates can be affected by unseen generators, editing, screenshots, resizing and compression.",
    };
    report.web_trace = buildSearchLinks(report);
    renderReport(report);
  } finally {
    bitmap.close();
  }
}

function createVideo(url, crossOrigin = false) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    if (crossOrigin) video.crossOrigin = "anonymous";
    video.src = url;
    const timer = setTimeout(() => reject(new Error("Video metadata timed out.")), 30000);
    video.addEventListener("loadedmetadata", () => {
      clearTimeout(timer);
      resolve(video);
    }, { once: true });
    video.addEventListener("error", () => {
      clearTimeout(timer);
      reject(new Error("The video could not be loaded by this browser."));
    }, { once: true });
    video.load();
  });
}

function seekVideo(video, time) {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      resolve();
    };
    const fail = () => {
      video.removeEventListener("error", fail);
      reject(new Error("Could not seek inside this video."));
    };
    video.addEventListener("seeked", done, { once: true });
    video.addEventListener("error", fail, { once: true });
    video.currentTime = Math.min(Math.max(time, 0), Math.max(video.duration - 0.05, 0));
  });
}

function captureThumbnail(source) {
  const canvas = document.createElement("canvas");
  canvas.width = 220;
  canvas.height = 124;
  const context = canvas.getContext("2d");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.72);
}

async function analyzeVideoSource(video, sourceName, sourceUrl = "") {
  if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error("The video duration could not be read.");
  if (video.duration > MAX_VIDEO_SECONDS + 1) throw new Error("Videos must be 5 minutes or shorter.");
  if (!video.videoWidth || !video.videoHeight) throw new Error("The video frames could not be decoded.");

  const duration = Math.min(video.duration, MAX_VIDEO_SECONDS);
  const frameCount = Math.max(6, Math.min(VIDEO_FRAME_LIMIT, Math.ceil(duration / 10)));
  const times = Array.from({ length: frameCount }, (_, index) => {
    if (frameCount === 1) return duration / 2;
    return 0.35 + ((duration - 0.7) * index / (frameCount - 1));
  });
  const digest = sourceUrl ? await sha256Text(sourceUrl) : (selectedFile ? await sha256Hex(selectedFile) : await sha256Text(sourceName));
  const memory = await loadFeedbackMemory();
  await seekVideo(video, times[0]);
  lastFingerprint = visualFingerprint(video);
  const visualMatch = visualCorrection(memory, lastFingerprint);
  if (visualMatch) {
    renderCorrectionReport(visualMatch, digest, sourceName, "video");
    return;
  }

  const frameReports = [];
  const frameProbabilities = [];
  const frameDifferences = [];
  let previousImage = null;

  for (let index = 0; index < times.length; index += 1) {
    const time = times[index];
    statusMessage.textContent = `Analyzing video frame ${index + 1} of ${times.length}...`;
    await seekVideo(video, time);
    const model = await runModelOnSource(video, index === 0);
    const probability = aggregateProbabilities(model.probabilities).probability;
    const imageData = renderView(video);
    const motion = frameDifference(previousImage, imageData);
    if (previousImage) frameDifferences.push(motion);
    previousImage = imageData;
    frameProbabilities.push(probability);
    frameReports.push({
      time,
      probability,
      motion,
      texture: textureComplexity(imageData),
      thumbnail: captureThumbnail(video),
    });
  }

  const assessment = aggregateProbabilities(frameProbabilities);
  const motionMean = average(frameDifferences);
  const motionVariance = average(frameDifferences.map((value) => (value - motionMean) ** 2));
  const motionInstability = clamp(Math.sqrt(motionVariance) * 2.2);
  const frozenSegments = frameDifferences.filter((value) => value < 0.018).length / Math.max(frameDifferences.length, 1);
  const probabilityJump = frameProbabilities.length > 1
    ? Math.max(...frameProbabilities.slice(1).map((value, index) => Math.abs(value - frameProbabilities[index])))
    : 0;
  const temporalRisk = clamp((0.38 * motionInstability) + (0.32 * frozenSegments) + (0.30 * probabilityJump));
  const adjustedProbability = clamp((0.78 * assessment.probability) + (0.22 * temporalRisk), 0.01, 0.99);
  const confidence = clamp(assessment.confidence + (0.12 * Math.abs(temporalRisk - 0.5)) - (0.18 * assessment.disagreement), 0.25, 0.95);
  const classification = classify(adjustedProbability, confidence);
  const report = {
    scan_id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
    media_type: "video",
    file_name: sourceName,
    source_url: sourceUrl,
    verdict: classification.verdict,
    label: classification.label,
    ai_probability: adjustedProbability,
    confidence,
    engine: "DoubleDeep video analyst",
    model: "SigLIP frame detector plus browser temporal forensics",
    runtime: "ONNX Runtime Web / WebAssembly",
    video: {
      duration_seconds: duration,
      width: video.videoWidth,
      height: video.videoHeight,
      sampled_frames: frameReports.length,
    },
    evidence: [
      {
        direction: adjustedProbability >= 0.5 ? "ai" : "real",
        title: "Frame-level detector consensus",
        detail: `${frameReports.length} sampled frames produced a ${Math.round(assessment.probability * 100)}% synthetic-frame likelihood before temporal adjustment.`,
      },
      {
        direction: temporalRisk > 0.55 ? "ai" : "neutral",
        title: "Temporal forensic signals",
        detail: `Motion instability ${Math.round(motionInstability * 100)}%, near-freeze segments ${Math.round(frozenSegments * 100)}%, frame-score jump ${Math.round(probabilityJump * 100)}%.`,
      },
      {
        direction: sourceUrl ? "neutral" : "real",
        title: sourceUrl ? "Direct-link limitation" : "Local file analysis",
        detail: sourceUrl
          ? "The URL was analyzed only if the host allowed cross-origin video frame reads. Blocked social pages require downloading a permitted file first."
          : "The uploaded video frames were processed in this browser and were not sent to an analysis server.",
      },
    ],
    signals: [
      { label: "AI/deepfake likelihood", value: adjustedProbability },
      { label: "Camera likelihood", value: 1 - adjustedProbability },
      { label: "Decision confidence", value: confidence },
      { label: "Frame agreement", value: assessment.support },
      { label: "Temporal risk", value: temporalRisk },
      { label: "Motion instability", value: motionInstability },
    ],
    frames: frameReports,
    sha256: digest,
    fingerprint: lastFingerprint,
    limitations: "This browser demo uses image-frame deepfake evidence and lightweight temporal checks. It does not inspect audio, camera sensor noise, container provenance, or platform-side metadata.",
  };
  report.web_trace = buildSearchLinks(report);
  renderReport(report);
}

async function analyze() {
  const isUrlMode = activeMode === "url";
  if (!selectedFile && !(isUrlMode && selectedUrl)) return;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  analyzeButton.querySelector("span").textContent = "Analyzing";
  statusMessage.textContent = "";
  try {
    if (activeMode === "image") {
      await analyzeImageFile(selectedFile);
    } else if (activeMode === "video") {
      statusMessage.textContent = "Checking local video correction memory...";
      const [memory, digest] = await Promise.all([loadFeedbackMemory(), sha256Hex(selectedFile)]);
      const exactMatch = exactCorrection(memory, digest);
      if (exactMatch) {
        renderCorrectionReport(exactMatch, digest, selectedFile.name, "video");
        return;
      }
      const url = URL.createObjectURL(selectedFile);
      try {
        const video = await createVideo(url);
        await analyzeVideoSource(video, selectedFile.name);
      } finally {
        URL.revokeObjectURL(url);
      }
    } else {
      const [memory, digest] = await Promise.all([loadFeedbackMemory(), sha256Text(selectedUrl)]);
      const exactMatch = exactCorrection(memory, digest);
      if (exactMatch) {
        renderCorrectionReport(exactMatch, digest, selectedUrl, "video");
        return;
      }
      statusMessage.textContent = "Loading remote video. The host must allow cross-origin frame reads...";
      const video = await createVideo(selectedUrl, true);
      await analyzeVideoSource(video, selectedUrl, selectedUrl);
    }
    statusMessage.textContent = "";
  } catch (error) {
    console.error(error);
    statusMessage.textContent = error.message.includes("cross-origin") || error.message.includes("tainted")
      ? "This URL blocks browser frame analysis. Download the video file or use a direct CORS-enabled MP4/WebM link."
      : (error.message || "The browser model could not complete analysis. Refresh and try again.");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("loading");
    analyzeButton.querySelector("span").textContent = "Analyze provenance";
  }
}

function saveFeedback() {
  if (!lastReport) return;
  const notes = $("#feedback-notes").value.trim();
  const inferredLabel = lastReport.ai_probability >= 0.5 ? "ai" : "real";
  const correctedLabel = selectedFeedback === "correct" ? inferredLabel : selectedCorrection;
  if (!["ai", "real", "unknown"].includes(correctedLabel)) {
    $("#feedback-status").textContent = "Choose the correct label before saving.";
    $("#feedback-status").classList.add("error");
    return;
  }
  const item = {
    id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
    source_name: lastReport.file_name || lastReport.source_url || "unknown",
    source_url: lastReport.source_url || "",
    media_type: lastReport.media_type,
    detector_label: inferredLabel,
    detector_probability: lastReport.ai_probability,
    detector_confidence: lastReport.confidence,
    feedback: selectedFeedback,
    corrected_label: correctedLabel,
    notes,
    sha256: lastReport.sha256 || lastReport.image_sha256 || "",
    fingerprint: $("#save-local-feedback").checked ? (lastReport.fingerprint || lastFingerprint) : null,
    report_id: lastReport.scan_id,
  };
  const memory = readLocalFeedback();
  memory.items = [item, ...(memory.items || []).filter((existing) => existing.sha256 !== item.sha256 || !item.sha256)].slice(0, 500);
  writeLocalFeedback(memory);
  feedbackMemoryPromise = null;
  $("#feedback-status").classList.remove("error");
  $("#feedback-status").textContent = "Saved in this browser. Future matching media can use this correction memory.";
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
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
$("#load-url").addEventListener("click", loadDirectUrl);
videoUrlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadDirectUrl();
});
analyzeButton.addEventListener("click", analyze);
$$(".mode-tab").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));

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
  if (activeMode !== "image") return;
  const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
  const pastedFile = imageItem?.getAsFile();
  if (!pastedFile) return;
  event.preventDefault();
  const extension = pastedFile.type.split("/")[1]?.replace("jpeg", "jpg") || "png";
  selectFile(new File([pastedFile], `pasted-image.${extension}`, { type: pastedFile.type }));
});

$("#download-report").addEventListener("click", () => {
  if (!lastReport) return;
  downloadJson(`doubledeep-${lastReport.scan_id}.json`, lastReport);
});

$("#export-feedback").addEventListener("click", () => {
  downloadJson(`doubledeep-feedback-${new Date().toISOString().slice(0, 10)}.json`, readLocalFeedback());
});

$$("[data-feedback]").forEach((button) => button.addEventListener("click", () => {
  selectedFeedback = button.dataset.feedback;
  $$("[data-feedback]").forEach((item) => item.classList.toggle("selected", item === button));
  $("#correction-fields").hidden = selectedFeedback !== "wrong";
  $("#feedback-submit").disabled = selectedFeedback === "wrong" && !selectedCorrection;
  if (selectedFeedback === "correct") $("#feedback-submit").disabled = false;
}));

$$("[data-correction]").forEach((button) => button.addEventListener("click", () => {
  selectedCorrection = button.dataset.correction;
  $$("[data-correction]").forEach((item) => item.classList.toggle("selected", item === button));
  $("#feedback-submit").disabled = !selectedFeedback || (selectedFeedback === "wrong" && !selectedCorrection);
}));

$("#feedback-submit").addEventListener("click", saveFeedback);
setMode("image");
