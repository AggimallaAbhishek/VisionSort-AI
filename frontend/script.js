const metaApiBase = document
  .querySelector('meta[name="visionsort-api-base-url"]')
  ?.getAttribute("content")
  ?.trim();

const apiOverrideFromQuery = new URLSearchParams(window.location.search).get("api")?.trim() || "";
if (apiOverrideFromQuery) {
  localStorage.setItem("visionsort_api_base_url", apiOverrideFromQuery);
}

const storedApiBase = localStorage.getItem("visionsort_api_base_url")?.trim() || "";
const configuredApiBase = (
  apiOverrideFromQuery || window.VISIONSORT_API_BASE_URL || metaApiBase || storedApiBase || ""
).trim();

const parseLimit = (value, fallback) => {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};

const configuredMaxFileSizeMb = parseLimit(
  document.querySelector('meta[name="visionsort-max-file-size-mb"]')?.getAttribute("content"),
  100
);
const configuredMaxFiles = parseLimit(
  document.querySelector('meta[name="visionsort-max-files"]')?.getAttribute("content"),
  50
);
const configuredBatchRequestSizeMb = parseLimit(
  document.querySelector('meta[name="visionsort-max-request-size-mb"]')?.getAttribute("content"),
  35
);
const configuredMaxBatchFiles = parseLimit(
  document.querySelector('meta[name="visionsort-max-batch-files"]')?.getAttribute("content"),
  10
);

const DEFAULT_DEPLOYED_API_BASE_URL = "https://visionsort-ai.onrender.com";
const MAX_FILE_SIZE_MB = configuredMaxFileSizeMb;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const MAX_FILES = configuredMaxFiles;
const MAX_REQUEST_SIZE_MB = configuredBatchRequestSizeMb;
const MAX_REQUEST_SIZE_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024;
const MAX_BATCH_FILES = configuredMaxBatchFiles;
const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/pjpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/bmp",
  "image/tiff",
]);
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "jfif", "png", "webp", "gif", "bmp", "tif", "tiff"]);
const CATEGORIES = ["good", "blurry", "dark", "overexposed", "duplicates"];
const CATEGORY_LABELS = {
  good: "Good",
  blurry: "Blurry",
  dark: "Dark",
  overexposed: "Overexposed",
  duplicates: "Duplicates",
};
const ASYNC_POLL_INTERVAL_MS = 700;
const ASYNC_TIMEOUT_MS = 8 * 60 * 1000;
const ASYNC_STATUS_DISCOVERY_TIMEOUT_MS = 18 * 1000;
const ASYNC_QUEUE_STALL_TIMEOUT_MS = 90 * 1000;

const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const cancelBtn = document.getElementById("cancelBtn");
const clearQueueBtn = document.getElementById("clearQueueBtn");
const queueList = document.getElementById("queueList");
const queueCount = document.getElementById("queueCount");
const statusText = document.getElementById("statusText");
const resultGrid = document.getElementById("resultGrid");
const errorBanner = document.getElementById("errorBanner");
const chipRow = document.getElementById("chipRow");

const metricSelected = document.getElementById("metricSelected");
const metricGood = document.getElementById("metricGood");
const metricIssues = document.getElementById("metricIssues");
const metricScore = document.getElementById("metricScore");
const metricScoreFill = document.getElementById("metricScoreFill");
const downloadAllZipBtn = document.getElementById("downloadAllZipBtn");

const chipAll = document.getElementById("chipAll");
const chipGood = document.getElementById("chipGood");
const chipBlurry = document.getElementById("chipBlurry");
const chipDark = document.getElementById("chipDark");
const chipOverexposed = document.getElementById("chipOverexposed");
const chipDuplicates = document.getElementById("chipDuplicates");
const fileSizeLimitBadge = document.getElementById("fileSizeLimitBadge");

const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const progressPercent = document.getElementById("progressPercent");
const progressMeta = document.getElementById("progressMeta");
const progressLabel = document.getElementById("progressLabel");
const progressTime = document.getElementById("progressTime");

let selectedFiles = [];
let currentFilter = "all";
let lastResultsByCategory = createEmptyResults();
let isUploading = false;
let pseudoProgressTimer = null;
let pseudoProgressValue = 0;
let analysisStartedAtMs = 0;
let analysisCompletedAtMs = 0;
let isPreparingZip = false;

function createEmptyResults() {
  return {
    good: [],
    blurry: [],
    dark: [],
    overexposed: [],
    duplicates: [],
  };
}

function normalizeApiBase(base) {
  return base.replace(/\/+$/, "");
}

function isRelativePath(base) {
  return base.startsWith("/");
}

function buildApiCandidates() {
  const candidates = [];
  const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);

  if (configuredApiBase) {
    if (isLocalHost && isRelativePath(configuredApiBase)) {
      candidates.push("http://localhost:10000");
    } else {
      candidates.push(configuredApiBase);
    }
  }

  if (isLocalHost) {
    candidates.push("http://localhost:10000");
  } else {
    candidates.push(DEFAULT_DEPLOYED_API_BASE_URL);
    candidates.push("/api");
  }

  return dedupeUrls(candidates);
}

const API_BASE_CANDIDATES = buildApiCandidates();

function dedupeUrls(urls) {
  const seen = new Set();
  const unique = [];

  urls.forEach((url) => {
    const normalized = normalizeApiBase(url || "");
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    unique.push(normalized);
  });

  return unique;
}

function buildEndpointCandidates(apiBases, suffixPath) {
  const endpoints = [];

  apiBases.forEach((base) => {
    const normalizedBase = normalizeApiBase(base);

    if (normalizedBase === "/api") {
      endpoints.push(`/api${suffixPath}`);
      return;
    }

    if (normalizedBase.endsWith("/api")) {
      endpoints.push(`${normalizedBase}${suffixPath}`);
      const baseWithoutApi = normalizedBase.slice(0, -4);
      if (baseWithoutApi) {
        endpoints.push(`${baseWithoutApi}${suffixPath}`);
      }
      return;
    }

    endpoints.push(`${normalizedBase}${suffixPath}`);
    endpoints.push(`${normalizedBase}/api${suffixPath}`);
  });

  return dedupeUrls(endpoints);
}

function prioritizeStoredEndpoint(endpoints, storageKey) {
  const storedEndpoint = localStorage.getItem(storageKey)?.trim() || "";
  if (!storedEndpoint) {
    return endpoints;
  }

  const normalizedStored = normalizeApiBase(storedEndpoint);
  if (!endpoints.includes(normalizedStored)) {
    return endpoints;
  }

  return [normalizedStored, ...endpoints.filter((endpoint) => endpoint !== normalizedStored)];
}

const API_UPLOAD_ENDPOINT_CANDIDATES = prioritizeStoredEndpoint(
  buildEndpointCandidates(API_BASE_CANDIDATES, "/upload"),
  "visionsort_api_upload_endpoint"
);
const API_ASYNC_UPLOAD_ENDPOINT_CANDIDATES = prioritizeStoredEndpoint(
  buildEndpointCandidates(API_BASE_CANDIDATES, "/upload/async"),
  "visionsort_api_async_upload_endpoint"
);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function safeFileName(name, fallback) {
  const cleaned = String(name || "")
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/\s+/g, "_")
    .trim();
  return cleaned || fallback;
}

function dataUrlToBase64Payload(dataUrl) {
  const raw = String(dataUrl || "");
  const match = raw.match(/^data:[^;]+;base64,(.+)$/);
  return match ? match[1] : "";
}

function makeZipTimestamp() {
  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  const hh = String(now.getHours()).padStart(2, "0");
  const mi = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

function bytesToSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 MB";
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function totalBytes(files) {
  return files.reduce((sum, file) => sum + (Number(file.size) || 0), 0);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function hasAnyResults(results) {
  return CATEGORIES.some((key) => Array.isArray(results[key]) && results[key].length > 0);
}

function mergeResults(target, incoming) {
  CATEGORIES.forEach((category) => {
    const sourceItems = Array.isArray(incoming[category]) ? incoming[category] : [];
    target[category].push(...sourceItems);
  });
  return target;
}

function buildUploadBatches(files) {
  const batches = [];
  let currentBatch = [];
  let currentBatchBytes = 0;

  files.forEach((file) => {
    const fileBytes = Number(file.size) || 0;
    const wouldExceedFileCount = currentBatch.length >= MAX_BATCH_FILES;
    const wouldExceedRequestSize = currentBatchBytes + fileBytes > MAX_REQUEST_SIZE_BYTES;

    if (currentBatch.length > 0 && (wouldExceedFileCount || wouldExceedRequestSize)) {
      batches.push(currentBatch);
      currentBatch = [];
      currentBatchBytes = 0;
    }

    currentBatch.push(file);
    currentBatchBytes += fileBytes;
  });

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function formatElapsedDuration(ms) {
  const safeMs = Math.max(0, Number(ms || 0));
  const seconds = safeMs / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
}

function startAnalysisClock() {
  analysisStartedAtMs = Date.now();
  analysisCompletedAtMs = 0;
  if (progressTime) {
    progressTime.textContent = "Elapsed: 0.0s";
  }
}

function updateElapsedTimeLabel() {
  if (!progressTime || !analysisStartedAtMs || analysisCompletedAtMs) {
    return;
  }

  progressTime.textContent = `Elapsed: ${formatElapsedDuration(Date.now() - analysisStartedAtMs)}`;
}

function finishAnalysisClock(success) {
  if (!analysisStartedAtMs) {
    return "0.0s";
  }

  analysisCompletedAtMs = Date.now();
  const elapsed = formatElapsedDuration(analysisCompletedAtMs - analysisStartedAtMs);

  if (progressTime) {
    progressTime.textContent = success ? `Processed in ${elapsed}` : `Failed after ${elapsed}`;
  }

  return elapsed;
}

function hasAllowedImageType(file) {
  const mimeType = (file.type || "").toLowerCase();
  if (mimeType && ALLOWED_TYPES.has(mimeType)) {
    return true;
  }

  const fileName = (file.name || "").toLowerCase();
  const extension = fileName.includes(".") ? fileName.split(".").pop() : "";
  return ALLOWED_EXTENSIONS.has(extension || "");
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

function setStatus(text) {
  statusText.textContent = text;
}

function requestIdSuffix(requestId) {
  return requestId ? ` [req:${String(requestId).slice(0, 8)}]` : "";
}

function logRequestTrace(action, endpoint, requestId, extra = "") {
  const requestPart = requestId ? ` request_id=${requestId}` : "";
  const extraPart = extra ? ` ${extra}` : "";
  console.info(`[VisionSort][${action}] ${endpoint}${requestPart}${extraPart}`);
}

function setProgress(percent, labelText, metaText) {
  if (!progressWrap || !progressFill || !progressPercent || !progressMeta || !progressLabel) {
    return;
  }

  const safePercent = clamp(Math.round(percent), 0, 100);
  progressWrap.classList.remove("hidden");
  progressFill.style.width = `${safePercent}%`;
  progressPercent.textContent = `${safePercent}%`;
  progressLabel.textContent = labelText;
  progressMeta.textContent = metaText;
  updateElapsedTimeLabel();
}

function resetProgress() {
  if (!progressWrap || !progressFill || !progressPercent || !progressMeta || !progressLabel) {
    return;
  }

  analysisStartedAtMs = 0;
  analysisCompletedAtMs = 0;
  progressWrap.classList.add("hidden");
  progressFill.style.width = "0%";
  progressPercent.textContent = "0%";
  progressLabel.textContent = "Analysis progress";
  progressMeta.textContent = "Waiting to start...";
  if (progressTime) {
    progressTime.textContent = "Elapsed: 0.0s";
  }
}

function startPseudoProgress() {
  stopPseudoProgress();
  pseudoProgressValue = 7;
  setProgress(pseudoProgressValue, "Analysis progress", "Analyzing images...");

  pseudoProgressTimer = window.setInterval(() => {
    pseudoProgressValue = Math.min(94, pseudoProgressValue + Math.max(1, Math.ceil((94 - pseudoProgressValue) * 0.12)));
    setProgress(pseudoProgressValue, "Analysis progress", "Analyzing images...");
  }, 480);
}

function stopPseudoProgress(finalPercent = null, finalMeta = "") {
  if (pseudoProgressTimer) {
    window.clearInterval(pseudoProgressTimer);
    pseudoProgressTimer = null;
  }

  if (Number.isFinite(finalPercent)) {
    setProgress(Number(finalPercent), "Analysis progress", finalMeta || "Analysis complete.");
  }
}

function setUploadingState(uploading) {
  isUploading = uploading;
  analyzeBtn.disabled = uploading || selectedFiles.length === 0;
  cancelBtn.disabled = uploading;
  clearQueueBtn.disabled = uploading || selectedFiles.length === 0;
  imageInput.disabled = uploading;
  dropZone.setAttribute("aria-busy", uploading ? "true" : "false");

  if (uploading) {
    setStatus("Analyzing images... This can take a few seconds.");
    analyzeBtn.textContent = "Analyzing...";
  } else {
    analyzeBtn.textContent = "Analyze Images";
  }

  updateDownloadControls();
}

function updateQueueMetrics(preserveStatus = false) {
  const count = selectedFiles.length;
  queueCount.textContent = `${count} Selected`;
  metricSelected.textContent = String(count);

  if (!isUploading && !preserveStatus) {
    const apiHint = API_ASYNC_UPLOAD_ENDPOINT_CANDIDATES[0] || API_UPLOAD_ENDPOINT_CANDIDATES[0] || "not configured";
    setStatus(count ? "Queue ready. Click Analyze Images." : `Ready for analysis. API: ${apiHint}`);
  }

  analyzeBtn.disabled = isUploading || count === 0;
  clearQueueBtn.disabled = isUploading || count === 0;
}

function renderQueue() {
  queueList.innerHTML = "";

  if (!selectedFiles.length) {
    const li = document.createElement("li");
    li.className = "queue-item";
    li.innerHTML = `
      <div class="queue-item-main">
        <strong>No files in queue</strong>
        <p class="queue-meta">Drag images into the drop zone or click to browse.</p>
      </div>
    `;
    queueList.appendChild(li);
    return;
  }

  selectedFiles.forEach((file, index) => {
    const safeName = escapeHtml(file.name);
    const li = document.createElement("li");
    li.className = "queue-item";
    li.innerHTML = `
      <div class="queue-item-main">
        <strong>${safeName}</strong>
        <p class="queue-meta">${bytesToSize(file.size)} • Ready</p>
      </div>
      <button class="queue-delete" type="button" aria-label="Remove ${safeName}" data-index="${index}">
        ✕
      </button>
    `;
    queueList.appendChild(li);
  });
}

function setFilter(filterKey) {
  currentFilter = filterKey;
  chipRow.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.filter === filterKey);
  });
  renderResultCards();
}

function updateSummaryChips(results) {
  const counts = {
    good: results.good.length,
    blurry: results.blurry.length,
    dark: results.dark.length,
    overexposed: results.overexposed.length,
    duplicates: results.duplicates.length,
  };

  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  const issues = total - counts.good;
  const score = total ? Math.round((counts.good / total) * 100) : 0;

  chipAll.textContent = String(total);
  chipGood.textContent = String(counts.good);
  chipBlurry.textContent = String(counts.blurry);
  chipDark.textContent = String(counts.dark);
  chipOverexposed.textContent = String(counts.overexposed);
  chipDuplicates.textContent = String(counts.duplicates);

  metricGood.textContent = String(counts.good);
  metricIssues.textContent = String(issues);
  metricScore.textContent = `${score}%`;
  if (metricScoreFill) {
    metricScoreFill.style.width = `${clamp(score, 0, 100)}%`;
  }
}

function flattenResults(results) {
  const items = [];
  CATEGORIES.forEach((category) => {
    (results[category] || []).forEach((item) => {
      items.push({ ...item, __category: category });
    });
  });
  return items;
}

function computeBlurQualityScore(item) {
  const providedScore = Number(item.blur_quality_score);
  if (Number.isFinite(providedScore)) {
    return clamp(providedScore, 0, 100);
  }

  const rawBlur = Number(item.blur_score || 0);
  return clamp((rawBlur / 300) * 100, 0, 100);
}

function computeBrightnessQualityScore(item) {
  const providedScore = Number(item.brightness_score);
  if (Number.isFinite(providedScore)) {
    return clamp(providedScore, 0, 100);
  }

  const brightnessValue = Number(item.brightness_value);
  if (Number.isFinite(brightnessValue)) {
    if (brightnessValue < 50) {
      return clamp((brightnessValue / 50) * 69, 0, 100);
    }
    if (brightnessValue > 200) {
      return clamp(((255 - brightnessValue) / 55) * 69, 0, 100);
    }
    const center = 125;
    const normalHalfSpan = 75;
    const normalizedDistance = Math.abs(brightnessValue - center) / normalHalfSpan;
    return clamp(70 + (1 - normalizedDistance) * 30, 0, 100);
  }

  const level = String(item.brightness_level || "normal").toLowerCase();
  if (level === "dark" || level === "overexposed") {
    return 35;
  }
  return 82;
}

function shortenPath(path, maxLength = 62) {
  const value = String(path || "");
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 3)}...`;
}

function resolveStorageFolder(item) {
  if (item.storage_folder) {
    return String(item.storage_folder);
  }

  const path = String(item.storage_path || "");
  if (path.startsWith("s3://")) {
    const withoutScheme = path.replace(/^s3:\/\/[^/]+\//, "");
    const segments = withoutScheme.split("/");
    if (segments.length >= 2) {
      return `${segments[0]}/${segments[1]}`;
    }
    return withoutScheme;
  }
  return "n/a";
}

function getVisibleCategories() {
  if (currentFilter === "all") {
    return [...CATEGORIES];
  }
  return CATEGORIES.includes(currentFilter) ? [currentFilter] : [...CATEGORIES];
}

function buildResultCardElement(item, categoryKey) {
  const blurPercent = clamp(computeBlurQualityScore(item), 0, 100);
  const brightnessPercent = clamp(computeBrightnessQualityScore(item), 0, 100);
  const renamedName = item.renamed_file_name || item.file_name || "processed_image.jpg";
  const originalName = item.original_file_name || item.file_name || "unknown";
  const safeName = escapeHtml(renamedName);
  const safeOriginal = escapeHtml(originalName);
  const safeCategory = escapeHtml(categoryKey);
  const safeLabel = escapeHtml(item.ai_label || "model_unavailable");
  const aiConfidenceRaw = Number(item.ai_confidence);
  const aiConfidenceLabel = Number.isFinite(aiConfidenceRaw) ? `${aiConfidenceRaw.toFixed(1)}%` : "n/a";
  const statusSource = escapeHtml(item.status_source || "rule");
  const safeBrightness = escapeHtml(item.brightness_level || "normal");
  const safeStorage = escapeHtml(shortenPath(item.storage_path || "n/a"));
  const safeProcessedStorage = escapeHtml(shortenPath(item.processed_storage_path || "n/a"));
  const fullFolderPath = resolveStorageFolder(item);
  const safeFolder = escapeHtml(shortenPath(fullFolderPath, 48));
  const safeFolderFull = escapeHtml(fullFolderPath);
  const blurRaw = Number(item.blur_score || 0);
  const brightnessRaw = Number(item.brightness_value);
  const brightnessRawLabel = Number.isFinite(brightnessRaw) ? `${brightnessRaw.toFixed(1)}/255` : "n/a";
  const imageSource = item.preview_data_url || "";

  const card = document.createElement("article");
  card.className = "result-card";
  card.innerHTML = `
    <div class="card-image-wrap">
      <img src="${imageSource}" alt="${safeName}" loading="lazy" />
      <span class="status-badge ${safeCategory}">${safeCategory}</span>
    </div>
    <div class="card-body">
      <h3 class="card-title">${safeName}</h3>
      <p class="card-subtitle">Original: ${safeOriginal}</p>
      <span class="folder-pill" title="${safeFolderFull}">Folder: ${safeFolder}</span>

      <div class="metric-line">
        <div class="label-row"><span>Blur Quality</span><strong>${blurPercent.toFixed(1)}%</strong></div>
        <div class="progress blur"><span style="width:${blurPercent.toFixed(2)}%"></span></div>
        <p class="metric-caption">Laplacian variance: ${blurRaw.toFixed(2)}</p>
      </div>

      <div class="metric-line">
        <div class="label-row"><span>Brightness Quality</span><strong>${brightnessPercent.toFixed(1)}%</strong></div>
        <div class="progress brightness"><span style="width:${brightnessPercent.toFixed(2)}%"></span></div>
        <p class="metric-caption">HSV mean: ${brightnessRawLabel} (${safeBrightness})</p>
      </div>

      <p class="card-subtitle">AI label: ${safeLabel} (${aiConfidenceLabel})</p>
      <p class="metric-caption">Decision source: ${statusSource}</p>
      <p class="footer-meta">Original object: ${safeStorage}<br />Processed object: ${safeProcessedStorage}</p>
    </div>
  `;

  return card;
}

function createFolderSection(categoryKey, items) {
  const section = document.createElement("section");
  section.className = "folder-section";
  section.dataset.category = categoryKey;
  const categoryLabel = CATEGORY_LABELS[categoryKey] || categoryKey;

  section.innerHTML = `
    <div class="folder-head">
      <div class="folder-title">
        <h3>${escapeHtml(categoryLabel)} Folder</h3>
        <span class="folder-count">${items.length} file${items.length === 1 ? "" : "s"}</span>
      </div>
      <button
        class="ghost-btn folder-download-btn"
        type="button"
        data-download-category="${categoryKey}"
        ${items.length ? "" : "disabled"}
      >
        Download ${escapeHtml(categoryLabel)}.zip
      </button>
    </div>
    <div class="folder-body"></div>
  `;

  const body = section.querySelector(".folder-body");
  if (!items.length) {
    const empty = document.createElement("article");
    empty.className = "folder-empty";
    empty.textContent = "No images in this folder.";
    body.appendChild(empty);
  } else {
    items.forEach((item) => {
      body.appendChild(buildResultCardElement(item, categoryKey));
    });
  }

  return section;
}

function updateDownloadControls() {
  const hasResults = hasAnyResults(lastResultsByCategory);
  if (downloadAllZipBtn) {
    downloadAllZipBtn.disabled = !hasResults || isUploading || isPreparingZip;
    downloadAllZipBtn.textContent = isPreparingZip
      ? "Preparing ZIP..."
      : "Download All Folders (.zip)";
  }

  document.querySelectorAll(".folder-download-btn").forEach((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }

    const category = button.dataset.downloadCategory || "";
    const count = Array.isArray(lastResultsByCategory[category]) ? lastResultsByCategory[category].length : 0;
    button.disabled = !count || isUploading || isPreparingZip;
  });
}

function renderResultCards() {
  resultGrid.innerHTML = "";
  const categoriesToRender = getVisibleCategories();
  const total = flattenResults(lastResultsByCategory).length;

  if (!total) {
    const empty = document.createElement("article");
    empty.className = "result-empty";
    empty.textContent = "No processed images yet. Upload files to start analysis.";
    resultGrid.appendChild(empty);
    updateDownloadControls();
    return;
  }

  categoriesToRender.forEach((categoryKey) => {
    const items = Array.isArray(lastResultsByCategory[categoryKey]) ? lastResultsByCategory[categoryKey] : [];
    resultGrid.appendChild(createFolderSection(categoryKey, items));
  });

  updateDownloadControls();
}

function addFilesToQueue(fileList) {
  const additions = [];
  const issues = [];

  Array.from(fileList).forEach((file) => {
    if (selectedFiles.length + additions.length >= MAX_FILES) {
      issues.push(`Skipped ${file.name}: max ${MAX_FILES} files per batch.`);
      return;
    }

    const duplicate = selectedFiles.some((selected) => selected.name === file.name && selected.size === file.size);
    if (duplicate) {
      return;
    }

    if (!hasAllowedImageType(file)) {
      issues.push(`Skipped ${file.name}: unsupported type.`);
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      issues.push(`Skipped ${file.name}: larger than ${MAX_FILE_SIZE_MB}MB.`);
      return;
    }

    additions.push(file);
  });

  if (issues.length) {
    showError(issues.join(" "));
  } else {
    clearError();
  }

  if (additions.length) {
    selectedFiles = [...selectedFiles, ...additions];
    renderQueue();
    updateQueueMetrics();
    resetProgress();
  }
}

function clearQueue(options = {}) {
  const { resetProgressUI = true, preserveStatus = false } = options;
  if (isUploading) {
    return;
  }
  selectedFiles = [];
  imageInput.value = "";
  renderQueue();
  updateQueueMetrics(preserveStatus);
  if (resetProgressUI) {
    resetProgress();
  }
}

function buildFormData(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return formData;
}

function normalizeResults(payload) {
  return {
    good: Array.isArray(payload.good) ? payload.good : [],
    blurry: Array.isArray(payload.blurry) ? payload.blurry : [],
    dark: Array.isArray(payload.dark) ? payload.dark : [],
    overexposed: Array.isArray(payload.overexposed) ? payload.overexposed : [],
    duplicates: Array.isArray(payload.duplicates) ? payload.duplicates : [],
  };
}

function downloadBlob(blob, filename) {
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(href);
}

function categoryItems(categoryKey) {
  return Array.isArray(lastResultsByCategory[categoryKey]) ? lastResultsByCategory[categoryKey] : [];
}

async function buildZipForCategories(categories, onProgress) {
  if (!window.JSZip) {
    throw new Error("ZIP library failed to load. Refresh the page and try again.");
  }

  const zip = new window.JSZip();
  const generatedAt = new Date().toISOString();
  const manifest = {
    generated_at: generatedAt,
    categories: {},
  };

  categories.forEach((categoryKey) => {
    const folder = zip.folder(categoryKey);
    const items = categoryItems(categoryKey);
    manifest.categories[categoryKey] = [];

    items.forEach((item, index) => {
      const renamedName = safeFileName(
        item.renamed_file_name || item.file_name,
        `${categoryKey}_${String(index + 1).padStart(3, "0")}.jpg`
      );
      const base64Payload = dataUrlToBase64Payload(item.preview_data_url);
      if (base64Payload) {
        folder.file(renamedName, base64Payload, { base64: true });
      }

      manifest.categories[categoryKey].push({
        file_name: item.file_name || null,
        renamed_file_name: renamedName,
        final_status: item.final_status || categoryKey,
        blur_score: item.blur_score ?? null,
        blur_quality_score: item.blur_quality_score ?? null,
        brightness_level: item.brightness_level ?? null,
        brightness_value: item.brightness_value ?? null,
        brightness_score: item.brightness_score ?? null,
        ai_label: item.ai_label ?? null,
        ai_confidence: item.ai_confidence ?? null,
        storage_path: item.storage_path ?? null,
        processed_storage_path: item.processed_storage_path ?? null,
      });
    });

    if (!items.length) {
      folder.file("README.txt", "No analyzed images in this category for this run.");
    }
  });

  zip.file("manifest.json", JSON.stringify(manifest, null, 2));

  return zip.generateAsync(
    {
      type: "blob",
      compression: "DEFLATE",
      compressionOptions: { level: 6 },
    },
    (meta) => {
      if (typeof onProgress === "function") {
        onProgress(meta);
      }
    }
  );
}

async function downloadResultsZip(categories, filenamePrefix) {
  if (isPreparingZip) {
    return;
  }

  const normalizedCategories = categories.filter((category) => CATEGORIES.includes(category));
  if (!normalizedCategories.length) {
    showError("No valid categories selected for ZIP download.");
    return;
  }

  isPreparingZip = true;
  clearError();
  updateDownloadControls();

  try {
    setStatus("Preparing ZIP file for download...");
    const zipBlob = await buildZipForCategories(normalizedCategories, (meta) => {
      const percent = clamp(Math.round(Number(meta.percent || 0)), 0, 100);
      setStatus(`Preparing ZIP... ${percent}%`);
    });

    const fileName = `${safeFileName(filenamePrefix, "visionsort_results")}_${makeZipTimestamp()}.zip`;
    downloadBlob(zipBlob, fileName);
    setStatus(`ZIP downloaded: ${fileName}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : "ZIP generation failed.";
    showError(message);
    setStatus("ZIP download failed.");
  } finally {
    isPreparingZip = false;
    updateDownloadControls();
  }
}

async function getResponseErrorMessage(response) {
  const responseRequestId = response.headers.get("x-request-id") || "";
  const raw = await response.text();
  if (!raw) {
    return `Upload failed (${response.status}).${requestIdSuffix(responseRequestId)}`;
  }

  try {
    const payload = JSON.parse(raw);
    if (payload?.error && typeof payload.error.message === "string") {
      const errorCode = typeof payload.error.code === "string" ? `${payload.error.code}: ` : "";
      const payloadRequestId = typeof payload.error.request_id === "string" ? payload.error.request_id : "";
      return `Upload failed (${response.status}): ${errorCode}${payload.error.message}${requestIdSuffix(payloadRequestId || responseRequestId)}`;
    }

    if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
      return `Upload failed (${response.status}): ${payload.detail}${requestIdSuffix(responseRequestId)}`;
    }
  } catch {
    // Fall through to raw text output.
  }

  const snippet = raw.length > 220 ? `${raw.slice(0, 220)}...` : raw;
  return `Upload failed (${response.status}): ${snippet}${requestIdSuffix(responseRequestId)}`;
}

async function attemptUpload(endpoint, files) {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: buildFormData(files),
    });
    const requestId = response.headers.get("x-request-id") || "";

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        endpoint,
        requestId,
        error: await getResponseErrorMessage(response),
      };
    }

    const payload = await response.json();
    logRequestTrace("upload", endpoint, requestId, `status=${response.status}`);
    return {
      ok: true,
      endpoint,
      requestId,
      data: payload,
    };
  } catch (error) {
    let message = error instanceof Error ? error.message : "Unexpected network error.";
    if (message === "Failed to fetch") {
      message = "Failed to fetch (network/CORS/request size/backend restart).";
    }
    return {
      ok: false,
      status: 0,
      endpoint,
      requestId: "",
      error: message,
    };
  }
}

async function fetchJobSnapshot(endpoint) {
  try {
    const response = await fetch(endpoint, {
      method: "GET",
      cache: "no-store",
    });
    const requestId = response.headers.get("x-request-id") || "";

    if (!response.ok) {
      return {
        ok: false,
        endpoint,
        requestId,
        error: await getResponseErrorMessage(response),
      };
    }

    const payload = await response.json();
    return {
      ok: true,
      endpoint,
      requestId,
      data: payload,
    };
  } catch (error) {
    return {
      ok: false,
      endpoint,
      requestId: "",
      error: error instanceof Error ? error.message : "Unexpected job polling error.",
    };
  }
}

function buildStatusEndpointCandidates(uploadEndpoint, payload) {
  const candidates = [];
  const jobId = payload?.job_id ? String(payload.job_id) : "";
  let uploadBase = "";

  try {
    uploadBase = new URL(uploadEndpoint, window.location.origin).origin;
  } catch {
    uploadBase = window.location.origin;
  }

  function addCandidate(pathOrUrl) {
    if (!pathOrUrl || typeof pathOrUrl !== "string") {
      return;
    }
    const trimmed = pathOrUrl.trim();
    if (!trimmed) {
      return;
    }

    candidates.push(trimmed);
    if (trimmed.startsWith("/")) {
      candidates.push(`${uploadBase}${trimmed}`);
    }
  }

  if (typeof payload?.api_status_endpoint === "string" && payload.api_status_endpoint.trim()) {
    addCandidate(payload.api_status_endpoint);
  }
  if (typeof payload?.status_endpoint === "string" && payload.status_endpoint.trim()) {
    addCandidate(payload.status_endpoint);
  }

  if (jobId) {
    const base = normalizeApiBase(uploadEndpoint.replace(/\/upload\/async$/, ""));
    if (base) {
      candidates.push(`${base}/jobs/${encodeURIComponent(jobId)}`);
      candidates.push(`${base}/api/jobs/${encodeURIComponent(jobId)}`);
    }
  }

  return dedupeUrls(candidates);
}

function isResultPayload(payload) {
  return payload && typeof payload === "object" && CATEGORIES.every((key) => Array.isArray(payload[key]));
}

async function pollJobUntilDone(statusEndpointCandidates) {
  const startedAt = Date.now();
  let activeEndpoint = "";
  let lastError = "Unable to fetch job status.";
  let firstPendingMessageShown = false;
  let queuedSinceMs = 0;
  let lastRequestId = "";

  while (Date.now() - startedAt < ASYNC_TIMEOUT_MS) {
    const probeEndpoints = activeEndpoint
      ? [activeEndpoint, ...statusEndpointCandidates.filter((endpoint) => endpoint !== activeEndpoint)]
      : statusEndpointCandidates;
    let snapshot = null;

    for (const endpoint of probeEndpoints) {
      const result = await fetchJobSnapshot(endpoint);
      if (result.ok) {
        snapshot = result;
        activeEndpoint = endpoint;
        lastRequestId = result.requestId || lastRequestId;
        logRequestTrace("job-status", endpoint, result.requestId);
        break;
      }
      lastError = `${endpoint} -> ${result.error}`;
    }

    if (!snapshot) {
      const elapsedSinceStart = Date.now() - startedAt;
      if (elapsedSinceStart > ASYNC_STATUS_DISCOVERY_TIMEOUT_MS) {
        throw new Error(`Could not reach job status endpoint. ${lastError}`);
      }

      if (!firstPendingMessageShown) {
        setProgress(12, "Queued", "Waiting for analysis worker to start...");
        firstPendingMessageShown = true;
      }
      await sleep(ASYNC_POLL_INTERVAL_MS);
      continue;
    }

    const job = snapshot.data || {};
    const total = Math.max(0, Number(job.total_files || selectedFiles.length || 0));
    const processed = Math.max(0, Number(job.processed_files || 0));
    const phase = String(job.phase || job.status || "processing");

    if (phase === "finalizing") {
      setProgress(99, "Finalizing", `Processed ${processed}/${total} files. Saving outputs...`);
    } else if (job.status === "completed") {
      setProgress(100, "Completed", `Processed ${processed || total}/${total || processed} files.`);
    } else {
      const percent = total ? Math.round((processed / total) * 100) : 0;
      setProgress(percent, "Analysis progress", `Processed ${processed}/${total} files.`);
    }

    setStatus(typeof job.message === "string" && job.message ? job.message : "Analyzing images...");

    if (job.status === "completed") {
      return {
        endpoint: activeEndpoint,
        requestId: lastRequestId,
        job,
      };
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Background job failed.");
    }

    if (job.status === "queued") {
      if (!queuedSinceMs) {
        queuedSinceMs = Date.now();
      }
      if (Date.now() - queuedSinceMs > ASYNC_QUEUE_STALL_TIMEOUT_MS) {
        throw new Error("Async job stayed queued too long. Falling back to direct upload.");
      }
    } else {
      queuedSinceMs = 0;
    }

    await sleep(ASYNC_POLL_INTERVAL_MS);
  }

  throw new Error(`Timed out while waiting for analysis. Last status error: ${lastError}`);
}

async function runAsyncUploadFlow(files) {
  const failures = [];

  for (const endpoint of API_ASYNC_UPLOAD_ENDPOINT_CANDIDATES) {
    setProgress(4, "Queueing", "Sending files to async analysis endpoint...");
    const startResult = await attemptUpload(endpoint, files);

    if (!startResult.ok) {
      failures.push(`${endpoint} -> ${startResult.error}`);
      if (startResult.status >= 400 && startResult.status < 500 && startResult.status !== 404) {
        break;
      }
      continue;
    }

    if (isResultPayload(startResult.data)) {
      localStorage.setItem("visionsort_api_upload_endpoint", endpoint.replace(/\/async$/, ""));
      return {
        ok: true,
        mode: "sync-response",
        endpoint,
        requestId: startResult.requestId || "",
        results: normalizeResults(startResult.data),
      };
    }

    const statusCandidates = buildStatusEndpointCandidates(endpoint, startResult.data);
    if (!statusCandidates.length) {
      failures.push(`${endpoint} -> Missing status endpoint in async response.`);
      continue;
    }

    try {
      localStorage.setItem("visionsort_api_async_upload_endpoint", endpoint);
      const polled = await pollJobUntilDone(statusCandidates);
      if (!isResultPayload(polled.job.results || {})) {
        throw new Error("Async job completed without valid categorized results.");
      }

      return {
        ok: true,
        mode: "async",
        endpoint,
        statusEndpoint: polled.endpoint,
        requestId: polled.requestId || startResult.requestId || "",
        results: normalizeResults(polled.job.results),
      };
    } catch (error) {
      failures.push(`${endpoint} -> ${error instanceof Error ? error.message : "Unknown async polling error."}`);
    }
  }

  return {
    ok: false,
    failures,
  };
}

async function runSyncUploadFlow(files) {
  const failures = [];
  startPseudoProgress();

  for (const endpoint of API_UPLOAD_ENDPOINT_CANDIDATES) {
    const result = await attemptUpload(endpoint, files);
    if (result.ok) {
      stopPseudoProgress(100, "Analysis complete.");
      localStorage.setItem("visionsort_api_upload_endpoint", result.endpoint);
      return {
        ok: true,
        endpoint,
        requestId: result.requestId || "",
        results: normalizeResults(result.data),
      };
    }

    failures.push(`${result.endpoint} -> ${result.error}`);
    if (result.status >= 400 && result.status < 500 && result.status !== 404) {
      break;
    }
  }

  stopPseudoProgress(0, "Analysis failed.");
  return {
    ok: false,
    failures,
  };
}

function applySuccessfulResults(endpoint, results, requestId = "") {
  lastResultsByCategory = results;
  updateSummaryChips(lastResultsByCategory);
  renderResultCards();

  const elapsed = finishAnalysisClock(true);
  if (progressMeta) {
    const currentMeta = (progressMeta.textContent || "Analysis complete.").trim();
    progressMeta.textContent = currentMeta.includes("Processed in")
      ? currentMeta
      : `${currentMeta} Processed in ${elapsed}.`;
  }

  selectedFiles = [];
  imageInput.value = "";
  renderQueue();
  updateQueueMetrics(true);
  setStatus(`Completed via ${endpoint} in ${elapsed}.${requestIdSuffix(requestId)}`);
}

async function runSingleBatchUpload(batchFiles, batchIndex, totalBatches) {
  const batchPrefix = totalBatches > 1 ? `Batch ${batchIndex}/${totalBatches}` : "Batch 1/1";
  const filesCount = batchFiles.length;
  const sizeLabel = bytesToSize(totalBytes(batchFiles));

  setProgress(
    clamp(Math.round(((batchIndex - 1) / totalBatches) * 100), 2, 99),
    "Queueing",
    `${batchPrefix}: preparing ${filesCount} files (${sizeLabel})...`
  );

  const asyncFlow = await runAsyncUploadFlow(batchFiles);
  if (asyncFlow.ok) {
    return {
      ok: true,
      endpoint: asyncFlow.endpoint,
      requestId: asyncFlow.requestId || "",
      results: asyncFlow.results,
    };
  }

  const syncFlow = await runSyncUploadFlow(batchFiles);
  if (syncFlow.ok) {
    return {
      ok: true,
      endpoint: syncFlow.endpoint,
      requestId: syncFlow.requestId || "",
      results: syncFlow.results,
    };
  }

  const allFailures = [...asyncFlow.failures, ...syncFlow.failures];
  const summary = allFailures.length ? allFailures[allFailures.length - 1] : `${batchPrefix}: no reachable endpoint.`;
  return {
    ok: false,
    error: `${batchPrefix} failed. ${summary}`,
  };
}

async function uploadImages() {
  if (!selectedFiles.length || isUploading) {
    return;
  }

  clearError();
  setUploadingState(true);
  startAnalysisClock();
  setProgress(2, "Queueing", "Preparing files for analysis...");
  let successful = false;

  try {
    const batches = buildUploadBatches(selectedFiles);
    const totalFileCount = selectedFiles.length;
    const totalSizeLabel = bytesToSize(totalBytes(selectedFiles));
    let processedFileCount = 0;
    const mergedResults = createEmptyResults();
    let lastEndpoint = "";
    let lastRequestId = "";

    if (batches.length > 1) {
      setStatus(
        `Large batch detected. Processing ${totalFileCount} files in ${batches.length} batches (${MAX_BATCH_FILES} files or ${MAX_REQUEST_SIZE_MB}MB per request).`
      );
    } else {
      setStatus(`Processing ${totalFileCount} file(s), total ${totalSizeLabel}...`);
    }

    for (let index = 0; index < batches.length; index += 1) {
      const batchFiles = batches[index];
      const batchResult = await runSingleBatchUpload(batchFiles, index + 1, batches.length);
      if (!batchResult.ok) {
        throw new Error(`${batchResult.error} Try again or reduce total batch size.`);
      }

      mergeResults(mergedResults, batchResult.results);
      lastEndpoint = batchResult.endpoint || lastEndpoint;
      lastRequestId = batchResult.requestId || lastRequestId;
      processedFileCount += batchFiles.length;

      const totalPercent = Math.round((processedFileCount / totalFileCount) * 100);
      setProgress(
        clamp(totalPercent, 5, 100),
        batches.length > 1 ? `Batch ${index + 1}/${batches.length} complete` : "Completed",
        `Processed ${processedFileCount}/${totalFileCount} files.`
      );
    }

    const endpointLabel = batches.length > 1 ? `${lastEndpoint} (batched x${batches.length})` : lastEndpoint;
    applySuccessfulResults(endpointLabel, mergedResults, lastRequestId);
    successful = true;
    return;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected upload failure.";
    showError(message);
    setProgress(0, "Analysis failed", message);
    finishAnalysisClock(false);
    setStatus("Upload failed. Fix the issue and try again.");
  } finally {
    setUploadingState(false);
    updateQueueMetrics(successful);
    if (!successful) {
      stopPseudoProgress(0, "Analysis failed.");
    }
  }
}

function handleQueueClick(event) {
  if (isUploading) {
    return;
  }

  const button = event.target.closest(".queue-delete");
  if (!button) {
    return;
  }

  const index = Number(button.dataset.index);
  if (!Number.isInteger(index)) {
    return;
  }

  selectedFiles.splice(index, 1);
  renderQueue();
  updateQueueMetrics();
}

function handleResultGridClick(event) {
  const button = event.target.closest("[data-download-category]");
  if (!button) {
    return;
  }

  const category = String(button.dataset.downloadCategory || "");
  if (!CATEGORIES.includes(category)) {
    return;
  }

  const items = categoryItems(category);
  if (!items.length) {
    showError(`No analyzed images available in ${category} folder.`);
    return;
  }

  downloadResultsZip([category], `${category}_folder`);
}

function handleDropZoneKey(event) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
}

dropZone.addEventListener("click", () => imageInput.click());
dropZone.addEventListener("keydown", handleDropZoneKey);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-active");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-active");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-active");
  if (event.dataTransfer?.files?.length) {
    addFilesToQueue(event.dataTransfer.files);
  }
});

imageInput.addEventListener("change", () => {
  if (imageInput.files?.length) {
    addFilesToQueue(imageInput.files);
  }
  imageInput.value = "";
});

queueList.addEventListener("click", handleQueueClick);
resultGrid.addEventListener("click", handleResultGridClick);
analyzeBtn.addEventListener("click", uploadImages);
clearQueueBtn.addEventListener("click", clearQueue);
cancelBtn.addEventListener("click", clearQueue);
if (downloadAllZipBtn) {
  downloadAllZipBtn.addEventListener("click", () => {
    const categoriesWithItems = CATEGORIES.filter((category) => categoryItems(category).length > 0);
    if (!categoriesWithItems.length) {
      showError("No analyzed images available for ZIP download.");
      return;
    }
    downloadResultsZip(categoriesWithItems, "visionsort_folders");
  });
}

chipRow.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip[data-filter]");
  if (!chip) {
    return;
  }
  setFilter(chip.dataset.filter);
});

renderQueue();
updateQueueMetrics();
updateSummaryChips(lastResultsByCategory);
renderResultCards();
setFilter("all");
resetProgress();
updateDownloadControls();

if (fileSizeLimitBadge) {
  fileSizeLimitBadge.textContent = `MAX ${MAX_FILE_SIZE_MB}MB`;
}
