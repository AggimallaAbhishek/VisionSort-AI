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

const DEFAULT_DEPLOYED_API_BASE_URL = "https://visionsort-ai.onrender.com";
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp"]);
const CATEGORIES = ["good", "blurry", "dark", "overexposed", "duplicates"];
const ASYNC_POLL_INTERVAL_MS = 700;
const ASYNC_TIMEOUT_MS = 8 * 60 * 1000;

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

const chipAll = document.getElementById("chipAll");
const chipGood = document.getElementById("chipGood");
const chipBlurry = document.getElementById("chipBlurry");
const chipDark = document.getElementById("chipDark");
const chipOverexposed = document.getElementById("chipOverexposed");
const chipDuplicates = document.getElementById("chipDuplicates");

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

function bytesToSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 MB";
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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

function mapBrightnessValue(level) {
  if (level === "dark") {
    return 18;
  }
  if (level === "overexposed") {
    return 92;
  }
  return 55;
}

function renderResultCards() {
  const allItems = flattenResults(lastResultsByCategory);
  const filtered = currentFilter === "all" ? allItems : allItems.filter((item) => item.__category === currentFilter);

  resultGrid.innerHTML = "";

  if (!filtered.length) {
    const empty = document.createElement("article");
    empty.className = "result-empty";
    empty.textContent = "No processed images for this category yet.";
    resultGrid.appendChild(empty);
    return;
  }

  filtered.forEach((item) => {
    const blurPercent = clamp((Number(item.blur_score || 0) / 2000) * 100, 2, 100);
    const brightnessPercent = mapBrightnessValue(item.brightness_level);
    const safeName = escapeHtml(item.file_name);
    const safeCategory = escapeHtml(item.__category);
    const safeLabel = escapeHtml(item.ai_label || "model_unavailable");
    const safeBrightness = escapeHtml(item.brightness_level || "normal");
    const safeStorage = escapeHtml(item.storage_path || "n/a");
    const safeProcessedStorage = escapeHtml(item.processed_storage_path || "n/a");

    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <div class="card-image-wrap">
        <img src="${item.preview_data_url}" alt="${safeName}" loading="lazy" />
        <span class="status-badge ${safeCategory}">${safeCategory}</span>
      </div>
      <div class="card-body">
        <h3 class="card-title">${safeName}</h3>
        <p class="card-subtitle">${safeLabel}</p>

        <div class="metric-line">
          <div class="label-row"><span>Blur Score</span><strong>${Number(item.blur_score || 0).toFixed(2)}</strong></div>
          <div class="progress blur"><span style="width:${blurPercent}%"></span></div>
        </div>

        <div class="metric-line">
          <div class="label-row"><span>Brightness</span><strong>${safeBrightness}</strong></div>
          <div class="progress brightness"><span style="width:${brightnessPercent}%"></span></div>
        </div>

        <p class="footer-meta">Original: ${safeStorage}<br />Processed: ${safeProcessedStorage}</p>
      </div>
    `;

    resultGrid.appendChild(card);
  });
}

function addFilesToQueue(fileList) {
  const additions = [];
  const issues = [];

  Array.from(fileList).forEach((file) => {
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

async function getResponseErrorMessage(response) {
  const raw = await response.text();
  if (!raw) {
    return `Upload failed (${response.status}).`;
  }

  try {
    const payload = JSON.parse(raw);
    if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
      return `Upload failed (${response.status}): ${payload.detail}`;
    }
  } catch {
    // Fall through to raw text output.
  }

  const snippet = raw.length > 220 ? `${raw.slice(0, 220)}...` : raw;
  return `Upload failed (${response.status}): ${snippet}`;
}

async function attemptUpload(endpoint, files) {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: buildFormData(files),
    });

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        endpoint,
        error: await getResponseErrorMessage(response),
      };
    }

    return {
      ok: true,
      endpoint,
      data: await response.json(),
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      endpoint,
      error: error instanceof Error ? error.message : "Unexpected network error.",
    };
  }
}

async function fetchJobSnapshot(endpoint) {
  try {
    const response = await fetch(endpoint, {
      method: "GET",
      cache: "no-store",
    });

    if (!response.ok) {
      return {
        ok: false,
        endpoint,
        error: await getResponseErrorMessage(response),
      };
    }

    return {
      ok: true,
      endpoint,
      data: await response.json(),
    };
  } catch (error) {
    return {
      ok: false,
      endpoint,
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
        break;
      }
      lastError = `${endpoint} -> ${result.error}`;
    }

    if (!snapshot) {
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
        job,
      };
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Background job failed.");
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

function applySuccessfulResults(endpoint, results) {
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
  setStatus(`Completed via ${endpoint} in ${elapsed}.`);
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
    const asyncFlow = await runAsyncUploadFlow(selectedFiles);
    if (asyncFlow.ok) {
      applySuccessfulResults(asyncFlow.endpoint, asyncFlow.results);
      successful = true;
      return;
    }

    const syncFlow = await runSyncUploadFlow(selectedFiles);
    if (syncFlow.ok) {
      applySuccessfulResults(syncFlow.endpoint, syncFlow.results);
      successful = true;
      return;
    }

    const allFailures = [...asyncFlow.failures, ...syncFlow.failures];
    const summary = allFailures.length ? allFailures[allFailures.length - 1] : "No upload endpoint was reachable.";
    throw new Error(summary);
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
analyzeBtn.addEventListener("click", uploadImages);
clearQueueBtn.addEventListener("click", clearQueue);
cancelBtn.addEventListener("click", clearQueue);

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
