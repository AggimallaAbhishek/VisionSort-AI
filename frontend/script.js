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
  apiOverrideFromQuery || window.VISIONSORT_API_BASE_URL || storedApiBase || metaApiBase || ""
).trim();

const DEFAULT_DEPLOYED_API_BASE_URL = "https://visionsort-ai.onrender.com";
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const CATEGORIES = ["good", "blurry", "dark", "overexposed", "duplicates"];

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

let selectedFiles = [];
let currentFilter = "all";
let lastResultsByCategory = createEmptyResults();
let isUploading = false;

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

  const unique = [];
  const seen = new Set();

  for (const base of candidates) {
    if (!base) {
      continue;
    }

    const normalized = normalizeApiBase(base);
    if (seen.has(normalized)) {
      continue;
    }

    seen.add(normalized);
    unique.push(normalized);
  }

  return unique;
}

const API_BASE_CANDIDATES = buildApiCandidates();

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

function updateQueueMetrics() {
  const count = selectedFiles.length;
  queueCount.textContent = `${count} Selected`;
  metricSelected.textContent = String(count);

  if (!isUploading) {
    setStatus(count ? "Queue ready. Click Analyze Images." : "Ready for analysis.");
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

    if (!ALLOWED_TYPES.has((file.type || "").toLowerCase())) {
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
  }
}

function clearQueue() {
  if (isUploading) {
    return;
  }
  selectedFiles = [];
  imageInput.value = "";
  renderQueue();
  updateQueueMetrics();
}

function buildFormData(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return formData;
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

async function attemptUpload(apiBase, files) {
  const endpoint = `${apiBase}/upload`;

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

async function uploadImages() {
  if (!selectedFiles.length || isUploading) {
    return;
  }

  clearError();
  setUploadingState(true);

  try {
    const failures = [];

    for (const apiBase of API_BASE_CANDIDATES) {
      const result = await attemptUpload(apiBase, selectedFiles);
      if (result.ok) {
        lastResultsByCategory = {
          good: Array.isArray(result.data.good) ? result.data.good : [],
          blurry: Array.isArray(result.data.blurry) ? result.data.blurry : [],
          dark: Array.isArray(result.data.dark) ? result.data.dark : [],
          overexposed: Array.isArray(result.data.overexposed) ? result.data.overexposed : [],
          duplicates: Array.isArray(result.data.duplicates) ? result.data.duplicates : [],
        };

        updateSummaryChips(lastResultsByCategory);
        renderResultCards();
        setStatus(`Completed via ${result.endpoint}`);
        return;
      }

      failures.push(`${result.endpoint} -> ${result.error}`);

      if (result.status >= 400 && result.status < 500 && result.status !== 404) {
        break;
      }
    }

    const summary = failures.length
      ? failures[failures.length - 1]
      : "No upload endpoint was reachable.";
    throw new Error(`${summary} (Tried: ${API_BASE_CANDIDATES.join(", ")})`);
  } catch (error) {
    showError(error instanceof Error ? error.message : "Unexpected upload failure.");
    setStatus("Upload failed. Fix the issue and try again.");
  } finally {
    setUploadingState(false);
    updateQueueMetrics();
  }
}

function handleQueueClick(event) {
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
