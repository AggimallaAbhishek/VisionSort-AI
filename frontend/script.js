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

const DEFAULT_DEPLOYED_API_BASE_URL = "https://visionsort-ai-backend.onrender.com";
const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);

function normalizeApiBase(base) {
  return base.replace(/\/+$/, "");
}

function isRelativePath(base) {
  return base.startsWith("/");
}

function buildApiCandidates() {
  const candidates = [];

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

const uploadButton = document.getElementById("uploadButton");
const inputEl = document.getElementById("imageInput");
const loadingEl = document.getElementById("loading");
const resultsGrid = document.getElementById("resultsGrid");

const categories = ["good", "blurry", "dark", "overexposed", "duplicates"];

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
  uploadButton.disabled = isLoading;
}

function clearError() {
  const existing = document.querySelector(".error");
  if (existing) {
    existing.remove();
  }
}

function clearResults() {
  categories.forEach((key) => {
    const container = document.getElementById(key);
    container.innerHTML = "";
  });
}

function renderEmpty(container) {
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = "No images";
  container.appendChild(empty);
}

function renderItem(container, item) {
  const card = document.createElement("div");
  card.className = "image-card";

  const image = document.createElement("img");
  image.src = item.preview_data_url;
  image.alt = item.file_name;

  const meta = document.createElement("div");
  meta.className = "image-meta";

  const lines = [
    `<strong>${item.file_name}</strong>`,
    `<span>Blur: ${item.blur_score}</span>`,
    `<span>Brightness: ${item.brightness_level}</span>`,
    `<span>AI Label: ${item.ai_label}</span>`,
  ];

  if (item.storage_path) {
    lines.push(`<span>Original: ${item.storage_path}</span>`);
  }

  if (item.processed_storage_path) {
    lines.push(`<span>Processed: ${item.processed_storage_path}</span>`);
  }

  meta.innerHTML = lines.join("\n");

  card.appendChild(image);
  card.appendChild(meta);
  container.appendChild(card);
}

function renderError(message) {
  clearError();

  const err = document.createElement("p");
  err.className = "error";
  err.textContent = message;
  resultsGrid.prepend(err);
}

function renderResults(data) {
  categories.forEach((key) => {
    const container = document.getElementById(key);
    const items = Array.isArray(data[key]) ? data[key] : [];

    if (items.length === 0) {
      renderEmpty(container);
      return;
    }

    items.forEach((item) => renderItem(container, item));
  });
}

function buildFormData(files) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  return formData;
}

async function getResponseErrorMessage(response) {
  const raw = await response.text();
  if (!raw) {
    return `Upload failed (${response.status})`;
  }

  try {
    const payload = JSON.parse(raw);
    if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
      return `Upload failed (${response.status}): ${payload.detail}`;
    }
  } catch {
    // Fall through to raw text output.
  }

  const snippet = raw.length > 240 ? `${raw.slice(0, 240)}...` : raw;
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
    const errorMessage = error instanceof Error ? error.message : "Unexpected network error.";
    return {
      ok: false,
      status: 0,
      endpoint,
      error: errorMessage,
    };
  }
}

async function uploadImages() {
  const files = inputEl.files;
  if (!files || files.length === 0) {
    alert("Please select one or more images first.");
    return;
  }

  clearError();
  clearResults();
  setLoading(true);

  try {
    const failures = [];

    for (const apiBase of API_BASE_CANDIDATES) {
      const result = await attemptUpload(apiBase, files);
      if (result.ok) {
        renderResults(result.data);
        return;
      }

      failures.push(`${result.endpoint} -> ${result.error}`);

      if (result.status >= 400 && result.status < 500 && result.status !== 404) {
        break;
      }
    }

    const summary = failures.length ? failures[failures.length - 1] : "No upload endpoint available.";
    throw new Error(`${summary} (Tried: ${API_BASE_CANDIDATES.join(", ")})`);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected upload error.";
    renderError(message);
  } finally {
    setLoading(false);
  }
}

uploadButton.addEventListener("click", uploadImages);
