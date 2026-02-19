const metaApiBase = document
  .querySelector('meta[name="visionsort-api-base-url"]')
  ?.getAttribute("content")
  ?.trim();

const DEFAULT_DEPLOYED_API_BASE_URL = "https://visionsort-ai-backend.onrender.com";
const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const isRelativeMeta = !!metaApiBase && metaApiBase.startsWith("/");

const fallbackApiBase = isLocalHost ? "http://localhost:10000" : DEFAULT_DEPLOYED_API_BASE_URL;
const apiBaseFromMeta = isLocalHost && isRelativeMeta ? "" : metaApiBase;
const API_BASE_URL = window.VISIONSORT_API_BASE_URL || apiBaseFromMeta || fallbackApiBase;

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

async function uploadImages() {
  const files = inputEl.files;
  if (!files || files.length === 0) {
    alert("Please select one or more images first.");
    return;
  }

  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  clearError();
  clearResults();
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = `Upload failed (${response.status})`;
      try {
        const payload = await response.json();
        if (payload && payload.detail) {
          detail = `Upload failed (${response.status}): ${payload.detail}`;
        }
      } catch {
        const text = await response.text();
        if (text) {
          detail = `Upload failed (${response.status}): ${text}`;
        }
      }
      throw new Error(detail);
    }

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected upload error.";
    renderError(`${message} (API: ${API_BASE_URL})`);
  } finally {
    setLoading(false);
  }
}

uploadButton.addEventListener("click", uploadImages);
