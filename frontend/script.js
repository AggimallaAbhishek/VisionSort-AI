const API_BASE_URL = window.VISIONSORT_API_BASE_URL || "http://localhost:10000";

const uploadButton = document.getElementById("uploadButton");
const inputEl = document.getElementById("imageInput");
const loadingEl = document.getElementById("loading");

const categories = ["good", "blurry", "dark", "overexposed", "duplicates"];

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
  uploadButton.disabled = isLoading;
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
  meta.innerHTML = `
    <strong>${item.file_name}</strong>
    <span>Blur: ${item.blur_score}</span>
    <span>Brightness: ${item.brightness_level}</span>
    <span>AI Label: ${item.ai_label}</span>
  `;

  card.appendChild(image);
  card.appendChild(meta);
  container.appendChild(card);
}

function renderError(message) {
  const resultsGrid = document.getElementById("resultsGrid");
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

  clearResults();
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Upload failed (${response.status}): ${text}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    renderError(error.message || "Unexpected upload error.");
  } finally {
    setLoading(false);
  }
}

uploadButton.addEventListener("click", uploadImages);
