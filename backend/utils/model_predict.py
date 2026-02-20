"""Optional quality classification via a pretrained CNN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


CLASS_NAMES: List[str] = ["good", "blurry", "dark", "overexposed", "duplicates"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "photo_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class QualityModel:
    """Loads the model once and exposes prediction method."""

    def __init__(self) -> None:
        self.model = mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        self.model.to(DEVICE)
        self.model.eval()
        self.available = False
        self._load_weights()

    def _load_weights(self) -> None:
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            logger.warning("Model file not found or empty: %s", MODEL_PATH)
            return

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)
            self.available = True
            logger.info("Loaded MobileNetV2 weights from %s", MODEL_PATH)
        except Exception as exc:
            logger.exception("Failed to load model weights from %s: %s", MODEL_PATH, exc)
            self.available = False

    def predict(self, image: Image.Image) -> str:
        if not self.available:
            return "model_unavailable"

        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_index = int(torch.argmax(logits, dim=1).item())
        return CLASS_NAMES[pred_index]


_MODEL = QualityModel()


def predict_quality(image: Image.Image) -> str:
    """Predict quality label using a preloaded CNN model."""
    return _MODEL.predict(image)
