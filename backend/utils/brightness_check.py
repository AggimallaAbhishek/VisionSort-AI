"""Brightness analysis utilities."""

from __future__ import annotations

import cv2
import numpy as np


def analyze_brightness(image: np.ndarray) -> str:
    """Classify image brightness as dark, overexposed, or normal."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image array for brightness analysis.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_brightness = float(hsv[:, :, 2].mean())

    if mean_brightness < 50:
        return "dark"
    if mean_brightness > 200:
        return "overexposed"
    return "normal"
