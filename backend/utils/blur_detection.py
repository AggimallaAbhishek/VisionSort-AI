"""Blur detection utilities."""

from __future__ import annotations

import cv2
import numpy as np


def detect_blur(image: np.ndarray) -> float:
    """Return the variance of Laplacian for blur detection.

    Lower values indicate blurrier images.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array for blur detection.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)
