"""Shared test fixtures.

The real backend/.env holds live AWS credentials and a production DATABASE_URL.
`load_dotenv()` does not override variables that already exist in the environment,
so we seed safe values here BEFORE `main` is imported. Nothing in the suite may
reach real AWS.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

SAFE_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_REGION": "us-east-1",
    "S3_UPLOADS_BUCKET": "",
    "S3_PROCESSED_BUCKET": "",
    "DATABASE_URL": "",
    "BACKEND_API_KEY": "",
    "ENABLE_AI_LABEL": "false",
    "SPOOL_UPLOADS_TO_DISK": "false",
    "INCLUDE_PREVIEW_DATA_URL": "false",
    "PREVIEW_INLINE_MAX_FILES": "0",
    "ALLOWED_ORIGINS": "*",
    "ALLOWED_HOSTS": "*",
    "STORAGE_TOKEN_SECRET": "unit-test-secret-do-not-use-in-production",
}
os.environ.update(SAFE_ENV)

# Cleared so tests assert the shipped code defaults rather than the developer's .env.
for _name in ("MAX_ZIP_SOURCE_BYTES_MB", "MAX_ZIP_ITEMS", "MAX_IMAGE_PIXELS", "MAX_FILES"):
    os.environ.pop(_name, None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="session")
def app_main():
    """Import the FastAPI module once, after the environment is neutralized."""
    import main

    assert not main.aws_service.s3_enabled, "tests must never run against real S3"
    assert not main.aws_service.db_enabled, "tests must never run against a real database"
    return main


def encode_jpeg(width: int = 240, height: int = 240, fill: int = 128, seed: int = 0) -> bytes:
    """Build a JPEG with mild noise so it is not flagged as blurry."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), fill, dtype=np.uint8)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-60, 60, base.shape), 0, 255)
    return cv2.imencode(".jpg", noisy.astype(np.uint8))[1].tobytes()


def encode_png(width: int, height: int) -> bytes:
    """Build a flat PNG. Compresses to almost nothing but decodes to width*height*3."""
    return cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))[1].tobytes()


def make_payload(raw: bytes, name: str = "image.jpg") -> dict:
    return {
        "file_name": name,
        "content_type": "image/jpeg",
        "raw_bytes": raw,
        "temp_path": None,
        "validation_error": None,
    }


@pytest.fixture
def jpeg_bytes() -> bytes:
    return encode_jpeg()


class FakeS3Client:
    """Records the exact Bucket/Key pairs handed to boto3."""

    def __init__(self, payload: bytes = b"OBJECT-BYTES") -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3 kwarg names
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.payload)}

    def put_object(self, **kwargs):  # noqa: ANN003
        self.calls.append((kwargs.get("Bucket", ""), kwargs.get("Key", "")))
        return {}
