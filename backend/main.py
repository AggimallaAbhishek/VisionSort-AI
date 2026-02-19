"""VisionSort AI backend service."""

from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from aws_client import AWSService
from utils.blur_detection import detect_blur
from utils.brightness_check import analyze_brightness
from utils.duplicate_check import is_duplicate
from utils.model_predict import predict_quality

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100"))
DUPLICATE_HASH_DISTANCE = int(os.getenv("DUPLICATE_HASH_DISTANCE", "5"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "1024"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES = int(os.getenv("MAX_FILES", "50"))
ENABLE_AI_LABEL = os.getenv("ENABLE_AI_LABEL", "true").lower() == "true"
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "anonymous")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
EXTENSION_TO_CONTENT_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

app = FastAPI(title="VisionSort AI", version="1.1.0")

origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
if not allow_origins:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

aws_service = AWSService()


@app.get("/")
def root() -> Dict[str, Any]:
    """Health endpoint with non-sensitive service status."""
    return {
        "message": "VisionSort AI backend is running.",
        "service": aws_service.health_snapshot(),
    }


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safer storage keys."""
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return clean_name or f"image_{uuid.uuid4().hex}.jpg"


def resolve_content_type(upload: UploadFile, filename: str) -> Optional[str]:
    """Resolve content type from request metadata or filename extension."""
    content_type = (upload.content_type or "").strip().lower()
    if content_type in ALLOWED_IMAGE_TYPES:
        return content_type

    ext = Path(filename).suffix.lower()
    inferred = EXTENSION_TO_CONTENT_TYPE.get(ext)
    if inferred in ALLOWED_IMAGE_TYPES:
        return inferred

    return None


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode image bytes to an OpenCV BGR array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    return image


def resize_image(image: np.ndarray, max_width: int) -> np.ndarray:
    """Resize while preserving aspect ratio."""
    if max_width <= 0:
        return image

    height, width = image.shape[:2]
    if width <= max_width:
        return image

    ratio = max_width / float(width)
    new_size = (max_width, int(height * ratio))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_preview_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    """Encode an image as compressed JPEG for preview + processed storage."""
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode preview image.")
    return encoded.tobytes()


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def choose_final_status(blur_score: float, brightness_level: str, duplicate: bool) -> str:
    """Choose final category based on image metrics."""
    if duplicate:
        return "duplicates"
    if brightness_level == "dark":
        return "dark"
    if brightness_level == "overexposed":
        return "overexposed"
    if blur_score < BLUR_THRESHOLD:
        return "blurry"
    return "good"


def build_item_payload(
    file_name: str,
    blur_score: float,
    brightness_level: str,
    ai_label: str,
    final_status: str,
    preview_data_url: str,
    storage_path: str | None,
    processed_storage_path: str | None,
) -> Dict[str, Any]:
    """Build response payload item for frontend rendering."""
    return {
        "file_name": file_name,
        "blur_score": round(blur_score, 2),
        "brightness_level": brightness_level,
        "ai_label": ai_label,
        "final_status": final_status,
        "preview_data_url": preview_data_url,
        "storage_path": storage_path,
        "processed_storage_path": processed_storage_path,
    }


def persist_metadata(
    file_name: str,
    blur_score: float,
    brightness_level: str,
    ai_label: str,
    final_status: str,
) -> None:
    """Persist image metadata in PostgreSQL images table."""
    row = {
        "id": str(uuid.uuid4()),
        "user_id": DEFAULT_USER_ID,
        "file_name": file_name,
        "blur_score": round(blur_score, 2),
        "brightness_level": brightness_level,
        "ai_label": ai_label,
        "final_status": final_status,
        "created_at": datetime.now(timezone.utc),
    }
    aws_service.insert_image_metadata(row)


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)) -> Dict[str, List[Dict[str, Any]]]:
    """Process uploaded files and return categorized image payloads."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum allowed is {MAX_FILES}.")

    results: Dict[str, List[Dict[str, Any]]] = {
        "good": [],
        "blurry": [],
        "dark": [],
        "overexposed": [],
        "duplicates": [],
    }

    seen_hashes: List[Any] = []
    processed_count = 0

    for upload in files:
        try:
            file_name = sanitize_filename(upload.filename or "unnamed_image")
            content_type = resolve_content_type(upload, file_name)

            if not content_type:
                logger.warning("Skipping unsupported file type: %s (%s)", file_name, upload.content_type)
                continue

            raw_bytes = await upload.read()
            if not raw_bytes:
                logger.warning("Skipping empty file: %s", file_name)
                continue

            if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
                logger.warning("Skipping oversized file: %s", file_name)
                continue

            image = decode_image(raw_bytes)
            resized_image = resize_image(image, MAX_IMAGE_WIDTH)
            pil_image = bgr_to_pil(resized_image)

            blur_score = detect_blur(resized_image)
            brightness_level = analyze_brightness(resized_image)
            duplicate = is_duplicate(pil_image, seen_hashes, threshold=DUPLICATE_HASH_DISTANCE)
            ai_label = predict_quality(pil_image) if ENABLE_AI_LABEL else "disabled"
            final_status = choose_final_status(blur_score, brightness_level, duplicate)

            object_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            object_token = uuid.uuid4().hex
            original_object_path = f"{object_prefix}/{object_token}_{file_name}"

            uploads_storage_path = None
            processed_storage_path = None

            try:
                uploads_storage_path = aws_service.upload_image(
                    path=original_object_path,
                    data=raw_bytes,
                    content_type=content_type,
                    bucket="uploads",
                )
            except Exception as exc:
                logger.exception("S3 original upload failed for %s: %s", file_name, exc)

            preview_bytes = encode_preview_jpeg(resized_image)
            processed_object_path = f"{object_prefix}/{object_token}_processed.jpg"

            try:
                processed_storage_path = aws_service.upload_image(
                    path=processed_object_path,
                    data=preview_bytes,
                    content_type="image/jpeg",
                    bucket="processed",
                )
            except Exception as exc:
                logger.exception("S3 processed upload failed for %s: %s", file_name, exc)

            try:
                persist_metadata(file_name, blur_score, brightness_level, ai_label, final_status)
            except Exception as exc:
                logger.exception("RDS metadata insert failed for %s: %s", file_name, exc)

            preview_data_url = f"data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode('utf-8')}"
            item = build_item_payload(
                file_name=file_name,
                blur_score=blur_score,
                brightness_level=brightness_level,
                ai_label=ai_label,
                final_status=final_status,
                preview_data_url=preview_data_url,
                storage_path=uploads_storage_path,
                processed_storage_path=processed_storage_path,
            )
            results[final_status].append(item)
            processed_count += 1

        except ValueError as exc:
            logger.warning("Skipping invalid image %s: %s", upload.filename, exc)
            continue
        except Exception as exc:
            logger.exception("Unexpected processing error for %s: %s", upload.filename, exc)
            continue

    if processed_count == 0:
        raise HTTPException(status_code=400, detail="No valid image files were processed.")

    return results
