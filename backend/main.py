"""VisionSort AI backend service."""

from __future__ import annotations

import base64
import logging
import os
import re
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

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
PERSIST_WORKERS = max(1, int(os.getenv("PERSIST_WORKERS", "6")))
UPLOAD_JOB_WORKERS = max(1, int(os.getenv("UPLOAD_JOB_WORKERS", "2")))
JOB_RETENTION_MINUTES = max(5, int(os.getenv("JOB_RETENTION_MINUTES", "60")))

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
EXTENSION_TO_CONTENT_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

app = FastAPI(title="VisionSort AI", version="1.1.0")

origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
allow_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
if not allow_origins:
    allow_origins = ["*"]

# Allow Vercel preview/production domains by default in non-wildcard mode.
origin_regex_env = os.getenv("ALLOWED_ORIGIN_REGEX", r"^https://.*\.vercel\.app$").strip()
allow_origin_regex = None if allow_origins == ["*"] else (origin_regex_env or None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

aws_service = AWSService()
persist_executor = ThreadPoolExecutor(max_workers=PERSIST_WORKERS)
job_executor = ThreadPoolExecutor(max_workers=UPLOAD_JOB_WORKERS)

UploadPayload = Dict[str, Any]
ProgressHook = Callable[[int, int, str, str], None]
upload_jobs: Dict[str, Dict[str, Any]] = {}
upload_jobs_lock = Lock()


@app.get("/")
@app.get("/api")
def root() -> Dict[str, Any]:
    """Health endpoint with non-sensitive service status."""
    return {
        "message": "VisionSort AI backend is running.",
        "service": aws_service.health_snapshot(),
        "cors": {
            "allow_origins": allow_origins,
            "allow_origin_regex": allow_origin_regex,
        },
        "workers": {
            "persist_workers": PERSIST_WORKERS,
            "upload_job_workers": UPLOAD_JOB_WORKERS,
        },
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


def build_metadata_row(
    file_name: str,
    blur_score: float,
    brightness_level: str,
    ai_label: str,
    final_status: str,
) -> Dict[str, Any]:
    """Build metadata row for PostgreSQL insertion."""
    return {
        "id": str(uuid.uuid4()),
        "user_id": DEFAULT_USER_ID,
        "file_name": file_name,
        "blur_score": round(blur_score, 2),
        "brightness_level": brightness_level,
        "ai_label": ai_label,
        "final_status": final_status,
        "created_at": datetime.now(timezone.utc),
    }


def build_results_template() -> Dict[str, List[Dict[str, Any]]]:
    """Create an empty response template for all categories."""
    return {
        "good": [],
        "blurry": [],
        "dark": [],
        "overexposed": [],
        "duplicates": [],
    }


def persist_artifacts(
    file_name: str,
    content_type: str,
    raw_bytes: bytes,
    preview_bytes: bytes,
) -> Tuple[str | None, str | None]:
    """Upload original and processed artifacts to S3."""
    object_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    object_token = uuid.uuid4().hex
    original_object_path = f"{object_prefix}/{object_token}_{file_name}"
    processed_object_path = f"{object_prefix}/{object_token}_processed.jpg"

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

    try:
        processed_storage_path = aws_service.upload_image(
            path=processed_object_path,
            data=preview_bytes,
            content_type="image/jpeg",
            bucket="processed",
        )
    except Exception as exc:
        logger.exception("S3 processed upload failed for %s: %s", file_name, exc)

    return uploads_storage_path, processed_storage_path


async def read_upload_payloads(files: List[UploadFile]) -> List[UploadPayload]:
    """Read and validate uploads once so they can be processed sync/async."""
    payloads: List[UploadPayload] = []

    for upload in files:
        file_name = sanitize_filename(upload.filename or "unnamed_image")
        content_type = resolve_content_type(upload, file_name)

        if not content_type:
            logger.warning("Skipping unsupported file type: %s (%s)", file_name, upload.content_type)
            payloads.append(
                {
                    "file_name": file_name,
                    "content_type": None,
                    "raw_bytes": b"",
                    "validation_error": "unsupported_file_type",
                }
            )
            continue

        raw_bytes = await upload.read()
        if not raw_bytes:
            logger.warning("Skipping empty file: %s", file_name)
            payloads.append(
                {
                    "file_name": file_name,
                    "content_type": content_type,
                    "raw_bytes": b"",
                    "validation_error": "empty_file",
                }
            )
            continue

        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            logger.warning("Skipping oversized file: %s", file_name)
            payloads.append(
                {
                    "file_name": file_name,
                    "content_type": content_type,
                    "raw_bytes": b"",
                    "validation_error": "file_too_large",
                }
            )
            continue

        payloads.append(
            {
                "file_name": file_name,
                "content_type": content_type,
                "raw_bytes": raw_bytes,
                "validation_error": None,
            }
        )

    return payloads


def process_upload_payloads(
    payloads: List[UploadPayload],
    progress_hook: ProgressHook | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process image payloads and return categorized results."""
    results = build_results_template()
    seen_hashes: List[Any] = []
    metadata_rows: List[Dict[str, Any]] = []
    persistence_tasks: List[Tuple[Future[Tuple[str | None, str | None]], Dict[str, Any], str]] = []
    processed_count = 0
    total_files = len(payloads)

    for index, payload in enumerate(payloads, start=1):
        file_name = payload["file_name"]
        try:
            validation_error = payload.get("validation_error")
            if validation_error:
                continue

            raw_bytes: bytes = payload["raw_bytes"]
            content_type: str = payload["content_type"]

            image = decode_image(raw_bytes)
            resized_image = resize_image(image, MAX_IMAGE_WIDTH)
            pil_image = bgr_to_pil(resized_image)

            blur_score = detect_blur(resized_image)
            brightness_level = analyze_brightness(resized_image)
            duplicate = is_duplicate(pil_image, seen_hashes, threshold=DUPLICATE_HASH_DISTANCE)
            ai_label = predict_quality(pil_image) if ENABLE_AI_LABEL else "disabled"
            final_status = choose_final_status(blur_score, brightness_level, duplicate)

            preview_bytes = encode_preview_jpeg(resized_image)
            preview_data_url = f"data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode('utf-8')}"
            item = build_item_payload(
                file_name=file_name,
                blur_score=blur_score,
                brightness_level=brightness_level,
                ai_label=ai_label,
                final_status=final_status,
                preview_data_url=preview_data_url,
                storage_path=None,
                processed_storage_path=None,
            )
            results[final_status].append(item)
            metadata_rows.append(build_metadata_row(file_name, blur_score, brightness_level, ai_label, final_status))

            persistence_tasks.append(
                (
                    persist_executor.submit(
                        persist_artifacts,
                        file_name=file_name,
                        content_type=content_type,
                        raw_bytes=raw_bytes,
                        preview_bytes=preview_bytes,
                    ),
                    item,
                    file_name,
                )
            )
            processed_count += 1

        except ValueError as exc:
            logger.warning("Skipping invalid image %s: %s", file_name, exc)
        except Exception as exc:
            logger.exception("Unexpected processing error for %s: %s", file_name, exc)
        finally:
            if progress_hook:
                progress_hook(index, total_files, file_name, "processing")

    if progress_hook and total_files:
        progress_hook(total_files, total_files, "", "finalizing")

    for future, item, file_name in persistence_tasks:
        try:
            uploads_storage_path, processed_storage_path = future.result()
            item["storage_path"] = uploads_storage_path
            item["processed_storage_path"] = processed_storage_path
        except Exception as exc:
            logger.exception("Persistence task failed for %s: %s", file_name, exc)

    if metadata_rows:
        try:
            aws_service.insert_many_image_metadata(metadata_rows)
        except Exception as exc:
            logger.exception("RDS batch metadata insert failed: %s", exc)

    if processed_count == 0:
        raise HTTPException(status_code=400, detail="No valid image files were processed.")

    return results


def _job_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_old_jobs_unlocked() -> None:
    cutoff = time.time() - (JOB_RETENTION_MINUTES * 60)
    stale_ids = [
        job_id
        for job_id, job in upload_jobs.items()
        if job["status"] in {"completed", "failed"} and job["_updated_unix"] < cutoff
    ]
    for job_id in stale_ids:
        upload_jobs.pop(job_id, None)


def create_upload_job(total_files: int) -> str:
    """Create a background processing job and return job id."""
    job_id = uuid.uuid4().hex
    now_iso = _job_iso_now()
    with upload_jobs_lock:
        _cleanup_old_jobs_unlocked()
        upload_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "message": "Job queued.",
            "total_files": total_files,
            "processed_files": 0,
            "progress_percent": 0,
            "current_file": None,
            "results": None,
            "error": None,
            "created_at": now_iso,
            "started_at": None,
            "completed_at": None,
            "updated_at": now_iso,
            "_updated_unix": time.time(),
        }
    return job_id


def update_upload_job(job_id: str, **updates: Any) -> None:
    """Update a background job state safely."""
    now_iso = _job_iso_now()
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = now_iso
        job["_updated_unix"] = time.time()


def get_upload_job_snapshot(job_id: str) -> Dict[str, Any] | None:
    """Fetch a JSON-serializable job snapshot."""
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if not job:
            return None
        return {key: value for key, value in job.items() if not key.startswith("_")}


def run_upload_job(job_id: str, payloads: List[UploadPayload]) -> None:
    """Execute a background upload job and update progress state."""
    update_upload_job(
        job_id,
        status="processing",
        phase="processing",
        message="Analyzing images...",
        started_at=_job_iso_now(),
    )

    total_files = len(payloads)

    def progress_hook(processed_files: int, total: int, current_file: str, phase: str) -> None:
        if phase == "finalizing":
            update_upload_job(
                job_id,
                status="processing",
                phase="finalizing",
                message="Finalizing uploads and metadata...",
                processed_files=processed_files,
                total_files=total,
                current_file=None,
                progress_percent=99 if total > 0 else 0,
            )
            return

        percent = int((processed_files / total) * 100) if total else 0
        update_upload_job(
            job_id,
            status="processing",
            phase="processing",
            message=f"Processed {processed_files}/{total} files.",
            processed_files=processed_files,
            total_files=total,
            current_file=current_file or None,
            progress_percent=max(0, min(percent, 98)),
        )

    try:
        results = process_upload_payloads(payloads, progress_hook=progress_hook)
        update_upload_job(
            job_id,
            status="completed",
            phase="completed",
            message="Processing completed.",
            processed_files=total_files,
            total_files=total_files,
            current_file=None,
            progress_percent=100,
            results=results,
            completed_at=_job_iso_now(),
            error=None,
        )
    except HTTPException as exc:
        update_upload_job(
            job_id,
            status="failed",
            phase="failed",
            message="Processing failed.",
            completed_at=_job_iso_now(),
            error=str(exc.detail),
        )
    except Exception as exc:
        logger.exception("Background upload job failed: %s", exc)
        update_upload_job(
            job_id,
            status="failed",
            phase="failed",
            message="Processing failed unexpectedly.",
            completed_at=_job_iso_now(),
            error=str(exc),
        )


@app.post("/upload")
@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)) -> Dict[str, List[Dict[str, Any]]]:
    """Process uploaded files synchronously and return categorized payloads."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum allowed is {MAX_FILES}.")

    payloads = await read_upload_payloads(files)
    return process_upload_payloads(payloads)


@app.post("/upload/async")
@app.post("/api/upload/async")
async def upload_images_async(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Queue uploaded files for background processing and return a job id."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum allowed is {MAX_FILES}.")

    payloads = await read_upload_payloads(files)
    job_id = create_upload_job(total_files=len(payloads))
    job_executor.submit(run_upload_job, job_id, payloads)

    return {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(payloads),
        "status_endpoint": f"/jobs/{job_id}",
        "api_status_endpoint": f"/api/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}")
@app.get("/api/jobs/{job_id}")
def get_upload_job(job_id: str) -> Dict[str, Any]:
    """Return background job state and results when complete."""
    snapshot = get_upload_job_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Job not found.")
    return snapshot
