"""VisionSort AI backend service."""

from __future__ import annotations

import base64
import json
import secrets
from collections import deque
from contextvars import ContextVar
from io import BytesIO
import logging
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from aws_client import AWSService
from utils.blur_detection import detect_blur
from utils.brightness_check import analyze_brightness
from utils.duplicate_check import is_duplicate

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _detect_total_memory_mb() -> int:
    """Best-effort total system memory detection in MB."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
            return int((page_size * phys_pages) / (1024 * 1024))
    except Exception:
        return 0
    return 0


BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100"))
DUPLICATE_HASH_DISTANCE = int(os.getenv("DUPLICATE_HASH_DISTANCE", "5"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "1024"))
PREVIEW_MAX_WIDTH = int(os.getenv("PREVIEW_MAX_WIDTH", "640"))
PREVIEW_JPEG_QUALITY = max(30, min(95, int(os.getenv("PREVIEW_JPEG_QUALITY", "70"))))
INCLUDE_PREVIEW_DATA_URL = os.getenv("INCLUDE_PREVIEW_DATA_URL", "true").lower() == "true"
PREVIEW_INLINE_MAX_FILES = max(1, int(os.getenv("PREVIEW_INLINE_MAX_FILES", "20")))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_REQUEST_TOTAL_MB = max(1, int(os.getenv("MAX_REQUEST_TOTAL_MB", "40")))
MAX_REQUEST_TOTAL_BYTES = MAX_REQUEST_TOTAL_MB * 1024 * 1024
MAX_FILES = int(os.getenv("MAX_FILES", "50"))
SPOOL_UPLOADS_TO_DISK = os.getenv("SPOOL_UPLOADS_TO_DISK", "true").lower() == "true"
ENABLE_AI_LABEL = os.getenv("ENABLE_AI_LABEL", "true").lower() == "true"
AI_DISABLE_ABOVE_FILES = max(0, int(os.getenv("AI_DISABLE_ABOVE_FILES", "24")))
DISABLE_AI_ON_LOW_MEMORY = os.getenv("DISABLE_AI_ON_LOW_MEMORY", "true").lower() == "true"
LOW_MEMORY_RAM_MB_THRESHOLD = max(0, int(os.getenv("LOW_MEMORY_RAM_MB_THRESHOLD", "700")))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "anonymous")
RENAMED_FILE_PREFIX = (
    re.sub(r"[^a-zA-Z0-9_-]", "_", os.getenv("RENAMED_FILE_PREFIX", "vin_img").strip()) or "vin_img"
)
AI_ASSISTED_STATUS = os.getenv("AI_ASSISTED_STATUS", "true").lower() == "true"
AI_FASTPATH_SKIP_STRONG_RULES = os.getenv("AI_FASTPATH_SKIP_STRONG_RULES", "true").lower() == "true"
AI_MIN_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("AI_MIN_CONFIDENCE", "0.70"))))
AI_PROMOTE_GOOD_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("AI_PROMOTE_GOOD_CONFIDENCE", "0.90"))))
AI_BORDERLINE_FACTOR = max(0.5, min(1.2, float(os.getenv("AI_BORDERLINE_FACTOR", "0.90"))))
DARK_PROMOTE_MIN_BRIGHTNESS = max(0.0, min(255.0, float(os.getenv("DARK_PROMOTE_MIN_BRIGHTNESS", "45"))))
OVEREXPOSED_PROMOTE_MAX_BRIGHTNESS = max(
    0.0, min(255.0, float(os.getenv("OVEREXPOSED_PROMOTE_MAX_BRIGHTNESS", "210")))
)
PERSIST_WORKERS = max(1, int(os.getenv("PERSIST_WORKERS", "6")))
UPLOAD_JOB_WORKERS = max(1, int(os.getenv("UPLOAD_JOB_WORKERS", "2")))
JOB_RETENTION_MINUTES = max(5, int(os.getenv("JOB_RETENTION_MINUTES", "60")))
PERSIST_WORKERS_MAX_SAFE = max(0, int(os.getenv("PERSIST_WORKERS_MAX_SAFE", "0")))
UPLOAD_JOB_WORKERS_MAX_SAFE = max(0, int(os.getenv("UPLOAD_JOB_WORKERS_MAX_SAFE", "0")))
PERSIST_TASK_TIMEOUT_SECONDS = max(2, int(os.getenv("PERSIST_TASK_TIMEOUT_SECONDS", "30")))
PERSIST_OVERALL_TIMEOUT_SECONDS = max(15, int(os.getenv("PERSIST_OVERALL_TIMEOUT_SECONDS", "90")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
RATE_LIMIT_MAX_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30")))
MAX_ZIP_ITEMS = max(1, int(os.getenv("MAX_ZIP_ITEMS", "400")))
MAX_ZIP_SOURCE_BYTES_MB = max(10, int(os.getenv("MAX_ZIP_SOURCE_BYTES_MB", "500")))
HSTS_MAX_AGE_SECONDS = max(0, int(os.getenv("HSTS_MAX_AGE_SECONDS", "31536000")))
SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "").strip()
ENABLE_API_DOCS = os.getenv("ENABLE_API_DOCS", "false").lower() == "true"
EXPOSE_HEALTH_DETAILS = os.getenv("EXPOSE_HEALTH_DETAILS", "true").lower() == "true"
allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "*").strip()
allowed_hosts = [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]
if not allowed_hosts:
    allowed_hosts = ["*"]
MAX_ZIP_SOURCE_BYTES = MAX_ZIP_SOURCE_BYTES_MB * 1024 * 1024
CATEGORY_KEYS = ("good", "blurry", "dark", "overexposed", "duplicates")
ZIP_SOURCES = {"processed", "original"}

if PERSIST_WORKERS_MAX_SAFE > 0:
    PERSIST_WORKERS = min(PERSIST_WORKERS, PERSIST_WORKERS_MAX_SAFE)
if UPLOAD_JOB_WORKERS_MAX_SAFE > 0:
    UPLOAD_JOB_WORKERS = min(UPLOAD_JOB_WORKERS, UPLOAD_JOB_WORKERS_MAX_SAFE)

TOTAL_MEMORY_MB = _detect_total_memory_mb()
if ENABLE_AI_LABEL and DISABLE_AI_ON_LOW_MEMORY and TOTAL_MEMORY_MB and TOTAL_MEMORY_MB < LOW_MEMORY_RAM_MB_THRESHOLD:
    logger.warning(
        "Low-memory guard active (%s MB < %s MB). Disabling AI inference to prevent OOM.",
        TOTAL_MEMORY_MB,
        LOW_MEMORY_RAM_MB_THRESHOLD,
    )
    ENABLE_AI_LABEL = False

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/gif",
}
EXTENSION_TO_CONTENT_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".gif": "image/gif",
}
CONTENT_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/pjpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/gif": ".gif",
}

app = FastAPI(
    title="VisionSort AI",
    version="1.1.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)

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
if allowed_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

aws_service = AWSService()
persist_executor = ThreadPoolExecutor(max_workers=PERSIST_WORKERS)
job_executor = ThreadPoolExecutor(max_workers=UPLOAD_JOB_WORKERS)

UploadPayload = Dict[str, Any]
ProgressHook = Callable[[int, int, str, str], None]
upload_jobs: Dict[str, Dict[str, Any]] = {}
upload_jobs_lock = Lock()
predict_quality_func: Callable[[Image.Image], Any] | None = None
predict_quality_init_attempted = False
rate_limit_events: Dict[str, deque[float]] = {}
rate_limit_lock = Lock()
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

REQUEST_ID_HEADER = "X-Request-ID"
RATE_LIMITED_PATHS = {
    "/upload",
    "/api/upload",
    "/upload/async",
    "/api/upload/async",
    "/download/zip",
    "/api/download/zip",
}
AUTH_REQUIRED_PATHS = set(RATE_LIMITED_PATHS)
AUTH_REQUIRED_PREFIXES = ("/jobs/", "/api/jobs/")
HTTP_ERROR_CODE_MAP = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "FILE_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    503: "SERVICE_UNAVAILABLE",
}


def current_request_id() -> str:
    """Return current request ID from context."""
    request_id = (request_id_ctx.get() or "").strip()
    return request_id or "unknown"


def extract_client_ip(request: Request) -> str:
    """Extract best-effort client IP from proxy headers or request client."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def is_rate_limited_request(request: Request) -> bool:
    """Return True when request should be subject to upload rate limiting."""
    return request.method.upper() == "POST" and request.url.path in RATE_LIMITED_PATHS


def consume_rate_limit_token(client_ip: str) -> Tuple[bool, int]:
    """Consume one request token, returning (allowed, retry_after_seconds)."""
    now = time.time()

    with rate_limit_lock:
        bucket = rate_limit_events.setdefault(client_ip, deque())
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after_seconds = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            return False, retry_after_seconds

        bucket.append(now)
        return True, 0




def requires_api_key(request: Request) -> bool:
    """Return True when request requires API key authentication."""
    if not BACKEND_API_KEY:
        return False
    if request.method.upper() == "OPTIONS":
        return False
    path = request.url.path
    return path in AUTH_REQUIRED_PATHS or path.startswith(AUTH_REQUIRED_PREFIXES)


def extract_api_key(request: Request) -> str:
    """Read API key from x-api-key or bearer token header."""
    direct_key = (request.headers.get("x-api-key") or "").strip()
    if direct_key:
        return direct_key

    auth_header = (request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return ""


def is_https_request(request: Request) -> bool:
    """Return True when request came via HTTPS."""
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or proto == "https"


def apply_security_headers(response: Response, request: Request) -> None:
    """Attach hardened baseline security headers."""
    if not SECURITY_HEADERS_ENABLED:
        return

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
    )
    if HSTS_MAX_AGE_SECONDS > 0 and is_https_request(request):
        response.headers.setdefault(
            "Strict-Transport-Security", f"max-age={HSTS_MAX_AGE_SECONDS}; includeSubDomains"
        )


def build_error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build structured error response payload."""
    error: Dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id or current_request_id(),
    }
    if details is not None:
        error["details"] = details
    return {"error": error}


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """Attach request IDs, enforce basic upload rate limit, and log requests."""
    request_id = (request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex).strip()[:128]
    context_token = request_id_ctx.set(request_id)
    request.state.request_id = request_id

    method = request.method
    path = request.url.path
    client_ip = extract_client_ip(request)
    started_at = time.perf_counter()
    status_code = 500
    rate_limited = False

    try:
        if requires_api_key(request):
            provided_api_key = extract_api_key(request)
            if not provided_api_key or not secrets.compare_digest(provided_api_key, BACKEND_API_KEY):
                response = JSONResponse(
                    status_code=401,
                    content=build_error_payload(
                        code="UNAUTHORIZED",
                        message="Missing or invalid API key.",
                        request_id=request_id,
                        details={"status_code": 401},
                    ),
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                response.headers[REQUEST_ID_HEADER] = request_id
                apply_security_headers(response, request)
                status_code = response.status_code
                return response

        if is_rate_limited_request(request):
            allowed, retry_after_seconds = consume_rate_limit_token(client_ip)
            if not allowed:
                rate_limited = True
                response = JSONResponse(
                    status_code=429,
                    content=build_error_payload(
                        code="RATE_LIMIT_EXCEEDED",
                        message=(
                            f"Too many upload requests. Maximum {RATE_LIMIT_MAX_REQUESTS} "
                            f"requests per {RATE_LIMIT_WINDOW_SECONDS} seconds."
                        ),
                        request_id=request_id,
                        details={
                            "retry_after_seconds": retry_after_seconds,
                            "limit": RATE_LIMIT_MAX_REQUESTS,
                            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                        },
                    ),
                )
                response.headers["Retry-After"] = str(retry_after_seconds)
                response.headers[REQUEST_ID_HEADER] = request_id
                apply_security_headers(response, request)
                status_code = response.status_code
                return response

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        apply_security_headers(response, request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "request_id=%s method=%s path=%s status=%s ip=%s rate_limited=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            status_code,
            client_ip,
            rate_limited,
            duration_ms,
        )
        request_id_ctx.reset(context_token)


@app.get("/")
@app.get("/api")
def root() -> Dict[str, Any]:
    """Health endpoint with non-sensitive service status."""
    if not EXPOSE_HEALTH_DETAILS:
        return {
            "message": "VisionSort AI backend is running.",
            "request_id": current_request_id(),
        }

    return {
        "message": "VisionSort AI backend is running.",
        "service": aws_service.health_snapshot(),
        "runtime": {
            "total_memory_mb": TOTAL_MEMORY_MB or None,
        },
        "cors": {
            "allow_origins": allow_origins,
            "allow_origin_regex": allow_origin_regex,
        },
        "workers": {
            "persist_workers": PERSIST_WORKERS,
            "upload_job_workers": UPLOAD_JOB_WORKERS,
            "persist_workers_max_safe": PERSIST_WORKERS_MAX_SAFE,
            "upload_job_workers_max_safe": UPLOAD_JOB_WORKERS_MAX_SAFE,
            "persist_task_timeout_seconds": PERSIST_TASK_TIMEOUT_SECONDS,
            "persist_overall_timeout_seconds": PERSIST_OVERALL_TIMEOUT_SECONDS,
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_request_total_mb": MAX_REQUEST_TOTAL_MB,
            "max_files": MAX_FILES,
            "max_image_width": MAX_IMAGE_WIDTH,
            "preview_max_width": PREVIEW_MAX_WIDTH,
            "preview_jpeg_quality": PREVIEW_JPEG_QUALITY,
            "include_preview_data_url": INCLUDE_PREVIEW_DATA_URL,
            "preview_inline_max_files": PREVIEW_INLINE_MAX_FILES,
            "spool_uploads_to_disk": SPOOL_UPLOADS_TO_DISK,
            "max_zip_items": MAX_ZIP_ITEMS,
            "max_zip_source_bytes_mb": MAX_ZIP_SOURCE_BYTES_MB,
            "renamed_file_prefix": RENAMED_FILE_PREFIX,
            "rate_limit_max_requests": RATE_LIMIT_MAX_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
        "ai": {
            "enabled": ENABLE_AI_LABEL,
            "disabled_on_low_memory": DISABLE_AI_ON_LOW_MEMORY,
            "low_memory_ram_mb_threshold": LOW_MEMORY_RAM_MB_THRESHOLD,
            "assisted_status": AI_ASSISTED_STATUS,
            "fastpath_skip_strong_rules": AI_FASTPATH_SKIP_STRONG_RULES,
            "disable_above_files": AI_DISABLE_ABOVE_FILES,
            "min_confidence": AI_MIN_CONFIDENCE,
            "promote_good_confidence": AI_PROMOTE_GOOD_CONFIDENCE,
        },
        "security": {
            "api_key_required": bool(BACKEND_API_KEY),
            "headers_enabled": SECURITY_HEADERS_ENABLED,
            "allowed_hosts": allowed_hosts,
            "api_docs_enabled": ENABLE_API_DOCS,
            "expose_health_details": EXPOSE_HEALTH_DETAILS,
            "hsts_max_age_seconds": HSTS_MAX_AGE_SECONDS,
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured payloads for HTTP errors."""
    request_id = getattr(request.state, "request_id", current_request_id())
    error_code = HTTP_ERROR_CODE_MAP.get(exc.status_code, "HTTP_ERROR")

    if isinstance(exc.detail, str):
        message = exc.detail
        details: Dict[str, Any] = {"status_code": exc.status_code}
    elif isinstance(exc.detail, dict):
        details = dict(exc.detail)
        message = str(details.get("message") or details.get("detail") or "Request failed.")
        details.setdefault("status_code", exc.status_code)
    else:
        message = str(exc.detail)
        details = {"status_code": exc.status_code}

    response = JSONResponse(
        status_code=exc.status_code,
        content=build_error_payload(
            code=error_code,
            message=message,
            request_id=request_id,
            details=details,
        ),
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return structured payloads for validation errors."""
    request_id = getattr(request.state, "request_id", current_request_id())
    response = JSONResponse(
        status_code=422,
        content=build_error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            request_id=request_id,
            details={"status_code": 422, "errors": exc.errors()},
        ),
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all error handler for consistent API error responses."""
    request_id = getattr(request.state, "request_id", current_request_id())
    logger.exception("Unhandled exception request_id=%s path=%s", request_id, request.url.path)

    response = JSONResponse(
        status_code=500,
        content=build_error_payload(
            code="INTERNAL_ERROR",
            message="Internal server error.",
            request_id=request_id,
        ),
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safer storage keys."""
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return clean_name or f"image_{uuid.uuid4().hex}.jpg"


def sanitize_archive_filename(filename: str, fallback: str) -> str:
    """Sanitize archive filenames and prevent zip path traversal."""
    original = (filename or "").strip()
    base_name = Path(original).name  # strip any path components
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", base_name)
    clean_name = clean_name.strip("._")
    if not clean_name:
        clean_name = fallback

    stem = Path(clean_name).stem[:100] or Path(fallback).stem or "file"
    ext = Path(clean_name).suffix[:12]
    if not ext:
        ext = Path(fallback).suffix or ".jpg"
    return f"{stem}{ext}"


def normalize_zip_categories(categories: Any, results: Dict[str, Any]) -> List[str]:
    """Normalize user-requested categories for server-side ZIP generation."""
    if not categories:
        return [category for category in CATEGORY_KEYS if isinstance(results.get(category), list)]

    if not isinstance(categories, list):
        raise HTTPException(status_code=400, detail="`categories` must be a list of category names.")

    normalized: List[str] = []
    for value in categories:
        category = str(value or "").strip().lower()
        if category in CATEGORY_KEYS and category not in normalized:
            normalized.append(category)

    if not normalized:
        raise HTTPException(status_code=400, detail="No valid categories requested for ZIP download.")
    return normalized


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
    if image is not None:
        return image

    # Fallback decoder for images that OpenCV fails to parse directly.
    try:
        with Image.open(BytesIO(raw_bytes)) as pil_img:
            rgb = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError("Failed to decode image bytes.") from exc


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


def choose_rule_status(blur_score: float, brightness_level: str, duplicate: bool) -> str:
    """Choose deterministic rule-based category from image metrics."""
    if duplicate:
        return "duplicates"
    if brightness_level == "dark":
        return "dark"
    if brightness_level == "overexposed":
        return "overexposed"
    if blur_score < BLUR_THRESHOLD:
        return "blurry"
    return "good"


def choose_final_status(
    *,
    blur_score: float,
    brightness_level: str,
    brightness_value: float,
    duplicate: bool,
    ai_label: str,
    ai_confidence: float,
) -> Tuple[str, str]:
    """Choose final status by combining deterministic rules with high-confidence AI."""
    rule_status = choose_rule_status(blur_score, brightness_level, duplicate)

    if not AI_ASSISTED_STATUS or ai_label in {
        "disabled",
        "model_unavailable",
        "disabled_large_batch",
        "skipped_fastpath",
    }:
        return rule_status, "rule"
    if ai_confidence < AI_MIN_CONFIDENCE:
        return rule_status, "rule"
    if rule_status == "duplicates":
        return rule_status, "rule"

    allowed_ai_labels = {"good", "blurry", "dark", "overexposed"}
    if ai_label not in allowed_ai_labels:
        return rule_status, "rule"

    # If rules say "good" but AI is confident it's an issue, allow downgrade.
    if rule_status == "good" and ai_label != "good":
        return ai_label, "ai_override"

    # If rules mark an issue but AI strongly predicts "good", only promote when borderline.
    if rule_status != "good" and ai_label == "good" and ai_confidence >= AI_PROMOTE_GOOD_CONFIDENCE:
        if rule_status == "blurry" and blur_score >= BLUR_THRESHOLD * AI_BORDERLINE_FACTOR:
            return "good", "ai_override"
        if rule_status == "dark" and brightness_value >= DARK_PROMOTE_MIN_BRIGHTNESS:
            return "good", "ai_override"
        if rule_status == "overexposed" and brightness_value <= OVEREXPOSED_PROMOTE_MAX_BRIGHTNESS:
            return "good", "ai_override"

    return rule_status, "rule"


def should_skip_ai_inference(*, rule_status: str, blur_score: float, brightness_value: float, duplicate: bool) -> bool:
    """Return True when AI inference can be safely skipped for performance."""
    if duplicate:
        return True
    if not AI_FASTPATH_SKIP_STRONG_RULES:
        return False

    if rule_status == "blurry" and blur_score < BLUR_THRESHOLD * AI_BORDERLINE_FACTOR:
        return True
    if rule_status == "dark" and brightness_value < DARK_PROMOTE_MIN_BRIGHTNESS:
        return True
    if rule_status == "overexposed" and brightness_value > OVEREXPOSED_PROMOTE_MAX_BRIGHTNESS:
        return True
    return False


def clamp_percent(value: float) -> float:
    """Clamp quality scores to 0-100 range."""
    return max(0.0, min(100.0, float(value)))


def resolve_extension(content_type: str, file_name: str) -> str:
    """Resolve extension from filename, then fallback to MIME type."""
    ext = Path(file_name).suffix.lower()
    if ext in EXTENSION_TO_CONTENT_TYPE:
        return ext
    return CONTENT_TYPE_TO_EXTENSION.get(content_type.lower(), ".jpg")


def calculate_brightness_value(image: np.ndarray) -> float:
    """Return average brightness value on HSV V channel (0-255)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def calculate_blur_quality_score(blur_score: float) -> float:
    """Map blur variance to a 0-100 quality score."""
    scale = max(BLUR_THRESHOLD * 3.0, 1.0)
    return round(clamp_percent((blur_score / scale) * 100.0), 2)


def calculate_brightness_quality_score(brightness_value: float) -> float:
    """Map brightness value to a 0-100 quality score around a normal range."""
    if brightness_value < 50:
        return round(clamp_percent((brightness_value / 50.0) * 69.0), 2)

    if brightness_value > 200:
        return round(clamp_percent(((255.0 - brightness_value) / 55.0) * 69.0), 2)

    center = 125.0
    normal_half_span = 75.0
    normalized_distance = abs(brightness_value - center) / normal_half_span
    return round(clamp_percent(70.0 + (1.0 - normalized_distance) * 30.0), 2)


def build_item_payload(
    original_file_name: str,
    renamed_file_name: str,
    blur_score: float,
    blur_quality_score: float,
    brightness_level: str,
    brightness_value: float,
    brightness_score: float,
    ai_label: str,
    ai_confidence: float,
    final_status: str,
    status_source: str,
    preview_data_url: str | None,
    storage_path: str | None,
    processed_storage_path: str | None,
    storage_folder: str,
) -> Dict[str, Any]:
    """Build response payload item for frontend rendering."""
    return {
        "file_name": renamed_file_name,
        "original_file_name": original_file_name,
        "renamed_file_name": renamed_file_name,
        "blur_score": round(blur_score, 2),
        "blur_quality_score": round(blur_quality_score, 2),
        "brightness_level": brightness_level,
        "brightness_value": round(brightness_value, 2),
        "brightness_score": round(brightness_score, 2),
        "ai_label": ai_label,
        "ai_confidence": round(clamp_percent(ai_confidence * 100.0), 2),
        "final_status": final_status,
        "status_source": status_source,
        "storage_folder": storage_folder,
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


def predict_quality_label(image: Image.Image) -> Tuple[str, float]:
    """Return AI label + confidence using lazy model initialization."""
    if not ENABLE_AI_LABEL:
        return "disabled", 0.0

    global predict_quality_func, predict_quality_init_attempted

    if predict_quality_func is None and not predict_quality_init_attempted:
        predict_quality_init_attempted = True
        try:
            from utils.model_predict import predict_quality_with_confidence as loaded_predict_quality

            predict_quality_func = loaded_predict_quality
        except Exception as exc:
            logger.exception("AI model initialization failed: %s", exc)
            return "model_unavailable", 0.0

    if predict_quality_func is None:
        return "model_unavailable", 0.0

    try:
        prediction = predict_quality_func(image)
        if isinstance(prediction, dict):
            label = str(prediction.get("label", "model_unavailable"))
            confidence = float(prediction.get("confidence", 0.0) or 0.0)
            return label, max(0.0, min(1.0, confidence))
        if isinstance(prediction, str):
            return prediction, 0.0
        return "model_unavailable", 0.0
    except Exception as exc:
        logger.exception("AI inference failed: %s", exc)
        return "model_unavailable", 0.0


def build_results_template() -> Dict[str, List[Dict[str, Any]]]:
    """Create an empty response template for all categories."""
    return {category: [] for category in CATEGORY_KEYS}


def persist_artifacts(
    renamed_file_name: str,
    final_status: str,
    request_batch_id: str,
    object_date_prefix: str,
    content_type: str,
    raw_bytes: bytes,
    preview_bytes: bytes,
) -> Tuple[str | None, str | None]:
    """Upload original and processed artifacts to S3 inside categorized folders."""
    status_prefix = final_status.lower()
    object_folder = f"{status_prefix}/{object_date_prefix}/{request_batch_id}"
    processed_name = f"{Path(renamed_file_name).stem}_processed.jpg"
    original_object_path = f"{object_folder}/{renamed_file_name}"
    processed_object_path = f"{object_folder}/{processed_name}"

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
        logger.exception("S3 original upload failed for %s: %s", renamed_file_name, exc)

    try:
        processed_storage_path = aws_service.upload_image(
            path=processed_object_path,
            data=preview_bytes,
            content_type="image/jpeg",
            bucket="processed",
        )
    except Exception as exc:
        logger.exception("S3 processed upload failed for %s: %s", renamed_file_name, exc)

    return uploads_storage_path, processed_storage_path


async def read_upload_payloads(
    files: List[UploadFile],
    *,
    spool_to_disk: bool = SPOOL_UPLOADS_TO_DISK,
) -> List[UploadPayload]:
    """Read and validate uploads once so they can be processed sync/async."""
    payloads: List[UploadPayload] = []
    total_request_bytes = 0

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
                    "temp_path": None,
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
                    "temp_path": None,
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
                    "temp_path": None,
                    "validation_error": "file_too_large",
                }
            )
            continue

        total_request_bytes += len(raw_bytes)
        if total_request_bytes > MAX_REQUEST_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Total upload size exceeded limit of {MAX_REQUEST_TOTAL_MB} MB. "
                    "Upload fewer files per request."
                ),
            )

        temp_path: str | None = None
        if spool_to_disk:
            suffix = resolve_extension(content_type, file_name)
            with tempfile.NamedTemporaryFile(prefix="visionsort_upload_", suffix=suffix, delete=False) as temp_file:
                temp_file.write(raw_bytes)
                temp_path = temp_file.name
            raw_bytes = b""

        payloads.append(
            {
                "file_name": file_name,
                "content_type": content_type,
                "raw_bytes": raw_bytes,
                "temp_path": temp_path,
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
    active_persistence_tasks: List[Tuple[Future[Tuple[str | None, str | None]], Dict[str, Any], str]] = []
    processed_count = 0
    total_files = len(payloads)
    valid_file_count = sum(1 for payload in payloads if not payload.get("validation_error"))
    include_inline_preview = INCLUDE_PREVIEW_DATA_URL and valid_file_count <= PREVIEW_INLINE_MAX_FILES
    ai_disabled_for_batch = AI_DISABLE_ABOVE_FILES > 0 and valid_file_count > AI_DISABLE_ABOVE_FILES
    s3_persistence_enabled = aws_service.s3_enabled
    request_batch_id = f"req_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    object_date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")

    def resolve_persistence_task(
        task: Tuple[Future[Tuple[str | None, str | None]], Dict[str, Any], str],
        *,
        timeout_seconds: float,
    ) -> None:
        future, item, file_name = task
        if timeout_seconds <= 0:
            future.cancel()
            logger.error("Persistence task timed out (overall deadline) for %s", file_name)
            return
        try:
            uploads_storage_path, processed_storage_path = future.result(timeout=timeout_seconds)
            item["storage_path"] = uploads_storage_path
            item["processed_storage_path"] = processed_storage_path
        except FutureTimeoutError:
            future.cancel()
            logger.error("Persistence task timed out for %s", file_name)
        except Exception as exc:
            logger.exception("Persistence task failed for %s: %s", file_name, exc)

    for index, payload in enumerate(payloads, start=1):
        original_file_name = payload["file_name"]
        temp_path = payload.get("temp_path")
        try:
            validation_error = payload.get("validation_error")
            if validation_error:
                continue

            raw_bytes: bytes
            if temp_path:
                raw_bytes = Path(temp_path).read_bytes()
            else:
                raw_bytes = payload["raw_bytes"]
            content_type: str = payload["content_type"]

            image = decode_image(raw_bytes)
            resized_image = resize_image(image, MAX_IMAGE_WIDTH)
            pil_image = bgr_to_pil(resized_image)

            blur_score = detect_blur(resized_image)
            blur_quality_score = calculate_blur_quality_score(blur_score)
            brightness_value = calculate_brightness_value(resized_image)
            brightness_score = calculate_brightness_quality_score(brightness_value)
            brightness_level = analyze_brightness(resized_image)
            duplicate = is_duplicate(pil_image, seen_hashes, threshold=DUPLICATE_HASH_DISTANCE)
            rule_status = choose_rule_status(blur_score, brightness_level, duplicate)
            if ai_disabled_for_batch:
                ai_label, ai_confidence = "disabled_large_batch", 0.0
            elif should_skip_ai_inference(
                rule_status=rule_status,
                blur_score=blur_score,
                brightness_value=brightness_value,
                duplicate=duplicate,
            ):
                ai_label, ai_confidence = "skipped_fastpath", 0.0
            else:
                ai_label, ai_confidence = predict_quality_label(pil_image)
            final_status, status_source = choose_final_status(
                blur_score=blur_score,
                brightness_level=brightness_level,
                brightness_value=brightness_value,
                duplicate=duplicate,
                ai_label=ai_label,
                ai_confidence=ai_confidence,
            )
            processed_count += 1

            renamed_extension = resolve_extension(content_type, original_file_name)
            renamed_file_name = f"{RENAMED_FILE_PREFIX}{processed_count}{renamed_extension}"
            storage_folder = f"{final_status}/{object_date_prefix}/{request_batch_id}"

            preview_bytes = b""
            preview_data_url = None
            if include_inline_preview or s3_persistence_enabled:
                preview_image = resize_image(resized_image, PREVIEW_MAX_WIDTH)
                preview_bytes = encode_preview_jpeg(preview_image, quality=PREVIEW_JPEG_QUALITY)
                if include_inline_preview:
                    preview_data_url = f"data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode('utf-8')}"
            item = build_item_payload(
                original_file_name=original_file_name,
                renamed_file_name=renamed_file_name,
                blur_score=blur_score,
                blur_quality_score=blur_quality_score,
                brightness_level=brightness_level,
                brightness_value=brightness_value,
                brightness_score=brightness_score,
                ai_label=ai_label,
                ai_confidence=ai_confidence,
                final_status=final_status,
                status_source=status_source,
                storage_folder=storage_folder,
                preview_data_url=preview_data_url,
                storage_path=None,
                processed_storage_path=None,
            )
            results[final_status].append(item)
            if aws_service.db_enabled:
                metadata_rows.append(
                    build_metadata_row(renamed_file_name, blur_score, brightness_level, ai_label, final_status)
                )

            if s3_persistence_enabled:
                active_persistence_tasks.append(
                    (
                        persist_executor.submit(
                            persist_artifacts,
                            renamed_file_name=renamed_file_name,
                            final_status=final_status,
                            request_batch_id=request_batch_id,
                            object_date_prefix=object_date_prefix,
                            content_type=content_type,
                            raw_bytes=raw_bytes,
                            preview_bytes=preview_bytes,
                        ),
                        item,
                        renamed_file_name,
                    )
                )
                if len(active_persistence_tasks) >= PERSIST_WORKERS:
                    resolve_persistence_task(
                        active_persistence_tasks.pop(0),
                        timeout_seconds=float(PERSIST_TASK_TIMEOUT_SECONDS),
                    )

        except ValueError as exc:
            logger.warning("Skipping invalid image %s: %s", original_file_name, exc)
        except Exception as exc:
            logger.exception("Unexpected processing error for %s: %s", original_file_name, exc)
        finally:
            payload["raw_bytes"] = b""
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to remove temp upload file %s: %s", temp_path, exc)
            if progress_hook:
                progress_hook(index, total_files, original_file_name, "processing")

    if progress_hook and total_files:
        progress_hook(total_files, total_files, "", "finalizing")

    if active_persistence_tasks:
        deadline = time.perf_counter() + float(PERSIST_OVERALL_TIMEOUT_SECONDS)
        for task in active_persistence_tasks:
            remaining = deadline - time.perf_counter()
            resolve_persistence_task(
                task,
                timeout_seconds=min(float(PERSIST_TASK_TIMEOUT_SECONDS), max(0.0, remaining)),
            )

    if metadata_rows and aws_service.db_enabled:
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
async def upload_images(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Process uploaded files synchronously and return categorized payloads."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum allowed is {MAX_FILES}.")

    payloads = await read_upload_payloads(files)
    results = process_upload_payloads(payloads)
    results["request_id"] = current_request_id()
    return results


@app.post("/download/zip")
@app.post("/api/download/zip")
async def download_results_zip(request: Request) -> Response:
    """Build and return a ZIP file from persisted S3 objects in analysis results."""
    if not aws_service.s3_enabled:
        raise HTTPException(status_code=503, detail="S3 is not configured for ZIP downloads.")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise HTTPException(status_code=400, detail="`results` object is required in request payload.")

    categories = normalize_zip_categories(payload.get("categories"), results)
    source = str(payload.get("source", "processed")).strip().lower()
    if source not in ZIP_SOURCES:
        raise HTTPException(status_code=400, detail=f"`source` must be one of: {', '.join(sorted(ZIP_SOURCES))}.")

    include_manifest = bool(payload.get("include_manifest", True))
    requested_zip_name = str(payload.get("zip_name", "visionsort_folders")).strip()
    requested_zip_name = sanitize_archive_filename(requested_zip_name, "visionsort_folders.zip")
    zip_base_name = requested_zip_name[:-4] if requested_zip_name.lower().endswith(".zip") else requested_zip_name
    zip_base_name = zip_base_name or "visionsort_folders"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    download_file_name = f"{zip_base_name}_{timestamp}.zip"

    downloaded_count = 0
    scanned_item_count = 0
    downloaded_source_bytes = 0
    archive_names: set[str] = set()
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "request_id": current_request_id(),
        "source": source,
        "categories": {},
        "limits": {
            "max_zip_items": MAX_ZIP_ITEMS,
            "max_source_bytes_mb": MAX_ZIP_SOURCE_BYTES_MB,
        },
    }
    source_field = "processed_storage_path" if source == "processed" else "storage_path"

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for category in categories:
            raw_items = results.get(category, [])
            if not isinstance(raw_items, list):
                continue

            category_manifest = {"requested": len(raw_items), "downloaded": 0, "skipped": []}
            manifest["categories"][category] = category_manifest

            if not raw_items:
                archive.writestr(f"{category}/README.txt", "No analyzed images in this category.")
                continue

            for index, raw_item in enumerate(raw_items, start=1):
                scanned_item_count += 1
                if scanned_item_count > MAX_ZIP_ITEMS:
                    category_manifest["skipped"].append(
                        {
                            "index": index,
                            "reason": f"Exceeded max ZIP items limit ({MAX_ZIP_ITEMS}).",
                        }
                    )
                    continue

                if not isinstance(raw_item, dict):
                    category_manifest["skipped"].append({"index": index, "reason": "Invalid result item payload."})
                    continue

                s3_uri = str(raw_item.get(source_field) or "").strip()
                if not s3_uri:
                    category_manifest["skipped"].append(
                        {"index": index, "reason": f"Missing `{source_field}` in result item."}
                    )
                    continue

                try:
                    object_bytes = aws_service.download_s3_uri(s3_uri)
                except Exception as exc:
                    logger.warning("ZIP download skipped for %s (%s): %s", category, s3_uri, exc)
                    category_manifest["skipped"].append(
                        {"index": index, "reason": f"Unable to download source object: {s3_uri}"}
                    )
                    continue

                downloaded_source_bytes += len(object_bytes)
                if downloaded_source_bytes > MAX_ZIP_SOURCE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"ZIP source size exceeded limit of {MAX_ZIP_SOURCE_BYTES_MB} MB. "
                            "Reduce selected images and retry."
                        ),
                    )

                original_name = str(raw_item.get("renamed_file_name") or raw_item.get("file_name") or "").strip()
                fallback_name = f"{category}_{index:03d}.jpg"
                archive_name = sanitize_archive_filename(original_name, fallback_name)
                archive_path = f"{category}/{archive_name}"

                suffix = 1
                while archive_path in archive_names:
                    stem = Path(archive_name).stem
                    ext = Path(archive_name).suffix or ".jpg"
                    archive_name = sanitize_archive_filename(f"{stem}_{suffix}{ext}", fallback_name)
                    archive_path = f"{category}/{archive_name}"
                    suffix += 1

                archive_names.add(archive_path)
                archive.writestr(archive_path, object_bytes)
                downloaded_count += 1
                category_manifest["downloaded"] += 1

        if include_manifest:
            manifest["downloaded_files"] = downloaded_count
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))

    if downloaded_count == 0:
        raise HTTPException(status_code=400, detail="No downloadable S3 objects found in the provided results.")

    zip_buffer.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="{download_file_name}"',
        REQUEST_ID_HEADER: current_request_id(),
    }
    return Response(content=zip_buffer.getvalue(), media_type="application/zip", headers=headers)


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
        "request_id": current_request_id(),
    }


@app.get("/jobs/{job_id}")
@app.get("/api/jobs/{job_id}")
def get_upload_job(job_id: str) -> Dict[str, Any]:
    """Return background job state and results when complete."""
    snapshot = get_upload_job_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Job not found.")
    snapshot["request_id"] = current_request_id()
    return snapshot
