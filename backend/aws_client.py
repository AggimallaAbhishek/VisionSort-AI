"""AWS helpers for S3 uploads and PostgreSQL metadata storage."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Literal, Optional

import boto3
import psycopg2
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    """Return bounded integer env value with safe fallback."""
    raw = (os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default=%s", name, raw, default)
        return default
    return max(min_value, min(max_value, value))


class AWSService:
    """Thin wrapper for S3 object uploads and PostgreSQL inserts."""

    def __init__(self) -> None:
        self.region = os.getenv("AWS_REGION", "").strip() or None
        self.uploads_bucket = os.getenv("S3_UPLOADS_BUCKET", "").strip()
        self.processed_bucket = os.getenv("S3_PROCESSED_BUCKET", "").strip()
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.s3_connect_timeout_seconds = _read_int_env(
            "S3_CONNECT_TIMEOUT_SECONDS", default=3, min_value=1, max_value=30
        )
        self.s3_read_timeout_seconds = _read_int_env(
            "S3_READ_TIMEOUT_SECONDS", default=12, min_value=1, max_value=120
        )
        self.s3_max_attempts = _read_int_env("S3_MAX_ATTEMPTS", default=2, min_value=1, max_value=5)
        self.db_connect_timeout_seconds = _read_int_env(
            "DB_CONNECT_TIMEOUT_SECONDS", default=8, min_value=2, max_value=60
        )
        self.db_statement_timeout_ms = _read_int_env(
            "DB_STATEMENT_TIMEOUT_MS", default=15000, min_value=1000, max_value=120000
        )

        access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip() or None
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip() or None

        self.s3_client: Optional[Any] = None
        if self.uploads_bucket or self.processed_bucket:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    retries={"max_attempts": self.s3_max_attempts, "mode": "standard"},
                    connect_timeout=self.s3_connect_timeout_seconds,
                    read_timeout=self.s3_read_timeout_seconds,
                ),
            )

        self.s3_enabled = bool(self.s3_client)
        self.db_enabled = bool(self.database_url)
        self.enabled = self.s3_enabled or self.db_enabled

    @contextmanager
    def db_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Create and yield a PostgreSQL connection."""
        if not self.db_enabled:
            raise RuntimeError("DATABASE_URL is not configured.")

        conn = psycopg2.connect(
            self.database_url,
            sslmode="require",
            connect_timeout=self.db_connect_timeout_seconds,
            options=f"-c statement_timeout={self.db_statement_timeout_ms}",
        )
        try:
            yield conn
        finally:
            conn.close()

    def _resolve_bucket(self, bucket: Literal["uploads", "processed"]) -> str:
        if bucket == "processed":
            return self.processed_bucket
        return self.uploads_bucket

    def upload_image(
        self,
        path: str,
        data: bytes,
        content_type: str,
        bucket: Literal["uploads", "processed"] = "uploads",
    ) -> Optional[str]:
        """Upload bytes to S3 and return `s3://` path."""
        if not self.s3_enabled or not self.s3_client:
            return None

        bucket_name = self._resolve_bucket(bucket)
        if not bucket_name:
            return None

        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=path,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"S3 upload failed for bucket={bucket_name}, key={path}") from exc

        return f"s3://{bucket_name}/{path}"

    def insert_image_metadata(self, row: Dict[str, Any]) -> None:
        """Insert metadata row into the `images` table."""
        if not self.db_enabled:
            return

        self.insert_many_image_metadata([row])

    def insert_many_image_metadata(self, rows: list[Dict[str, Any]]) -> None:
        """Insert multiple metadata rows in one DB transaction."""
        if not self.db_enabled or not rows:
            return

        query = """
            INSERT INTO images (
                id,
                user_id,
                file_name,
                blur_score,
                brightness_level,
                ai_label,
                final_status,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = [
            (
                row["id"],
                row["user_id"],
                row["file_name"],
                row["blur_score"],
                row["brightness_level"],
                row["ai_label"],
                row["final_status"],
                row["created_at"],
            )
            for row in rows
        ]

        with self.db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.executemany(query, values)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def health_snapshot(self) -> Dict[str, Any]:
        """Return non-sensitive service readiness flags."""
        return {
            "s3_enabled": self.s3_enabled,
            "db_enabled": self.db_enabled,
            "uploads_bucket": self.uploads_bucket or None,
            "processed_bucket": self.processed_bucket or None,
            "region": self.region,
            "s3_connect_timeout_seconds": self.s3_connect_timeout_seconds,
            "s3_read_timeout_seconds": self.s3_read_timeout_seconds,
            "s3_max_attempts": self.s3_max_attempts,
            "db_connect_timeout_seconds": self.db_connect_timeout_seconds,
            "db_statement_timeout_ms": self.db_statement_timeout_ms,
        }
