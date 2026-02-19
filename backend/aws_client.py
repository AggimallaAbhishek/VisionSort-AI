"""AWS helpers for S3 uploads and RDS PostgreSQL metadata storage."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import boto3
import psycopg2
from botocore.exceptions import BotoCoreError, ClientError


class AWSService:
    """Thin wrapper for S3 object uploads and PostgreSQL inserts."""

    def __init__(self) -> None:
        self.region = os.getenv("AWS_REGION", "").strip() or None
        self.uploads_bucket = os.getenv("S3_UPLOADS_BUCKET", "").strip()
        self.database_url = os.getenv("DATABASE_URL", "").strip()

        access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip() or None
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip() or None

        self.s3_client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        self.s3_enabled = bool(self.uploads_bucket)
        self.db_enabled = bool(self.database_url)
        self.enabled = self.s3_enabled or self.db_enabled

    @contextmanager
    def db_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Create and yield a PostgreSQL connection."""
        if not self.db_enabled:
            raise RuntimeError("DATABASE_URL is not configured.")

        conn = psycopg2.connect(self.database_url, sslmode="require")
        try:
            yield conn
        finally:
            conn.close()

    def upload_image(self, path: str, data: bytes, content_type: str) -> Optional[str]:
        """Upload bytes to S3 uploads bucket and return storage path."""
        if not self.s3_enabled:
            return None

        try:
            self.s3_client.put_object(
                Bucket=self.uploads_bucket,
                Key=path,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError):
            raise

        return f"s3://{self.uploads_bucket}/{path}"

    def insert_image_metadata(self, row: Dict[str, Any]) -> None:
        """Insert metadata row into the images table."""
        if not self.db_enabled:
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

        values = (
            row["id"],
            row["user_id"],
            row["file_name"],
            row["blur_score"],
            row["brightness_level"],
            row["ai_label"],
            row["final_status"],
            row["created_at"],
        )

        with self.db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
            conn.commit()
