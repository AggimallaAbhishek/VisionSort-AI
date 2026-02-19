"""Supabase client helpers for storage and metadata persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from supabase import Client, create_client


@dataclass
class SupabaseConfig:
    url: str
    key: str


class SupabaseService:
    """Thin wrapper around Supabase storage and table operations."""

    def __init__(self) -> None:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()

        self.client: Optional[Client] = None
        self.enabled = False

        if url and key:
            self.client = create_client(url, key)
            self.enabled = True

    def upload_image(self, bucket: str, path: str, data: bytes, content_type: str) -> Optional[str]:
        """Upload bytes to Supabase storage and return bucket path."""
        if not self.enabled or not self.client:
            return None

        self.client.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "false"},
        )
        return f"{bucket}/{path}"

    def insert_image_metadata(self, row: Dict[str, Any]) -> None:
        """Insert image metadata into the `images` table."""
        if not self.enabled or not self.client:
            return

        self.client.table("images").insert(row).execute()
