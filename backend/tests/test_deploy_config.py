"""The deployed configuration must not re-introduce limits the code guards against."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RENDER_YAML = Path(__file__).resolve().parents[2] / "render.yaml"


def render_value(key: str) -> str | None:
    match = re.search(rf'- key: {key}\n\s+value: "?([^"\n]+)"?', RENDER_YAML.read_text())
    return match.group(1) if match else None


def test_zip_source_cap_fits_the_instance():
    """MAX_ZIP_SOURCE_BYTES_MB must leave headroom on a 512 MB dyno."""
    assert int(render_value("MAX_ZIP_SOURCE_BYTES_MB")) <= 64


def test_pixel_budget_is_configured():
    assert 0 < int(render_value("MAX_IMAGE_PIXELS")) <= 50_000_000


def test_storage_token_secret_is_provisioned():
    """Without a stable secret, ZIP links break on every restart."""
    assert "STORAGE_TOKEN_SECRET" in RENDER_YAML.read_text()
