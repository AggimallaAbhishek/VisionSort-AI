"""P0 regression tests: D2 (event loop starvation) and D4 (ZIP held in memory)."""

from __future__ import annotations

import asyncio
import io
import time
import zipfile

import pytest
from fastapi.responses import StreamingResponse

from conftest import FakeS3Client


async def _heartbeats_during(coro, tick_seconds: float = 0.005) -> tuple[int, float]:
    """Run `coro` and count how many times the event loop got a turn meanwhile."""
    ticks = 0
    running = True

    async def beat():
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(tick_seconds)

    beater = asyncio.create_task(beat())
    await asyncio.sleep(tick_seconds * 4)
    baseline = ticks

    started = time.perf_counter()
    await coro
    elapsed = time.perf_counter() - started

    running = False
    beater.cancel()
    return ticks - baseline, elapsed


class TestD2EventLoopStarvation:
    """CPU-bound and blocking-IO work must be offloaded, not run on the event loop."""

    @pytest.mark.asyncio
    async def test_upload_handler_yields_to_event_loop(self, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "read_upload_payloads", lambda files, **kw: _async([_valid_payload()]))
        monkeypatch.setattr(app_main, "process_upload_payloads", _blocking_processor(0.30))

        ticks, elapsed = await _heartbeats_during(app_main.upload_images(files=[object()]))

        assert ticks >= elapsed / 0.005 * 0.5, (
            f"event loop starved: {ticks} heartbeats over {elapsed:.2f}s "
            f"(expected >= {elapsed / 0.005 * 0.5:.0f})"
        )

    @pytest.mark.asyncio
    async def test_async_upload_inline_fastpath_yields(self, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "read_upload_payloads", lambda files, **kw: _async([_valid_payload()]))
        monkeypatch.setattr(app_main, "payload_size_bytes", lambda payload: 1024)
        monkeypatch.setattr(app_main, "process_upload_payloads", _blocking_processor(0.30))

        ticks, elapsed = await _heartbeats_during(app_main.upload_images_async(files=[object()]))

        assert ticks >= elapsed / 0.005 * 0.5, f"inline fast path starved the loop: {ticks} ticks"

    @pytest.mark.asyncio
    async def test_zip_download_yields_during_s3_reads(self, app_main, monkeypatch):
        """boto3 is synchronous; the ZIP builder must not block the loop on it."""
        service = _enable_s3(app_main)

        def slow_download(s3_uri):
            time.sleep(0.05)
            return b"x" * 512

        monkeypatch.setattr(service, "download_s3_uri", slow_download)
        request = _json_request(app_main, _zip_payload(app_main, count=6))

        ticks, elapsed = await _heartbeats_during(app_main.download_results_zip(request))

        assert ticks >= elapsed / 0.005 * 0.5, f"ZIP S3 reads starved the loop: {ticks} ticks"


class TestD4ZipMemory:
    """The archive must be spooled to disk and streamed, never doubled in RAM."""

    def test_source_cap_fits_a_512mb_dyno(self, app_main):
        assert app_main.MAX_ZIP_SOURCE_BYTES_MB <= 64, (
            "a 500 MB in-memory cap cannot fit a 512 MB instance"
        )

    @pytest.mark.asyncio
    async def test_response_is_streamed(self, app_main, monkeypatch):
        service = _enable_s3(app_main)
        monkeypatch.setattr(service, "download_s3_uri", lambda uri: b"IMAGE-BYTES")

        response = await app_main.download_results_zip(
            _json_request(app_main, _zip_payload(app_main, count=3))
        )

        assert isinstance(response, StreamingResponse), "ZIP must stream, not buffer getvalue()"
        await response.body_iterator.aclose()  # release the spooled file

    @pytest.mark.asyncio
    async def test_streamed_archive_is_valid_and_cleans_up(self, app_main, monkeypatch):
        service = _enable_s3(app_main)
        monkeypatch.setattr(service, "download_s3_uri", lambda uri: b"IMAGE-BYTES")

        spool_dir = app_main.Path(app_main.tempfile.gettempdir())
        before = set(spool_dir.glob("visionsort_zip_*"))

        response = await app_main.download_results_zip(
            _json_request(app_main, _zip_payload(app_main, count=3))
        )
        spooled = set(spool_dir.glob("visionsort_zip_*")) - before
        assert len(spooled) == 1, "the archive must be spooled to exactly one temp file"

        body = b"".join([chunk async for chunk in response.body_iterator])

        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
        assert any(n.startswith("good/") for n in names)
        assert "manifest.json" in names
        assert not [path for path in spooled if path.exists()], (
            "spooled ZIP temp file must be removed once the response is consumed"
        )


# --- helpers -----------------------------------------------------------------

def _valid_payload() -> dict:
    return {
        "file_name": "a.jpg",
        "content_type": "image/jpeg",
        "raw_bytes": b"x",
        "temp_path": None,
        "validation_error": None,
    }


def _async(value):
    async def runner():
        return value
    return runner()


def _blocking_processor(seconds: float):
    def processor(payloads, progress_hook=None, session_id=""):
        time.sleep(seconds)  # stands in for cv2 + torch CPU work
        return {key: [] for key in ("good", "blurry", "dark", "overexposed", "duplicates")}
    return processor


def _enable_s3(app_main):
    service = app_main.aws_service
    service.uploads_bucket = "visionsort-uploads-vin"
    service.processed_bucket = "visionsort-processed-vin"
    service.s3_enabled = True
    service.s3_client = FakeS3Client()
    return service


def _zip_payload(app_main, count: int) -> dict:
    items = []
    for index in range(1, count + 1):
        uri = f"s3://visionsort-processed-vin/good/2026/09/06/req_abc/vin_img{index}.jpg"
        items.append(
            {
                "renamed_file_name": f"vin_img{index}.jpg",
                "processed_storage_path": uri,
                "processed_storage_token": app_main.sign_storage_uri(uri),
                "final_status": "good",
            }
        )
    return {"categories": ["good"], "source": "processed", "results": {"good": items}}


def _json_request(app_main, payload: dict):
    class _Request:
        url = type("U", (), {"path": "/download/zip"})()
        state = type("S", (), {"request_id": "test"})()

        async def json(self):
            return payload

    return _Request()
