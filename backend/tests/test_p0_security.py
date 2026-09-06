"""P0 regression tests: D1 (S3 arbitrary-key read) and D3 (decompression bomb)."""

from __future__ import annotations

import pytest

from conftest import FakeS3Client, encode_png


class TestD1StoragePathForgery:
    """`/download/zip` must not fetch arbitrary keys just because the bucket matches."""

    def _service(self, app_main):
        service = app_main.aws_service
        service.uploads_bucket = "visionsort-uploads-vin"
        service.processed_bucket = "visionsort-processed-vin"
        service.s3_enabled = True
        service.s3_client = FakeS3Client()
        return service

    def test_unsigned_storage_path_is_rejected(self, app_main):
        """A path with no token must not be downloadable, even in an allowed bucket."""
        service = self._service(app_main)
        forged = "s3://visionsort-uploads-vin/some/other/users/private_photo.jpg"

        with pytest.raises(app_main.StoragePathNotAuthorized):
            app_main.authorize_storage_uri(forged, token="")

        assert service.s3_client.calls == [], "no S3 call may be attempted for an unsigned path"

    def test_token_from_one_object_cannot_authorize_another(self, app_main):
        """Tokens must bind to their exact object, so they cannot be replayed."""
        self._service(app_main)
        legitimate = "s3://visionsort-processed-vin/good/2026/09/06/req_abc/vin_img1.jpg"
        victim = "s3://visionsort-uploads-vin/good/2026/09/06/req_other/vin_img1.jpg"

        token = app_main.sign_storage_uri(legitimate)
        app_main.authorize_storage_uri(legitimate, token=token)  # must not raise

        with pytest.raises(app_main.StoragePathNotAuthorized):
            app_main.authorize_storage_uri(victim, token=token)

    def test_signed_path_round_trips(self, app_main):
        """The token the server issued for an object must authorize that object."""
        self._service(app_main)
        uri = "s3://visionsort-processed-vin/blurry/2026/09/06/req_abc/vin_img2.jpg"

        bucket, key = app_main.authorize_storage_uri(uri, token=app_main.sign_storage_uri(uri))

        assert (bucket, key) == ("visionsort-processed-vin", "blurry/2026/09/06/req_abc/vin_img2.jpg")

    def test_tampered_token_is_rejected(self, app_main):
        self._service(app_main)
        uri = "s3://visionsort-processed-vin/good/2026/09/06/req_abc/vin_img1.jpg"
        token = app_main.sign_storage_uri(uri)
        tampered = ("0" if token[0] != "0" else "1") + token[1:]

        with pytest.raises(app_main.StoragePathNotAuthorized):
            app_main.authorize_storage_uri(uri, token=tampered)

    def test_traversal_key_is_rejected_even_when_signed(self, app_main):
        """Defense in depth: keys must match the categorized layout the server writes."""
        self._service(app_main)
        uri = "s3://visionsort-uploads-vin/../../etc/secrets.json"

        with pytest.raises(app_main.StoragePathNotAuthorized):
            app_main.authorize_storage_uri(uri, token=app_main.sign_storage_uri(uri))

    def test_response_items_carry_a_storage_token(self, app_main, jpeg_bytes):
        """Clients can only send back a token if the server issued one."""
        from conftest import make_payload

        results = app_main.process_upload_payloads([make_payload(jpeg_bytes)])
        item = next(results[c][0] for c in app_main.CATEGORY_KEYS if results[c])

        assert "storage_token" in item


class TestD3DecompressionBomb:
    """decode_image must refuse images whose decoded size exceeds the pixel budget."""

    def test_oversized_image_is_rejected(self, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "MAX_IMAGE_PIXELS", 1_000_000)
        bomb = encode_png(4000, 4000)  # 16M pixels, ~50 KB on the wire

        with pytest.raises(ValueError, match="too large"):
            app_main.decode_image(bomb)

    def test_normal_image_still_decodes(self, app_main, jpeg_bytes):
        assert app_main.decode_image(jpeg_bytes).shape[:2] == (240, 240)

    def test_guard_runs_before_allocation(self, app_main, monkeypatch):
        """The budget check must happen before cv2 allocates the full array."""
        monkeypatch.setattr(app_main, "MAX_IMAGE_PIXELS", 1_000_000)
        calls = []
        real_imdecode = app_main.cv2.imdecode
        monkeypatch.setattr(
            app_main.cv2, "imdecode", lambda *a, **k: (calls.append(1), real_imdecode(*a, **k))[1]
        )

        with pytest.raises(ValueError):
            app_main.decode_image(encode_png(4000, 4000))

        assert calls == [], "cv2.imdecode must not be reached for an oversized image"


class TestD3FallbackPaths:
    """The paths the pre-decode probe cannot cover must still be bounded."""

    def test_pil_guard_tracks_our_budget(self, app_main):
        """Setting Image.MAX_IMAGE_PIXELS to None would disable it entirely."""
        from PIL import Image

        assert Image.MAX_IMAGE_PIXELS == app_main.MAX_IMAGE_PIXELS

    def test_unparseable_header_is_still_bounded_after_decode(self, app_main, monkeypatch):
        """When PIL cannot probe, the post-decode backstop must still reject."""
        monkeypatch.setattr(app_main, "MAX_IMAGE_PIXELS", 1_000_000)
        monkeypatch.setattr(
            app_main.Image, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no plugin"))
        )

        with pytest.raises(ValueError, match="too large"):
            app_main.decode_image(encode_png(4000, 4000))

    def test_probe_failure_does_not_reject_valid_images(self, app_main, monkeypatch, jpeg_bytes):
        monkeypatch.setattr(
            app_main.Image, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no plugin"))
        )

        assert app_main.decode_image(jpeg_bytes).shape[:2] == (240, 240)
