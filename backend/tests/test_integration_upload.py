"""End-to-end checks through the real ASGI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import encode_jpeg, encode_png


@pytest.fixture
def client(app_main):
    with TestClient(app_main.app) as test_client:
        yield test_client


def upload(client, images, session_id=""):
    files = [("files", (name, blob, "image/jpeg")) for name, blob in images]
    data = {"session_id": session_id} if session_id else {}
    return client.post("/upload", files=files, data=data)


class TestUploadFlow:
    def test_single_upload_succeeds(self, client):
        response = upload(client, [("a.jpg", encode_jpeg(seed=11))])

        assert response.status_code == 200
        body = response.json()
        assert sum(len(body[key]) for key in ("good", "blurry", "dark", "overexposed", "duplicates")) == 1

    def test_batches_sharing_a_session_dedupe_and_keep_counting(self, client, app_main):
        """Reproduces the real frontend pattern: one upload split into 2-file requests."""
        session = app_main.new_upload_session_id()
        image = encode_jpeg(seed=12)

        first = upload(client, [("a.jpg", image)], session).json()
        second = upload(client, [("b.jpg", image)], session).json()

        assert len(second["duplicates"]) == 1, "the repeat must be caught across requests"
        names = [
            item["renamed_file_name"]
            for payload in (first, second)
            for key in ("good", "blurry", "dark", "overexposed", "duplicates")
            for item in payload[key]
        ]
        assert names == ["vin_img1.jpg", "vin_img2.jpg"]

    def test_decompression_bomb_is_rejected_not_crashed(self, client, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "MAX_IMAGE_PIXELS", 500_000)
        response = client.post(
            "/upload", files=[("files", ("bomb.png", encode_png(4000, 4000), "image/png"))]
        )

        # The bomb is skipped, so the request reports no processable image rather than dying.
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BAD_REQUEST"

    def test_a_bomb_does_not_poison_the_rest_of_the_batch(self, client, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "MAX_IMAGE_PIXELS", 500_000)
        response = client.post(
            "/upload",
            files=[
                ("files", ("bomb.png", encode_png(4000, 4000), "image/png")),
                ("files", ("ok.jpg", encode_jpeg(seed=13), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        body = response.json()
        assert sum(len(body[key]) for key in ("good", "blurry", "dark", "overexposed", "duplicates")) == 1

    def test_malformed_session_id_is_ignored_not_fatal(self, client):
        response = upload(client, [("a.jpg", encode_jpeg(seed=14))], "../../etc/passwd")

        assert response.status_code == 200


class TestZipEndpoint:
    def test_zip_requires_s3(self, client):
        response = client.post("/download/zip", json={"results": {"good": []}})

        assert response.status_code == 503

    def test_forged_storage_path_yields_no_objects(self, client, app_main, monkeypatch):
        """The whole point of D1: an unsigned path must never be fetched."""
        from conftest import FakeS3Client

        service = app_main.aws_service
        monkeypatch.setattr(service, "uploads_bucket", "visionsort-uploads-vin")
        monkeypatch.setattr(service, "processed_bucket", "visionsort-processed-vin")
        monkeypatch.setattr(service, "s3_enabled", True)
        fake = FakeS3Client()
        monkeypatch.setattr(service, "s3_client", fake)

        response = client.post(
            "/download/zip",
            json={
                "categories": ["good"],
                "source": "processed",
                "results": {
                    "good": [
                        {
                            "renamed_file_name": "stolen.jpg",
                            "processed_storage_path": "s3://visionsort-uploads-vin/good/2026/09/06/req_x/other.jpg",
                        }
                    ]
                },
            },
        )

        assert response.status_code == 400
        assert fake.calls == [], "no S3 request may be issued for an unsigned path"
