"""P0 regression tests: D5 (cross-batch duplicates) and D6 (filename collisions).

Both stem from one root cause: dedupe state and the filename counter are scoped to a
single `process_upload_payloads` call, while a user's upload spans many calls because
the frontend batches into groups of 2.
"""

from __future__ import annotations

import pytest

from conftest import encode_jpeg, make_payload


def category_of(app_main, results):
    return next(key for key in app_main.CATEGORY_KEYS if results[key])


def only_item(app_main, results):
    return results[category_of(app_main, results)][0]


@pytest.fixture
def session_id(app_main):
    return app_main.new_upload_session_id()


class TestD5CrossBatchDuplicates:
    def test_same_image_in_a_later_batch_is_flagged(self, app_main, session_id):
        """The frontend sends 2 files per request, so duplicates must survive across calls."""
        image = encode_jpeg(seed=1)

        first = app_main.process_upload_payloads([make_payload(image, "a.jpg")], session_id=session_id)
        second = app_main.process_upload_payloads([make_payload(image, "b.jpg")], session_id=session_id)

        assert category_of(app_main, first) == "good"
        assert category_of(app_main, second) == "duplicates"

    def test_distinct_images_are_not_flagged(self, app_main, session_id):
        a = app_main.process_upload_payloads([make_payload(encode_jpeg(seed=2), "a.jpg")], session_id=session_id)
        b = app_main.process_upload_payloads([make_payload(encode_jpeg(seed=99, fill=40), "b.jpg")], session_id=session_id)

        assert category_of(app_main, a) == "good"
        assert category_of(app_main, b) != "duplicates"

    def test_sessions_are_isolated_from_each_other(self, app_main):
        """One user's images must never mark another user's upload as a duplicate."""
        image = encode_jpeg(seed=3)

        mine = app_main.process_upload_payloads([make_payload(image)], session_id=app_main.new_upload_session_id())
        theirs = app_main.process_upload_payloads([make_payload(image)], session_id=app_main.new_upload_session_id())

        assert category_of(app_main, mine) == "good"
        assert category_of(app_main, theirs) == "good"

    def test_within_batch_duplicates_still_work_without_a_session(self, app_main):
        """Behaviour must not regress when the client sends no session id."""
        image = encode_jpeg(seed=4)
        results = app_main.process_upload_payloads(
            [make_payload(image, "a.jpg"), make_payload(image, "b.jpg")]
        )

        assert len(results["duplicates"]) == 1

    def test_expired_sessions_are_evicted(self, app_main, monkeypatch):
        """The registry must not grow without bound."""
        session = app_main.new_upload_session_id()
        app_main.process_upload_payloads([make_payload(encode_jpeg(seed=5))], session_id=session)
        assert session in app_main.upload_sessions

        monkeypatch.setattr(app_main, "SESSION_RETENTION_SECONDS", 0)
        app_main.process_upload_payloads(
            [make_payload(encode_jpeg(seed=6))], session_id=app_main.new_upload_session_id()
        )

        assert session not in app_main.upload_sessions


class TestD6FilenameCollisions:
    def test_names_keep_counting_across_batches(self, app_main, session_id):
        first = app_main.process_upload_payloads([make_payload(encode_jpeg(seed=7), "a.jpg")], session_id=session_id)
        second = app_main.process_upload_payloads([make_payload(encode_jpeg(seed=8, fill=90), "b.jpg")], session_id=session_id)

        assert only_item(app_main, first)["renamed_file_name"] == "vin_img1.jpg"
        assert only_item(app_main, second)["renamed_file_name"] == "vin_img2.jpg"

    def test_no_duplicate_names_across_many_batches(self, app_main, session_id):
        names = []
        for index in range(6):
            results = app_main.process_upload_payloads(
                [make_payload(encode_jpeg(seed=100 + index, fill=60 + index * 12))], session_id=session_id
            )
            names.extend(
                item["renamed_file_name"]
                for key in app_main.CATEGORY_KEYS
                for item in results[key]
            )

        assert len(names) == len(set(names)), f"collisions across batches: {names}"

    def test_names_are_unique_without_a_session(self, app_main):
        """With no session id, names must still not collide between requests."""
        a = only_item(app_main, app_main.process_upload_payloads([make_payload(encode_jpeg(seed=9))]))
        b = only_item(app_main, app_main.process_upload_payloads([make_payload(encode_jpeg(seed=10, fill=70))]))

        assert a["renamed_file_name"] != b["renamed_file_name"]
