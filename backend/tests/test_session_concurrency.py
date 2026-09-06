"""Regressions found by code review of the D5/D6 session fix.

Every test here failed before the session registry was made refcounted, serialized
and idempotent.
"""

from __future__ import annotations

import threading

import pytest

from conftest import encode_jpeg, make_payload


def categories_of(app_main, results):
    return [key for key in app_main.CATEGORY_KEYS if results[key]]


class TestRetryIdempotency:
    """A client-side timeout retry re-sends a batch the server may already have done."""

    def test_resent_batch_is_not_flagged_duplicate(self, app_main):
        session = app_main.new_upload_session_id()
        image = encode_jpeg(seed=21)

        first = app_main.process_upload_payloads(
            [make_payload(image, "a.jpg")], session_id=session, batch_key="batch-1"
        )
        retry = app_main.process_upload_payloads(
            [make_payload(image, "a.jpg")], session_id=session, batch_key="batch-1"
        )

        assert categories_of(app_main, first) == ["good"]
        assert categories_of(app_main, retry) == ["good"], (
            "a retry of the same batch must not see its own first attempt as a duplicate"
        )

    def test_retry_does_not_advance_the_filename_counter(self, app_main):
        session = app_main.new_upload_session_id()
        image = encode_jpeg(seed=22)

        app_main.process_upload_payloads([make_payload(image)], session_id=session, batch_key="b1")
        retry = app_main.process_upload_payloads([make_payload(image)], session_id=session, batch_key="b1")
        nxt = app_main.process_upload_payloads(
            [make_payload(encode_jpeg(seed=23, fill=80))], session_id=session, batch_key="b2"
        )

        assert retry["good"][0]["renamed_file_name"] == "vin_img1.jpg"
        assert nxt["good"][0]["renamed_file_name"] == "vin_img2.jpg"

    def test_a_genuinely_new_batch_still_dedupes(self, app_main):
        """Idempotency must not disable cross-batch duplicate detection."""
        session = app_main.new_upload_session_id()
        image = encode_jpeg(seed=24)

        app_main.process_upload_payloads([make_payload(image)], session_id=session, batch_key="b1")
        second = app_main.process_upload_payloads(
            [make_payload(image)], session_id=session, batch_key="b2"
        )

        assert categories_of(app_main, second) == ["duplicates"]


class TestConcurrentBatches:
    def test_overlapping_batches_of_one_session_do_not_corrupt_state(self, app_main):
        """Batches ran in threadpool and job_executor threads with no lock."""
        session = app_main.new_upload_session_id()
        names: list[str] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        # Enough files per batch, and enough threads, to actually open the
        # read-modify-write window on the shared filename counter.
        batches = [
            [
                make_payload(encode_jpeg(seed=200 + index * 10 + n, fill=40 + n * 9), f"f{n}.jpg")
                for n in range(6)
            ]
            for index in range(8)
        ]
        barrier = threading.Barrier(8)

        def worker(index: int):
            try:
                barrier.wait()  # release all threads at once
                results = app_main.process_upload_payloads(
                    batches[index],
                    session_id=session,
                    batch_key=f"batch-{index}",
                )
                with lock:
                    names.extend(
                        item["renamed_file_name"]
                        for key in app_main.CATEGORY_KEYS
                        for item in results[key]
                    )
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"concurrent batches raised: {errors}"
        assert len(names) == len(set(names)), f"filename collision under concurrency: {names}"


class TestEvictionSafety:
    def test_in_flight_session_is_never_evicted(self, app_main, monkeypatch):
        """Evicting a live session resets its counter and reintroduces D6."""
        monkeypatch.setattr(app_main, "MAX_TRACKED_SESSIONS", 16)
        monkeypatch.setattr(app_main, "SESSION_RETENTION_SECONDS", 60)
        victim = app_main.new_upload_session_id()

        held = app_main.acquire_upload_session(victim)  # simulates a batch in flight
        try:
            for _ in range(80):  # a flood of junk ids
                app_main.acquire_upload_session(app_main.new_upload_session_id())
                app_main.release_upload_session(app_main.upload_sessions[victim])
                app_main.acquire_upload_session(victim)
            assert victim in app_main.upload_sessions
        finally:
            app_main.release_upload_session(held)

    def test_idle_sessions_are_still_reclaimed(self, app_main, monkeypatch):
        monkeypatch.setattr(app_main, "SESSION_RETENTION_SECONDS", 0)
        stale = app_main.new_upload_session_id()
        app_main.release_upload_session(app_main.acquire_upload_session(stale))

        app_main.release_upload_session(
            app_main.acquire_upload_session(app_main.new_upload_session_id())
        )

        assert stale not in app_main.upload_sessions


class TestHashBounding:
    def test_hash_list_is_capped(self, app_main, monkeypatch):
        """A client may post unlimited batches under one session id."""
        monkeypatch.setattr(app_main, "MAX_SESSION_HASHES", 4)
        session = app_main.new_upload_session_id()

        for index in range(12):
            app_main.process_upload_payloads(
                [make_payload(encode_jpeg(seed=300 + index, fill=40 + index * 15))],
                session_id=session,
                batch_key=f"b{index}",
            )

        assert len(app_main.upload_sessions[session]["hashes"]) <= 4 + 1
