"""Regression tests for durable job state and cancellation publication."""

import json

import pytest

from web.api import store
from web.api.models import ClipDetail, JobStatus


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PERSIST_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(store, "_jobs", {})
    monkeypatch.setattr(store, "_persistence_error_type", None)
    monkeypatch.setattr(store, "_persistence_failed_at", None)
    monkeypatch.setattr(store, "_load_recovery_required", False)


def test_restart_marks_active_jobs_interrupted_and_preserves_terminal_jobs():
    active = store.create_job()
    complete = store.create_job()
    store.set_status(complete, JobStatus.COMPLETED)
    store._jobs.clear()

    store._load()

    assert store.get_job(active)["status"] == "failed"
    assert "restarted" in store.get_job(active)["error"]
    assert store.get_job(complete)["status"] == "completed"
    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        assert json.load(handle)[active]["status"] == "failed"


def test_restart_preserves_auto_story_user_checkpoints():
    job_id = store.create_job(config={"auto_story_mode": True})
    store.update_job(
        job_id,
        status=JobStatus.AWAITING_RECIPE.value,
        auto_story={"phase": JobStatus.AWAITING_RECIPE.value, "stories": [{"story_id": "story-1"}]},
    )
    store._jobs.clear()

    store._load()

    job = store.get_job(job_id)
    assert job["status"] == JobStatus.AWAITING_RECIPE.value
    assert job["auto_story"]["stories"] == [{"story_id": "story-1"}]


def test_restart_preserves_legacy_auto_story_checkpoint_without_phase():
    job_id = store.create_job(config={"auto_story_mode": True})
    store.update_job(job_id, status=JobStatus.AWAITING_STORYLINE.value)
    store._jobs.clear()

    store._load()

    assert store.get_job(job_id)["status"] == JobStatus.AWAITING_STORYLINE.value


def test_failed_atomic_replace_preserves_previous_snapshot(monkeypatch, caplog):
    job_id = store.create_job()
    original_replace = store.os.replace
    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        original = handle.read()

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    store.set_status(job_id, JobStatus.RENDERING)

    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        assert handle.read() == original
    assert "Could not persist" in caplog.text
    assert store.get_persistence_status()["healthy"] is False
    assert store.get_persistence_status()["error_type"] == "OSError"

    monkeypatch.setattr(store.os, "replace", original_replace)
    store.update_job(job_id, status=JobStatus.RENDERING.value)
    assert store.get_persistence_status()["healthy"] is True


def test_unreadable_snapshot_is_reported_and_not_overwritten(tmp_path):
    original = "{truncated snapshot"
    with open(store.PERSIST_PATH, "w", encoding="utf-8") as handle:
        handle.write(original)

    store._load()

    status = store.get_persistence_status()
    assert status["healthy"] is False
    assert status["recovery_required"] is True
    assert status["error_type"] == "JSONDecodeError"
    store.create_job()
    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        assert handle.read() == original
    assert store.get_persistence_status()["healthy"] is False


def test_cancellation_wins_over_success_and_record_cannot_disappear_while_running():
    job_id = store.create_job()
    store.set_status(job_id, JobStatus.RENDERING)
    assert store.request_cancel(job_id)
    assert store.get_job(job_id)["status"] == "rendering"
    assert not store.delete_job(job_id)
    clips = [ClipDetail(rank=1, filename="one.mp4", download_url="/one.mp4")]
    store.finish_job(job_id, clips=clips)
    job = store.get_job(job_id)
    assert job["status"] == "cancelled"
    assert job["clips"] == clips


def test_job_snapshots_cannot_mutate_store_and_active_reuse_is_rejected():
    job_id = store.create_job()
    copy = store.get_job(job_id)
    copy["log"].append("external change")
    assert store.get_job(job_id)["log"] == []
    with pytest.raises(ValueError, match="still active"):
        store.create_job(job_id=job_id)
