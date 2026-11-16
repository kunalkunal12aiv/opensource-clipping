"""API tests for the dashboard-guided Auto Story workflow."""

import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from clipping import auto_story as pipeline
from web.api import auto_story_worker, config_adapter, store, worker
from web.api.app import app
from web.api.models import JobStatus
from web.api.routes import auto_story as auto_story_routes, files


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PERSIST_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(store, "_jobs", {})
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(files, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(worker, "_settings_env", {
        "GOOGLE_API_KEY": "test-key",
        "HF_TOKEN": "test-hf-token",
    })
    monkeypatch.setattr(worker, "_tasks", {})
    monkeypatch.setattr(worker, "_running_jobs", set())
    monkeypatch.setattr(auto_story_worker, "_tasks", {})
    monkeypatch.setattr(auto_story_routes, "get_readiness", lambda: {"ready": True})
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "uploaded_episode.mp4").write_bytes(b"episode")
    with TestClient(app) as client:
        yield client, tmp_path


def test_auto_story_requires_keys_and_uploaded_video(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(worker, "_settings_env", {})

    missing_video = client.post(
        "/api/auto-story",
        json={"upload_filename": "missing.mp4"},
    )
    assert missing_video.status_code == 400
    assert "upload" in missing_video.json()["detail"].lower()

    monkeypatch.setattr(worker, "_settings_env", {"GOOGLE_API_KEY": "test-key"})
    missing_token = client.post(
        "/api/auto-story",
        json={"upload_filename": "uploaded_episode.mp4"},
    )
    assert missing_token.status_code == 400
    assert "HF_TOKEN" in missing_token.json()["detail"]
    assert store.list_jobs() == []


def test_auto_story_create_uploads_job_and_queues_discovery(api, monkeypatch):
    client, _ = api
    submitted = []
    monkeypatch.setattr(auto_story_routes, "cuda_available", lambda: False)

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(auto_story_worker, "submit_discovery", fake_submit)
    response = client.post(
        "/api/auto-story",
        json={
            "upload_filename": "uploaded_episode.mp4",
            "source_name": "Episode one.mp4",
            "target_duration_sec": 90,
            "ai_provider": "gemini",
        },
    )

    assert response.status_code == 201, response.text
    job = response.json()
    assert job["source"] == "local"
    assert job["upload_filename"] == "uploaded_episode.mp4"
    assert job["auto_story"]["phase"] == "queued"
    assert job["auto_story"]["target_duration_sec"] == 90
    saved_config = store.get_job(job["id"])["config"]
    assert saved_config["auto_story_whisper_device"] == "cpu"
    assert saved_config["auto_story_whisper_compute_type"] == "int8"
    assert submitted == [job["id"]]


def test_auto_story_uses_cuda_when_available(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(auto_story_routes, "cuda_available", lambda: True)

    async def fake_submit(_job_id):
        return None

    monkeypatch.setattr(auto_story_worker, "submit_discovery", fake_submit)
    response = client.post(
        "/api/auto-story",
        json={"upload_filename": "uploaded_episode.mp4"},
    )

    assert response.status_code == 201, response.text
    saved_config = store.get_job(response.json()["id"])["config"]
    assert saved_config["auto_story_whisper_device"] == "cuda"
    assert saved_config["auto_story_whisper_compute_type"] == "float16"


def test_auto_story_worker_preserves_selected_whisper_settings(monkeypatch):
    captured = {}
    expected_config = object()

    def build_config(payload, job_id, *, env_overrides):
        captured.update(payload=payload, job_id=job_id, env_overrides=env_overrides)
        return expected_config

    monkeypatch.setattr(config_adapter, "build_config_from_payload", build_config)
    monkeypatch.setattr(worker, "get_settings_env", lambda: {})
    job = {
        "id": "test-job",
        "upload_filename": "uploaded_episode.mp4",
        "config": {
            "auto_story_mode": True,
            "auto_story_whisper_model": "small",
            "auto_story_whisper_device": "cuda",
            "auto_story_whisper_compute_type": "float16",
        },
    }

    assert auto_story_worker._build_config(job) is expected_config
    assert captured["job_id"] == "test-job"
    assert captured["payload"]["auto_story_whisper_device"] == "cuda"
    assert captured["payload"]["auto_story_whisper_compute_type"] == "float16"
    assert captured["payload"]["whisper_device"] == "cuda"
    assert captured["payload"]["whisper_compute_type"] == "float16"


def test_auto_story_review_render_and_cancel_transitions(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_recipe",
            "storylines": [],
            "recipe": {"beats": []},
            "validation": None,
        },
        status=JobStatus.AWAITING_RECIPE.value,
    )

    async def fake_validate(_job_id, recipe):
        return {
            "recipe": recipe,
            "validation": {
                "beat_count": 8,
                "estimated_runtime_sec": 60.2,
                "episode_span_sec": 2400,
            },
        }

    submitted = []

    async def fake_render(_job_id):
        submitted.append(_job_id)

    monkeypatch.setattr(auto_story_worker, "validate_recipe", fake_validate)
    monkeypatch.setattr(auto_story_worker, "submit_render", fake_render)

    validated = client.put(
        f"/api/auto-story/{job_id}/recipe",
        json={"recipe": {"beats": [{"dialogue_start_sec": 1.0}]}},
    )
    assert validated.status_code == 200, validated.text
    assert validated.json()["auto_story"]["validation"]["beat_count"] == 8

    rendered = client.post(f"/api/auto-story/{job_id}/render")
    assert rendered.status_code == 200, rendered.text
    assert rendered.json()["status"] == "rendering"
    assert submitted == [job_id]

    cancel_job = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        cancel_job,
        auto_story={"phase": "awaiting_storyline"},
        status=JobStatus.AWAITING_STORYLINE.value,
    )
    cancelled = client.delete(f"/api/auto-story/{cancel_job}")
    assert cancelled.status_code == 200
    assert store.get_job(cancel_job)["status"] == JobStatus.CANCELLED.value


def test_invalid_recipe_is_saved_and_previous_validation_is_cleared(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_recipe",
            "storylines": [],
            "recipe": {"beats": [{"dialogue_start_sec": 1.0}]},
            "validation": {"beat_count": 1},
        },
        status=JobStatus.AWAITING_RECIPE.value,
    )
    submitted_recipe = {"beats": [{"dialogue_start_sec": "2.5"}]}

    async def reject_recipe(_job_id, _recipe):
        raise ValueError("Beat is too short.")

    monkeypatch.setattr(auto_story_worker, "validate_recipe", reject_recipe)
    response = client.put(
        f"/api/auto-story/{job_id}/recipe",
        json={"recipe": submitted_recipe},
    )

    assert response.status_code == 422
    saved = store.get_job(job_id)["auto_story"]
    assert saved["recipe"] == submitted_recipe
    assert saved["validation"] is None
    assert saved["message"] == "Recipe needs changes: Beat is too short."


def test_recipe_validation_saves_exact_cached_boundaries_and_dialogue(api, monkeypatch):
    _, tmp_path = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    storyline = {"id": "arc-1"}
    store.update_job(
        job_id,
        auto_story={
            "source_id": "episode-test",
            "source_duration_sec": 500.0,
            "selected_storyline_id": "arc-1",
            "storylines": [storyline],
            "recipe_filename": "review.json",
        },
    )
    recipe = {
        "source_id": "episode-test",
        "storyline": storyline,
        "beats": [{
            "beat_id": 1,
            "start_word_id": "W000001",
            "end_word_id": "W000002",
            "dialogue_start_sec": 10.01,
            "dialogue_end_sec": 12.02,
            "text": "old text",
            "speaker_ids": ["SPEAKER_00"],
        }],
    }
    accepted_beat = {
        "start_word_id": "W000001",
        "end_word_id": "W000003",
        "dialogue_start_sec": 10.0,
        "dialogue_end_sec": 13.0,
        "text": "updated cached words",
        "speaker_ids": ["SPEAKER_01"],
    }
    cfg = SimpleNamespace(
        auto_story_target_duration=60,
        auto_story_min_beat_sec=3.5,
        auto_story_max_beat_sec=6.0,
        auto_story_min_gap_sec=30,
        auto_story_min_episode_span_ratio=0.4,
        auto_story_padding_sec=0.18,
        auto_story_crossfade_ms=60,
    )
    monkeypatch.setattr(
        auto_story_worker, "_source_context",
        lambda _job: {"cfg": cfg, "output_dir": str(tmp_path)},
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    monkeypatch.setattr(
        pipeline,
        "validate_review_recipe",
        lambda *_args: {
            "beats": [accepted_beat],
            "estimated_runtime_sec": 60.0,
            "episode_span_sec": 300.0,
        },
    )

    result = auto_story_worker._validate_recipe_sync(job_id, recipe)

    assert result["recipe"]["beats"][0]["dialogue_end_sec"] == 13.0
    assert result["recipe"]["beats"][0]["text"] == "updated cached words"
    assert result["recipe"]["beats"][0]["speaker_ids"] == ["SPEAKER_01"]
    saved = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert saved == result["recipe"]


def test_discovery_stage_publishes_storylines_without_blocking_api(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(job_id, auto_story={"phase": "queued"})
    context = {
        "cfg": object(),
        "source_id": "episode-test",
        "source": {"name": "Episode one"},
        "video_path": "uploaded_episode.mp4",
        "auto_cache_dir": "cache",
        "output_dir": "output",
    }
    storylines = [{
        "id": "arc-1",
        "title": "A story",
        "summary": "A beginning. A payoff.",
        "key_characters": [],
        "speaker_ids": [],
        "start_sec": 1.0,
        "end_sec": 300.0,
    }]
    monkeypatch.setattr(auto_story_worker, "_source_context", lambda _job: context)
    monkeypatch.setattr(pipeline, "_provider_model", lambda _cfg: ("gemini", "model"))
    monkeypatch.setattr(pipeline, "_probe_duration", lambda _path: 3600.0)

    progress_updates = []
    original_update_progress = store.update_progress

    def record_progress(_job_id, *args, **kwargs):
        progress_updates.append(kwargs.get("percent", args[4] if len(args) > 4 else None))
        return original_update_progress(_job_id, *args, **kwargs)

    def fake_transcription(
        *_args,
        progress_callback=None,
        phase_callback=None,
    ):
        progress_callback(50, 100)
        phase_callback("diarization")
        return [{"word_id": "W1"}], "key"

    monkeypatch.setattr(store, "update_progress", record_progress)
    monkeypatch.setattr(
        pipeline, "_load_or_create_transcript", fake_transcription
    )
    monkeypatch.setattr(
        pipeline, "_discover_storylines", lambda *_args: storylines
    )

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(worker, "run_serialized_call", run_inline)
    asyncio.run(auto_story_worker._run_stage(
        job_id, "discovery", auto_story_worker._discover_sync
    ))

    job = store.get_job(job_id)
    assert job["status"] == JobStatus.AWAITING_STORYLINE.value
    assert job["auto_story"]["storylines"] == storylines
    assert job["auto_story"]["transcript_key"] == "key"
    assert 25.0 in progress_updates
    assert 45 in progress_updates
