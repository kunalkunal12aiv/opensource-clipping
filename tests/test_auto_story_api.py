"""API tests for the dashboard-guided Auto Story workflow."""

import asyncio
import hashlib
import json
import zipfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from clipping import auto_story as pipeline
from web.api import auto_story_projects, auto_story_worker, config_adapter, store, worker
from web.api.app import app
from web.api.models import JobStatus
from web.api.routes import auto_story as auto_story_routes, files, settings as settings_routes


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
        json={
            "upload_filename": "uploaded_episode.mp4",
            "diarization_enabled": True,
        },
    )
    assert missing_token.status_code == 400
    assert "HF_TOKEN" in missing_token.json()["detail"]
    assert store.list_jobs() == []


def test_auto_story_does_not_require_huggingface_for_dialogue_only_mode(api, monkeypatch):
    client, _ = api
    submitted = []
    monkeypatch.setattr(worker, "_settings_env", {"GOOGLE_API_KEY": "test-key"})

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(auto_story_worker, "submit_discovery", fake_submit)
    response = client.post(
        "/api/auto-story",
        json={"upload_filename": "uploaded_episode.mp4"},
    )

    assert response.status_code == 201
    assert submitted == [response.json()["id"]]
    assert store.get_job(response.json()["id"])["config"]["diarization_enabled"] is False


def test_auto_story_accepts_freellmapi_unified_key(api, monkeypatch):
    client, _ = api
    submitted = []
    monkeypatch.setattr(worker, "_settings_env", {"FREELLMAPI_API_KEY": "unified-key"})
    monkeypatch.setattr(auto_story_routes, "cuda_available", lambda: False)

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(auto_story_worker, "submit_discovery", fake_submit)
    response = client.post(
        "/api/auto-story",
        json={
            "upload_filename": "uploaded_episode.mp4",
            "ai_provider": "freellmapi",
        },
    )

    assert response.status_code == 201, response.text
    assert store.get_job(response.json()["id"])["config"]["ai_provider"] == "freellmapi"
    assert submitted == [response.json()["id"]]


def test_settings_store_freellmapi_key_and_gateway_url(api, monkeypatch):
    client, tmp_path = api
    monkeypatch.setattr(settings_routes, "cuda_available", lambda: False)
    monkeypatch.setattr(settings_routes, "_check_ffmpeg", lambda: True)

    response = client.put(
        "/api/settings",
        json={
            "freellmapi_api_key": "a-secret-unified-key",
            "freellmapi_base_url": "https://gateway.example/v1",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["freellmapi_api_key_set"] is True
    assert response.json()["freellmapi_base_url"] == "https://gateway.example/v1"
    assert "a-secret-unified-key" not in response.text
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "FREELLMAPI_API_KEY=" in env_text
    assert "a-secret-unified-key" in env_text


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
            "min_gap_sec": 20,
            "diarization_speakers": 2,
            "whisper_model": "medium",
            "ai_provider": "gemini",
        },
    )

    assert response.status_code == 201, response.text
    job = response.json()
    assert job["source"] == "local"
    assert job["upload_filename"] == "uploaded_episode.mp4"
    assert job["auto_story"]["phase"] == "queued"
    assert job["auto_story"]["target_duration_sec"] == 90
    assert job["auto_story"]["story_count"] == 0
    saved_config = store.get_job(job["id"])["config"]
    assert saved_config["auto_story_whisper_device"] == "cpu"
    assert saved_config["auto_story_whisper_compute_type"] == "int8"
    assert saved_config["diarization_speakers"] == 2
    assert saved_config["min_gap_sec"] == 20
    assert saved_config["auto_story_whisper_model"] == "medium"
    assert job["persistence"]["healthy"] is True
    assert submitted == [job["id"]]


def test_auto_story_rejects_out_of_range_diarization_speaker_count(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(auto_story_routes, "cuda_available", lambda: False)

    response = client.post(
        "/api/auto-story",
        json={
            "upload_filename": "uploaded_episode.mp4",
            "diarization_speakers": 17,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("min_gap_sec", [14, 61])
def test_auto_story_rejects_out_of_range_min_gap(api, min_gap_sec):
    client, _ = api
    response = client.post(
        "/api/auto-story",
        json={"upload_filename": "uploaded_episode.mp4", "min_gap_sec": min_gap_sec},
    )
    assert response.status_code == 422


def test_auto_story_rejects_unknown_whisper_model(api):
    client, _ = api
    response = client.post(
        "/api/auto-story",
        json={
            "upload_filename": "uploaded_episode.mp4",
            "whisper_model": "made-up-model",
        },
    )
    assert response.status_code == 422


def test_health_reports_degraded_job_store_persistence(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(store, "_persistence_error_type", "OSError")
    monkeypatch.setattr(store, "_persistence_failed_at", "2026-11-17T00:00:00+00:00")
    monkeypatch.setattr(store, "_load_recovery_required", True)
    monkeypatch.setattr(settings_routes, "cuda_available", lambda: False)
    monkeypatch.setattr(settings_routes, "_check_ffmpeg", lambda: True)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["job_store"] == {
        "healthy": False,
        "error_type": "OSError",
        "last_failure_at": "2026-11-17T00:00:00+00:00",
        "recovery_required": True,
    }


def test_auto_story_source_supports_range_streaming_for_audio_preview(api):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )

    response = client.get(
        f"/api/auto-story/{job_id}/source",
        headers={"Range": "bytes=1-3"},
    )

    assert response.status_code == 206
    assert response.content == b"pis"
    assert response.headers["content-range"] == "bytes 1-3/7"
    assert response.headers["content-type"].startswith("video/mp4")


def test_auto_story_project_library_returns_compact_saved_project_summaries(api):
    client, _ = api
    story_job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True, "source_name": "Episode source"},
    )
    store.update_job(
        story_job_id,
        status=JobStatus.AWAITING_STORYLINE.value,
        auto_story={"source_name": "Episode project", "story_count": 3},
        log=["large worker diagnostic line"],
    )
    store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": False},
    )

    response = client.get("/api/auto-story/projects")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["storage"]["healthy"] is True
    project = response.json()["projects"][0]
    assert project["id"] == story_job_id
    assert project["source_name"] == "Episode project"
    assert project["story_count"] == 3
    assert "log" not in project
    assert "config" not in project


def test_auto_story_project_library_searches_and_pages_server_side(api):
    client, _ = api
    excluded_id = None
    for index in range(5):
        job_id = store.create_job(
            upload_filename=f"episode-{index}.mp4",
            source="local",
            config={"auto_story_mode": True, "source_name": f"Podcast {index}"},
        )
        store.update_job(
            job_id,
            auto_story={"source_name": f"Podcast {index}", "stories": [], "story_count": index},
            log=["large payload omitted from project summary"] * 100,
        )
        if index == 2:
            excluded_id = job_id
    store.create_job(config={"auto_story_mode": False})

    response = client.get("/api/auto-story/projects?q=PODCAST&offset=1&limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 5
    assert payload["offset"] == 1
    assert payload["limit"] == 2
    assert len(payload["projects"]) == 2
    assert all(project["source_name"].startswith("Podcast") for project in payload["projects"])
    assert all("log" not in project and "config" not in project for project in payload["projects"])

    filtered = client.get("/api/auto-story/projects?q=Podcast%202&limit=10")
    assert filtered.json()["total"] == 1
    assert filtered.json()["projects"][0]["id"] == excluded_id


def test_auto_story_diagnostics_export_is_safe_and_includes_cache_timings(api, tmp_path):
    client, project_dir = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={
            "auto_story_mode": True,
            "ai_provider": "gemini",
            "target_duration_sec": 90,
            "auto_story_whisper_model": "small",
            "auto_story_whisper_device": "cpu",
            "auto_story_whisper_compute_type": "int8",
            "api_key": "must-never-be-exported",
        },
    )
    source_id = "episode-" + "a" * 24
    store.update_job(
        job_id,
        status=JobStatus.FAILED.value,
        auto_story={
            "phase": "failed",
            "last_failed_stage": "speaker_diarization",
            "diagnostic_error_type": "RuntimeError",
            "transcript_key": "private-cache-key",
            "raw_provider_error": "private transcript and provider response",
            "source_id": source_id,
            "analysis_cache": {
                "transcript": "created",
                "diarization": "missing",
                "combined": "missing",
                "unexpected": "private",
            },
            "stage_runs": [{
                "stage": "discovery",
                "elapsed_ms": 1234.5,
                "outcome": "failed",
                "private": "must not be exported",
            }],
            "storylines": [{"id": "arc-1"}],
            "stories": [],
            "failed_stories": [{"reason": "private error text"}],
        },
    )
    output_dir = project_dir / "outputs" / job_id
    output_dir.mkdir(parents=True)
    (output_dir / "story_plan_manifest.json").write_text("{}", encoding="utf-8")
    cache_dir = project_dir / "outputs" / "story_cache" / "auto_story" / source_id
    cache_dir.mkdir(parents=True)
    (cache_dir / "transcript_words.json").write_text("secret text", encoding="utf-8")

    response = client.get(f"/api/auto-story/{job_id}/diagnostics")

    assert response.status_code == 200
    assert "attachment;" in response.headers["content-disposition"]
    payload = response.json()
    assert payload["last_failed_stage"] == "speaker_diarization"
    assert payload["error_type"] == "RuntimeError"
    assert payload["job_store"]["healthy"] is True
    assert payload["analysis_cache"]["usage"] == {
        "transcript": "created",
        "diarization": "missing",
        "combined": "missing",
    }
    assert payload["analysis_cache"]["files_available"] == {
        "transcript": True,
        "diarization": False,
        "combined": False,
    }
    assert payload["stage_runs"] == [{
        "stage": "discovery",
        "elapsed_ms": 1234.5,
        "outcome": "failed",
    }]
    encoded = response.text
    for private_value in (
        "must-never-be-exported",
        "private-cache-key",
        "private transcript",
        "must not be exported",
        "secret text",
        str(project_dir),
    ):
        assert private_value not in encoded


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


def test_storyline_selection_and_rediscovery_are_available_after_discovery(api, monkeypatch):
    client, _ = api
    storylines = [
        {"id": f"arc-{index}", "title": f"Story {index}"}
        for index in range(1, 4)
    ]
    job_ids = []
    for revision in (2, 0, 0):
        job_id = store.create_job(
            upload_filename="uploaded_episode.mp4",
            source="local",
            config={"auto_story_mode": True},
        )
        store.update_job(
            job_id,
            auto_story={
                "phase": "awaiting_storyline",
                "storylines": storylines,
                "story_count": len(storylines),
                "discovery_revision": revision,
            },
            status=JobStatus.AWAITING_STORYLINE.value,
        )
        job_ids.append(job_id)

    selected = []
    rediscovered = []

    async def fake_select(job_id, storyline_ids, review_before_render=False):
        selected.append((job_id, storyline_ids, review_before_render))

    async def fake_rediscover(job_id):
        rediscovered.append(job_id)

    monkeypatch.setattr(
        auto_story_worker, "submit_storyline_selection", fake_select
    )
    monkeypatch.setattr(
        auto_story_worker, "submit_rediscovery", fake_rediscover
    )

    selection_response = client.post(
        f"/api/auto-story/{job_ids[0]}/storylines/select",
        json={"storyline_ids": ["arc-1", "arc-3"]},
    )
    assert selection_response.status_code == 200, selection_response.text
    assert selection_response.json()["status"] == JobStatus.ANALYZING.value
    assert (
        selection_response.json()["auto_story"]["selected_storyline_ids"]
        == ["arc-1", "arc-3"]
    )
    assert selected == [(job_ids[0], ["arc-1", "arc-3"], False)]

    review_response = client.post(
        f"/api/auto-story/{job_ids[2]}/storylines/select",
        json={"storyline_ids": ["arc-2"], "review_before_render": True},
    )
    assert review_response.status_code == 200, review_response.text
    assert selected[-1] == (job_ids[2], ["arc-2"], True)
    assert review_response.json()["auto_story"]["review_before_render"] is True

    rediscovery_response = client.post(
        f"/api/auto-story/{job_ids[1]}/storylines/rediscover"
    )
    assert rediscovery_response.status_code == 200, rediscovery_response.text
    assert rediscovery_response.json()["status"] == JobStatus.ANALYZING.value
    assert (
        rediscovery_response.json()["auto_story"]["discovery_revision"] == 1
    )
    assert rediscovery_response.json()["auto_story"]["storylines"] == []
    assert rediscovered == [job_ids[1]]


def test_speaker_names_are_validated_and_trigger_fresh_discovery(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_storyline",
            "speaker_ids": ["SPEAKER_00"],
            "speaker_names": {},
            "storylines": [{"id": "arc-1", "title": "Story"}],
            "discovery_revision": 0,
        },
        status=JobStatus.AWAITING_STORYLINE.value,
    )
    submitted = []

    async def fake_rediscover(submitted_job_id):
        submitted.append(submitted_job_id)

    monkeypatch.setattr(auto_story_worker, "submit_rediscovery", fake_rediscover)
    monkeypatch.setattr(
        auto_story_worker,
        "_load_uncorrected_words",
        lambda _state: [{"word_id": "W000001", "speaker_id": "SPEAKER_00"}],
    )
    rejected = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_names": {"SPEAKER_99": "Unknown"}},
    )
    assert rejected.status_code == 422

    response = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_names": {"SPEAKER_00": "Alex"}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["auto_story"]["speaker_names"] == {"SPEAKER_00": "Alex"}
    assert response.json()["auto_story"]["storylines"] == []
    assert response.json()["auto_story"]["discovery_revision"] == 1
    assert submitted == [job_id]


def test_speaker_turn_corrections_are_paged_saved_and_used_for_discovery(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    words = [
        {"word_id": "W000001", "word": "I", "start": 0.0, "end": 0.2, "speaker_id": "SPEAKER_00"},
        {"word_id": "W000002", "word": "agree", "start": 0.3, "end": 0.6, "speaker_id": "SPEAKER_00"},
        {"word_id": "W000003", "word": "No", "start": 3.0, "end": 3.2, "speaker_id": "SPEAKER_01"},
    ]
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_storyline",
            "source_id": "episode-test",
            "transcript_key": "transcript-key",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_names": {},
            "storylines": [{"id": "arc-1", "title": "Story"}],
            "discovery_revision": 0,
        },
        status=JobStatus.AWAITING_STORYLINE.value,
    )
    monkeypatch.setattr(auto_story_worker, "_load_uncorrected_words", lambda _state: words)
    monkeypatch.setattr(
        auto_story_worker,
        "_load_words",
        lambda state: auto_story_worker._apply_speaker_corrections(words, state),
    )
    submitted = []

    async def fake_rediscover(submitted_job_id):
        submitted.append(submitted_job_id)

    monkeypatch.setattr(auto_story_worker, "submit_rediscovery", fake_rediscover)
    page = client.get(f"/api/auto-story/{job_id}/speakers/transcript?offset=1&limit=1")
    assert page.status_code == 200, page.text
    assert page.json()["total"] == 2
    assert page.json()["turns"][0]["text"] == "No"
    assert page.json()["has_more"] is False

    invalid_speaker = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_corrections": [{
            "start_word_id": "W000001",
            "end_word_id": "W000002",
            "speaker_id": "SPEAKER_99",
        }]},
    )
    assert invalid_speaker.status_code == 422

    response = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={
            "speaker_corrections": [{
                "start_word_id": "W000001",
                "end_word_id": "W000002",
                "speaker_id": "SPEAKER_01",
            }, {
                "start_word_id": "W000003",
                "end_word_id": "W000003",
                "speaker_id": None,
            }],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["auto_story"]["speaker_corrections"] == [{
        "start_word_id": "W000001",
        "end_word_id": "W000002",
        "speaker_id": "SPEAKER_01",
    }, {
        "start_word_id": "W000003",
        "end_word_id": "W000003",
        "speaker_id": None,
    }]
    assert [word["speaker_id"] for word in auto_story_worker._load_words(
        response.json()["auto_story"]
    )] == ["SPEAKER_01", "SPEAKER_01", None]
    assert submitted == [job_id]


def test_speaker_labels_can_be_merged_with_cycle_and_unknown_validation(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    words = [
        {"word_id": "W000001", "word": "hello", "start": 0.0, "end": 0.3, "speaker_id": "SPEAKER_00"},
        {"word_id": "W000002", "word": "again", "start": 1.0, "end": 1.3, "speaker_id": "SPEAKER_01"},
    ]
    store.update_job(
        job_id,
        status=JobStatus.AWAITING_STORYLINE.value,
        auto_story={
            "phase": "awaiting_storyline",
            "source_id": "episode-test",
            "transcript_key": "transcript-key",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_names": {},
            "speaker_merges": {},
            "speaker_corrections": [],
            "storylines": [{"id": "arc-1", "title": "Story"}],
            "discovery_revision": 0,
        },
    )
    monkeypatch.setattr(auto_story_worker, "_load_uncorrected_words", lambda _state: words)
    submitted = []

    async def fake_rediscover(submitted_job_id):
        submitted.append(submitted_job_id)

    monkeypatch.setattr(auto_story_worker, "submit_rediscovery", fake_rediscover)
    cyclic = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_merges": {
            "SPEAKER_00": "SPEAKER_01",
            "SPEAKER_01": "SPEAKER_00",
        }},
    )
    assert cyclic.status_code == 422

    unknown = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_merges": {"SPEAKER_00": "SPEAKER_02"}},
    )
    assert unknown.status_code == 422

    merged = client.put(
        f"/api/auto-story/{job_id}/speakers",
        json={"speaker_merges": {"SPEAKER_01": "SPEAKER_00"}},
    )
    assert merged.status_code == 200, merged.text
    saved = merged.json()["auto_story"]
    assert saved["speaker_merges"] == {"SPEAKER_01": "SPEAKER_00"}
    assert [word["speaker_id"] for word in auto_story_worker._apply_speaker_corrections(
        words, saved
    )] == ["SPEAKER_00", "SPEAKER_00"]
    assert submitted == [job_id]


def test_speaker_merge_maps_names_to_the_canonical_label():
    state = {
        "speaker_ids": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "speaker_names": {"SPEAKER_01": "Alex", "SPEAKER_02": "Host"},
        "speaker_merges": {"SPEAKER_01": "SPEAKER_00"},
    }
    assert auto_story_worker._speaker_names_for_state(state) == {
        "SPEAKER_00": "Alex",
        "SPEAKER_02": "Host",
    }


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
            "min_gap_sec": 45,
        },
    }

    assert auto_story_worker._build_config(job) is expected_config
    assert captured["job_id"] == "test-job"
    assert captured["payload"]["auto_story_whisper_device"] == "cuda"
    assert captured["payload"]["auto_story_whisper_compute_type"] == "float16"
    assert captured["payload"]["whisper_device"] == "cuda"
    assert captured["payload"]["whisper_compute_type"] == "float16"
    assert captured["payload"]["diarization_speakers"] == "auto"
    assert captured["payload"]["min_gap_sec"] == 45


def test_auto_story_restarts_analysis_automatically_from_saved_checkpoint(api, monkeypatch):
    client, _ = api
    submitted = []

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(auto_story_worker, "submit_discovery", fake_submit)
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.TRANSCRIBING.value,
        auto_story={"phase": "speaker_diarization", "analysis_cache": {"transcript": "created"}},
    )
    store._jobs.clear()
    store._load()

    resumed = asyncio.run(auto_story_worker.recover_interrupted_auto_story_jobs())

    assert resumed == 1
    assert submitted == [job_id]
    recovered = store.get_job(job_id)
    assert recovered["status"] == JobStatus.ANALYZING.value
    assert recovered["restart_recovery"] is None
    assert recovered["auto_story"]["phase"] == "speaker_diarization"
    assert "resuming from saved transcript" in recovered["auto_story"]["message"].lower()


def test_auto_story_restarts_rendering_stage_from_saved_plan(api, monkeypatch):
    _, _project_dir = api
    submitted = []

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(auto_story_worker, "submit_render", fake_submit)
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.RENDERING.value,
        auto_story={"phase": "rendering", "stories": [{"story_id": "story-1"}]},
    )
    store._jobs.clear()
    store._load()

    resumed = asyncio.run(auto_story_worker.recover_interrupted_auto_story_jobs())

    assert resumed == 1
    assert submitted == [job_id]
    recovered = store.get_job(job_id)
    assert recovered["status"] == JobStatus.RENDERING.value
    assert recovered["restart_recovery"] is None
    assert "resuming from saved story plans" in recovered["auto_story"]["message"].lower()


def test_failed_auto_story_can_retry_render_without_rediscovery(api, monkeypatch):
    client, _ = api
    submitted = []

    async def fake_submit(job_id):
        submitted.append(job_id)

    monkeypatch.setattr(
        auto_story_worker,
        "_source_context",
        lambda _job: {"source_id": "episode-saved"},
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [{"word": "hello"}])
    monkeypatch.setattr(auto_story_worker, "submit_render", fake_submit)
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.FAILED.value,
        auto_story={
            "phase": "rendering",
            "source_id": "episode-saved",
            "stories": [{"story_id": "story-1", "validation": {"estimated_runtime_sec": 60}}],
            "validation": {"valid_story_count": 1},
            "rendered_story_count": 0,
        },
    )

    response = client.post(f"/api/auto-story/{job_id}/retry-render")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == JobStatus.RENDERING.value
    assert submitted == [job_id]


def test_auto_story_render_restart_keeps_completed_story_outputs(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs" / "render-job"
    output_dir.mkdir(parents=True)
    first_files = []
    for name in ("story-1.mp4", "story-1.vtt", "story-1.json", "story-1_recipe.json"):
        (output_dir / name).write_bytes(b"complete")
        first_files.append({"filename": name, "download_url": f"/api/outputs/render-job/{name}"})

    def story(story_id, outputs=None):
        return {
            "story_id": story_id,
            "storyline": {"id": story_id, "title": story_id},
            "recipe_filename": f"{story_id}_recipe.json",
            "manifest_file": f"{story_id}.json",
            "manifest": {"video_file": f"{story_id}.mp4"},
            "output_files": outputs or [],
            "validation": {"estimated_runtime_sec": 60.0},
            "actual_runtime_sec": 59.0,
        }

    stories = [story("story-1", first_files), story("story-2")]
    state = {
        "phase": "rendering",
        "source_id": "episode-id",
        "source_name": "Episode",
        "stories": stories,
        "failed_stories": [],
        "rendered_story_count": 1,
        "validation": {"valid_story_count": 2},
        "selected_story_count": 2,
    }
    context = {
        "source_id": "episode-id",
        "cfg": SimpleNamespace(auto_story_target_duration=60),
        "output_dir": str(output_dir),
    }
    rendered = []

    def fake_render(_job_id, item, *_args):
        rendered.append(item["story_id"])
        files = [{"filename": "story-2.mp4", "download_url": "/story-2.mp4"}]
        return {**item, "manifest": {"video_file": "story-2.mp4"}, "output_files": files}, files

    def fake_update(_job_id, **updates):
        state.update(updates)
        return state

    monkeypatch.setattr(auto_story_worker, "_source_context", lambda _job: context)
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    monkeypatch.setattr(auto_story_worker, "_render_story_sync", fake_render)
    monkeypatch.setattr(auto_story_worker, "_update_state", fake_update)
    monkeypatch.setattr(auto_story_worker, "_check_cancelled", lambda _job_id: None)

    result = auto_story_worker._render_stories_sync("render-job", {}, state)

    assert rendered == ["story-2"]
    assert [item["story_id"] for item in result["stories"]] == ["story-1", "story-2"]


def test_auto_story_cache_identity_uses_episode_content_not_upload_name(
    api, monkeypatch, tmp_path
):
    _, project_dir = api
    upload_dir = project_dir / "uploads"
    first_path = upload_dir / "first-upload.mp4"
    second_path = upload_dir / "second-upload.mp4"
    first_path.write_bytes(b"identical episode content")
    second_path.write_bytes(b"identical episode content")
    monkeypatch.setattr(
        auto_story_worker,
        "_build_config",
        lambda _job: SimpleNamespace(auto_story_output_dir=str(tmp_path / "out")),
    )
    signature_calls = []
    real_signature = pipeline._file_signature

    def track_signature(path):
        signature_calls.append(path)
        return real_signature(path)

    monkeypatch.setattr(pipeline, "_file_signature", track_signature)

    first = auto_story_worker._source_context({
        "upload_filename": first_path.name,
        "config": {"source_name": "Episode"},
    })
    second = auto_story_worker._source_context({
        "upload_filename": second_path.name,
        "config": {"source_name": "Episode"},
    })

    assert first["source_id"] == second["source_id"]
    assert first["auto_cache_dir"] == second["auto_cache_dir"]
    assert first["source_signature"] == second["source_signature"]

    reused = auto_story_worker._source_context({
        "upload_filename": first_path.name,
        "config": {"source_name": "Episode"},
        "auto_story": {
            "source_id": first["source_id"],
            "source_signature": first["source_signature"],
            "source_file_stat": first["source_file_stat"],
        },
    })
    assert reused["source_id"] == first["source_id"]
    assert len(signature_calls) == 2

    first_path.write_bytes(b"changed episode bytes with a different length")
    changed = auto_story_worker._source_context({
        "upload_filename": first_path.name,
        "config": {"source_name": "Episode"},
        "auto_story": {
            "source_id": first["source_id"],
            "source_signature": first["source_signature"],
            "source_file_stat": first["source_file_stat"],
        },
    })
    assert changed["source_id"] != first["source_id"]
    assert len(signature_calls) == 3


def test_auto_story_can_export_recovery_backup_during_analysis_but_not_rendering(
    api, monkeypatch
):
    client, project_dir = api
    outputs_dir = project_dir / "outputs"
    monkeypatch.setattr(auto_story_projects, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(auto_story_projects, "UPLOAD_DIR", project_dir / "uploads")
    source = project_dir / "uploads" / "uploaded_episode.mp4"
    job_id = store.create_job(
        upload_filename=source.name,
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.TRANSCRIBING.value,
        auto_story={"phase": "speaker_diarization", "source_id": "episode-pending"},
    )

    backup = client.get(f"/api/auto-story/{job_id}/project")
    assert backup.status_code == 200
    assert backup.headers["content-type"] == "application/zip"

    store.update_job(job_id, status=JobStatus.RENDERING.value)
    blocked = client.get(f"/api/auto-story/{job_id}/project")
    assert blocked.status_code == 409


def test_auto_story_project_export_import_restores_source_cache_and_outputs(
    api, monkeypatch
):
    client, project_dir = api
    outputs_dir = project_dir / "outputs"
    upload_dir = project_dir / "uploads"
    monkeypatch.setattr(auto_story_projects, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(auto_story_projects, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(files, "OUTPUTS_DIR", str(outputs_dir))
    source_path = upload_dir / "uploaded_episode.mp4"
    source_id = f"episode-{pipeline._file_signature(str(source_path))['content_sha256'][:24]}"
    job_id = store.create_job(
        upload_filename=source_path.name,
        source="local",
        config={
            "auto_story_mode": True,
            "target_duration_sec": 60,
            "min_gap_sec": 45,
            "diarization_speakers": 2,
            "auto_story_whisper_model": "large-v3-turbo",
            "ai_provider": "gemini",
            "source_name": "Episode project",
        },
    )
    state = {
        "phase": "awaiting_recipe",
        "source_id": source_id,
        "source_name": "Episode project",
        "transcript_key": "cache-key",
        "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
        "speaker_names": {"SPEAKER_00": "Alex"},
        "speaker_merges": {"SPEAKER_01": "SPEAKER_00"},
        "speaker_corrections": [{
            "start_word_id": "W000001",
            "end_word_id": "W000001",
            "speaker_id": "SPEAKER_00",
        }],
        "storylines": [{"id": "arc-1", "title": "A story"}],
        "stories": [{
            "story_id": "story-1",
            "storyline": {"id": "arc-1", "title": "A story"},
            "recipe": {"beats": []},
            "recipe_filename": "story-1_recipe_test.json",
            "validation": {"estimated_runtime_sec": 60.0},
        }],
        "output_files": [],
        "validation": {"valid_story_count": 1},
    }
    store.update_job(
        job_id,
        status=JobStatus.AWAITING_RECIPE.value,
        auto_story=state,
    )
    output_dir = outputs_dir / job_id
    output_dir.mkdir(parents=True)
    (output_dir / "story_plan_manifest.json").write_text('{"stories": []}', encoding="utf-8")
    (output_dir / "story-1_recipe_test.json").write_text('{"beats": []}', encoding="utf-8")
    cache_dir = outputs_dir / "story_cache" / "auto_story" / source_id
    cache_dir.mkdir(parents=True)
    (cache_dir / "combined_transcript.json").write_text(
        '{"_cache_key": "cache-key", "words": [{"word_id": "W000001", "word": "hello", "start": 0.0, "end": 0.2, "speaker_id": "SPEAKER_00"}, {"word_id": "W000002", "word": "again", "start": 0.3, "end": 0.5, "speaker_id": "SPEAKER_01"}]}',
        encoding="utf-8",
    )

    exported = client.get(f"/api/auto-story/{job_id}/project")
    assert exported.status_code == 200
    assert exported.headers["content-type"] == "application/zip"
    (cache_dir / "combined_transcript.json").unlink()

    restored = client.post(
        "/api/auto-story/projects/import",
        files={"file": ("episode-project.zip", exported.content, "application/zip")},
    )
    assert restored.status_code == 201, restored.text
    restored_job = restored.json()
    restored_id = restored_job["id"]
    assert restored_id != job_id
    assert restored_job["status"] == JobStatus.AWAITING_RECIPE.value
    assert restored_job["auto_story"]["source_id"] == source_id
    assert restored_job["auto_story"]["speaker_names"] == {"SPEAKER_00": "Alex"}
    assert restored_job["config"]["diarization_speakers"] == 2
    assert restored_job["config"]["min_gap_sec"] == 45
    assert restored_job["config"]["auto_story_whisper_model"] == "large-v3-turbo"
    assert restored_job["auto_story"]["speaker_merges"] == {"SPEAKER_01": "SPEAKER_00"}
    assert restored_job["auto_story"]["speaker_corrections"][0]["speaker_id"] == "SPEAKER_00"
    restored_output_names = {
        item["filename"] for item in restored_job["auto_story"]["output_files"]
    }
    assert restored_output_names == {
        "story_plan_manifest.json", "story-1_recipe_test.json"
    }
    restored_story = restored_job["auto_story"]["stories"][0]
    assert restored_story["output_files"][0]["download_url"].startswith(
        f"/api/outputs/{restored_id}/"
    )
    assert (cache_dir / "combined_transcript.json").is_file()
    assert store.get_job(restored_id)["upload_filename"] != source_path.name
    assert client.get(
        f"/api/outputs/{restored_id}/story_plan_manifest.json"
    ).status_code == 200


def test_auto_story_project_import_defaults_legacy_null_whisper_model(api, monkeypatch):
    _, project_dir = api
    outputs_dir = project_dir / "outputs"
    upload_dir = project_dir / "uploads"
    monkeypatch.setattr(auto_story_projects, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(auto_story_projects, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(files, "OUTPUTS_DIR", str(outputs_dir))
    source_bytes = b"synthetic source for legacy backup validation"
    source_id = f"episode-{hashlib.sha256(source_bytes).hexdigest()[:24]}"
    manifest = {
        "format": "opensource-clipping-auto-story",
        "schema_version": 1,
        "source_id": source_id,
        "config": {
            "target_duration_sec": 60,
            "min_gap_sec": 30,
            "diarization_speakers": None,
            "ai_provider": "gemini",
            "source_name": "Legacy project",
            "whisper_model": None,
        },
        "status": "analyzing",
        "auto_story": {
            "phase": "creating_stories",
            "storylines": [{"id": "arc-1"}, {"id": "arc-2"}],
            "selected_storyline_ids": ["arc-1", "arc-2"],
            "stories": [],
        },
    }
    archive_path = project_dir / "legacy-project.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("project.json", json.dumps(manifest))
        archive.writestr("source/episode.mp4", source_bytes)

    restored_id = auto_story_projects._restore_archive(archive_path)

    assert store.get_job(restored_id)["config"]["auto_story_whisper_model"] == "small"
    assert store.get_job(restored_id)["config"]["workflow_mode"] == "choose"


def test_failed_discovery_can_retry_without_recreating_the_job(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.FAILED.value,
        auto_story={"phase": "failed", "discovery_revision": 2, "source_name": "Episode"},
    )
    submitted = []

    async def fake_rediscover(submitted_job_id):
        submitted.append(submitted_job_id)

    monkeypatch.setattr(auto_story_worker, "submit_rediscovery", fake_rediscover)
    response = client.post(f"/api/auto-story/{job_id}/retry-discovery")

    assert response.status_code == 200, response.text
    assert submitted == [job_id]
    assert response.json()["status"] == JobStatus.ANALYZING.value
    assert response.json()["auto_story"]["discovery_revision"] == 3


def test_story_discovery_retry_can_switch_provider_and_keep_analysis(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True, "ai_provider": "nvidia"},
    )
    store.update_job(
        job_id,
        status=JobStatus.AWAITING_STORYLINE.value,
        auto_story={
            "phase": "awaiting_storyline",
            "analysis_ready": True,
            "transcript_key": "saved-transcript",
            "speaker_ids": ["SPEAKER_00"],
            "last_failed_stage": "storyline_discovery",
        },
    )
    submitted = []
    monkeypatch.setattr(auto_story_routes, "_required_key", lambda _key: True)

    async def fake_rediscover(submitted_job_id):
        submitted.append(submitted_job_id)

    monkeypatch.setattr(auto_story_worker, "submit_rediscovery", fake_rediscover)
    response = client.post(
        f"/api/auto-story/{job_id}/storylines/rediscover",
        json={"ai_provider": "gemini"},
    )

    assert response.status_code == 200, response.text
    saved = store.get_job(job_id)
    assert saved["config"]["ai_provider"] == "gemini"
    assert saved["auto_story"]["analysis_ready"] is True
    assert saved["auto_story"]["transcript_key"] == "saved-transcript"
    assert saved["auto_story"]["speaker_ids"] == ["SPEAKER_00"]
    assert "saved transcript and speaker analysis will be reused" in saved["auto_story"]["message"]
    assert submitted == [job_id]


def test_completed_project_can_reopen_manual_review_when_cache_is_present(api):
    client, project_dir = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    source_id = "episode-" + "a" * 24
    state = {
        "source_id": source_id,
        "transcript_key": "transcript-key",
        "stories": [{"story_id": "story-1", "recipe": {"beats": []}, "validation": {"ok": True}}],
        "validation": {"valid_story_count": 1},
    }
    store.update_job(
        job_id,
        status=JobStatus.COMPLETED.value,
        auto_story=state,
    )
    cache_dir = project_dir / "outputs" / "story_cache" / "auto_story" / source_id
    cache_dir.mkdir(parents=True)
    (cache_dir / "combined_transcript.json").write_text(
        '{"_cache_key": "transcript-key", "words": []}', encoding="utf-8"
    )

    response = client.post(f"/api/auto-story/{job_id}/review-again")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == JobStatus.AWAITING_RECIPE.value
    assert response.json()["auto_story"]["stories"][0]["story_id"] == "story-1"


def test_auto_story_project_import_rejects_unsafe_archive_paths(api, monkeypatch):
    client, project_dir = api
    monkeypatch.setattr(auto_story_projects, "OUTPUTS_DIR", project_dir / "outputs")
    monkeypatch.setattr(auto_story_projects, "UPLOAD_DIR", project_dir / "uploads")
    import io
    import zipfile

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("../escape.txt", "no")
    response = client.post(
        "/api/auto-story/projects/import",
        files={"file": ("unsafe.zip", archive_buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 400
    assert "unsafe path" in response.json()["detail"]


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


def test_storyline_selection_prevalidates_the_scheduled_recipe(api, monkeypatch):
    _, tmp_path = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    words = [
        {
            "word_id": "W000001",
            "word": "Here",
            "start": 10.0,
            "end": 10.2,
            "speaker_id": "SPEAKER_00",
        },
        {
            "word_id": "W000002",
            "word": "we are.",
            "start": 13.8,
            "end": 14.2,
            "speaker_id": "SPEAKER_00",
        },
    ]
    storyline = {
        "id": "arc-1",
        "source_ranges": [{
            "start_word_id": "W000001",
            "end_word_id": "W000002",
            "start_sec": 10.0,
            "end_sec": 14.2,
        }],
    }
    store.update_job(
        job_id,
        auto_story={
            "source_id": "episode-test",
            "source_duration_sec": 600.0,
            "transcript_key": "transcript-key",
            "storylines": [storyline],
        },
    )
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
        auto_story_worker,
        "_source_context",
        lambda _job: {
            "source_id": "episode-test",
            "cfg": cfg,
            "auto_cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "outputs"),
        },
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: words)
    monkeypatch.setattr(
        pipeline,
        "_select_beats",
        lambda *_args, **_kwargs: ({
            "beats": [{
                "start_word_id": "W000001",
                "end_word_id": "W000002",
                "narrative_role": "hook",
                "reason": "Opens the story.",
            }],
        }, "cache-key"),
    )
    accepted_beat = {
        "start_word_id": "W000001",
        "end_word_id": "W000002",
        "dialogue_start_sec": 10.0,
        "dialogue_end_sec": 14.2,
        "text": "Here we are.",
        "speaker_ids": ["SPEAKER_00"],
    }
    monkeypatch.setattr(
        pipeline,
        "validate_review_recipe",
        lambda *_args: {
            "beats": [accepted_beat],
            "estimated_runtime_sec": 4.56,
            "episode_span_sec": 4.56,
        },
    )

    result = auto_story_worker._select_beats_sync(job_id, "arc-1")

    assert result["validation"] == {
        "beat_count": 1,
        "estimated_runtime_sec": 4.56,
        "episode_span_sec": 4.56,
    }
    assert result["recipe"]["beats"][0]["dialogue_end_sec"] == 14.2
    assert (tmp_path / "outputs" / result["recipe_filename"]).is_file()


def test_discovery_stage_publishes_storylines_without_blocking_api(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True, "workflow_mode": "choose"},
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "queued",
            "selected_storyline_ids": ["stale-arc"],
            "selected_story_count": 1,
        },
    )
    context = {
        "cfg": SimpleNamespace(
            auto_story_target_duration=60,
            auto_story_min_gap_sec=30.0,
            auto_story_diarization_enabled=True,
        ),
        "source_id": "episode-test",
        "source_signature": {"content_sha256": "test"},
        "source_file_stat": {"size": 123, "mtime_ns": 456},
        "source": {"name": "Episode one"},
        "video_path": "uploaded_episode.mp4",
        "auto_cache_dir": "cache",
        "output_dir": "output",
    }
    storylines = [
        {
            "id": f"arc-{index}",
            "title": f"Story {index}",
            "summary": f"A beginning {index}. A payoff {index}.",
            "key_characters": [],
            "speaker_ids": [],
            "start_sec": 1.0,
            "end_sec": 300.0,
        }
        for index in range(1, 5)
    ]
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
        cache_callback=None,
        source_signature=None,
    ):
        assert source_signature is not None
        progress_callback(50, 100)
        phase_callback("diarization")
        cache_callback({
            "transcript": "created",
            "diarization": "created",
            "combined": "created",
        })
        return [{"word_id": "W1"}], "key"

    monkeypatch.setattr(store, "update_progress", record_progress)
    monkeypatch.setattr(
        pipeline, "_load_or_create_transcript", fake_transcription
    )
    discovery_options = {}

    def fake_discover(*_args, **kwargs):
        discovery_options.update(kwargs)
        return storylines

    monkeypatch.setattr(pipeline, "_discover_storylines", fake_discover)

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(worker, "run_serialized_call", run_inline)
    asyncio.run(auto_story_worker._run_stage(
        job_id, "discovery", auto_story_worker._discover_sync
    ))

    job = store.get_job(job_id)
    assert job["status"] == JobStatus.AWAITING_STORYLINE.value
    assert job["auto_story"]["storylines"] == storylines
    assert job["auto_story"]["stories"] == []
    assert job["auto_story"]["story_count"] == 4
    assert job["auto_story"]["selected_storyline_ids"] == []
    assert job["auto_story"]["selected_story_count"] == 0
    assert "Choose which ones to create" in job["auto_story"]["message"]
    assert discovery_options["count"] is None
    assert job["auto_story"]["transcript_key"] == "key"
    assert job["auto_story"]["analysis_cache"]["combined"] == "created"
    assert job["auto_story"]["stage_runs"][-1]["stage"] == "discovery"
    assert job["auto_story"]["stage_runs"][-1]["outcome"] == "completed"
    assert 25.0 in progress_updates
    assert 45 in progress_updates


def test_default_auto_workflow_queues_one_story_after_discovery(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True, "workflow_mode": "auto"},
    )
    scheduled = []
    monkeypatch.setattr(
        auto_story_worker,
        "_start",
        lambda *_args: scheduled.append(_args),
    )
    result = {"storylines": [{"id": "best-arc", "title": "Best Arc"}], "story_count": 1}

    async def run_stage():
        task = asyncio.create_task(
            auto_story_worker._run_stage(
                job_id, "discovery", lambda _job_id: result
            )
        )
        await task
        await asyncio.sleep(0)

    asyncio.run(run_stage())

    state = store.get_job(job_id)["auto_story"]
    assert state["phase"] == "story_planning"
    assert scheduled and scheduled[0][1] == "generation"
    assert scheduled[0][3] == ["best-arc"]
    assert scheduled[0][4] is False


def test_storyline_ai_failure_identifies_saved_post_diarization_checkpoint(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "storyline_discovery",
            "analysis_ready": True,
            "transcript_key": "saved-transcript",
        },
    )

    def fail_discovery(_job_id):
        raise RuntimeError("provider temporarily unavailable")

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(worker, "run_serialized_call", run_inline)
    asyncio.run(auto_story_worker._run_stage(job_id, "discovery", fail_discovery))

    job = store.get_job(job_id)
    assert job["status"] == JobStatus.AWAITING_STORYLINE.value
    assert job["auto_story"]["analysis_ready"] is True
    assert job["auto_story"]["last_failed_stage"] == "storyline_discovery"
    assert "transcript and speaker analysis were saved" in job["auto_story"]["message"]


def test_story_plan_retries_distinct_candidate_rounds(api, monkeypatch, tmp_path):
    _, _ = api
    cfg = SimpleNamespace(
        auto_story_target_duration=60,
        auto_story_min_beat_sec=3.5,
        auto_story_max_beat_sec=6.0,
        auto_story_min_gap_sec=30.0,
        auto_story_min_episode_span_ratio=0.4,
        auto_story_padding_sec=0.18,
        auto_story_crossfade_ms=60,
    )
    state = {
        "source_id": "episode-test",
        "source_duration_sec": 3600.0,
        "transcript_key": "transcript-key",
    }
    context = {
        "cfg": cfg,
        "auto_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "outputs"),
    }
    monkeypatch.setattr(
        auto_story_worker, "_load_words", lambda _state: [{"word_id": "W1"}]
    )
    candidate_rounds = []
    pacing_profiles = []

    def select_beats(*_args, candidate_round, **_kwargs):
        candidate_rounds.append(candidate_round)
        pacing_profiles.append((_args[5], _args[6]))
        if candidate_round < 2:
            raise ValueError("No valid schedule from these candidates.")
        return {"beats": []}, "cache-key"

    accepted_beat = {
        "start_word_id": "W1",
        "end_word_id": "W2",
        "dialogue_start_sec": 1.0,
        "dialogue_end_sec": 5.0,
        "text": "A complete sentence.",
        "speaker_ids": ["SPEAKER_00"],
    }
    monkeypatch.setattr(pipeline, "_select_beats", select_beats)
    monkeypatch.setattr(
        pipeline, "_build_review_recipe",
        lambda *_args: {"beats": [{"beat_id": 1}]},
    )
    monkeypatch.setattr(
        pipeline, "validate_review_recipe",
        lambda *_args: {
            "beats": [accepted_beat],
            "estimated_runtime_sec": 60.0,
            "episode_span_sec": 300.0,
        },
    )
    monkeypatch.setattr(pipeline, "_write_json", lambda *_args: None)
    monkeypatch.setattr(
        pipeline, "_build_review_timing_options", lambda *_args: {}
    )

    story = auto_story_worker._create_story_plan(
        state,
        context,
        {"id": "arc-1", "title": "A supported story"},
        "story-1",
    )

    assert candidate_rounds == [0, 1, 2]
    assert pacing_profiles == [(30.0, 0.4), (30.0, 0.4), (15.0, 0.2)]
    assert story["validation"]["estimated_runtime_sec"] == 60.0
    assert story["settings"]["min_gap_sec"] == 15.0


def test_story_plan_uses_saved_candidates_for_best_effort_export(api, monkeypatch, tmp_path):
    _, _ = api
    cfg = SimpleNamespace(
        auto_story_target_duration=60,
        auto_story_min_beat_sec=3.5,
        auto_story_max_beat_sec=6.0,
        auto_story_min_gap_sec=30.0,
        auto_story_min_episode_span_ratio=0.4,
        auto_story_padding_sec=0.18,
        auto_story_crossfade_ms=60,
    )
    state = {
        "source_id": "episode-test",
        "source_duration_sec": 3600.0,
        "transcript_key": "transcript-key",
    }
    context = {
        "cfg": cfg,
        "auto_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "outputs"),
    }
    monkeypatch.setattr(
        auto_story_worker, "_load_words", lambda _state: [{"word_id": "W1"}]
    )
    calls = []
    beat = {
        "beat_id": 1,
        "start_word_id": "W1",
        "end_word_id": "W2",
        "dialogue_start_sec": 1.0,
        "dialogue_end_sec": 5.0,
        "text": "A complete sentence.",
        "speaker_ids": ["SPEAKER_00"],
    }

    def select_beats(*_args, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("cache_only"):
            raise ValueError("strict plan not schedulable")
        return {"beats": [{"start_word_id": "W1", "end_word_id": "W2"}]}, "cache-key"

    monkeypatch.setattr(pipeline, "_select_beats", select_beats)
    monkeypatch.setattr(
        pipeline, "_build_review_recipe",
        lambda _response, _arc, _words, _source, settings: {
            "beats": [dict(beat)], "settings": settings,
        },
    )
    monkeypatch.setattr(
        pipeline, "validate_review_recipe",
        lambda *_args: {
            "beats": [beat],
            "estimated_runtime_sec": 4.36,
            "episode_span_sec": 4.36,
        },
    )
    monkeypatch.setattr(pipeline, "_write_json", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_build_review_timing_options", lambda *_args: {})

    story = auto_story_worker._create_story_plan(
        state, context, {"id": "arc-1", "title": "A supported story"}, "story-1"
    )

    assert len(calls) == 4
    assert calls[-1]["candidate_round"] == 2
    assert calls[-1]["best_effort"] is True
    assert calls[-1]["cache_only"] is True
    assert story["settings"]["best_effort"] is True
    assert story["validation"]["estimated_runtime_sec"] == 4.36


def test_selected_storylines_are_planned_then_rendered_without_review(api, monkeypatch):
    _, _ = api
    storylines = [
        {"id": f"arc-{index}", "title": f"Story {index}"}
        for index in range(1, 4)
    ]
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "source_id": "episode-test",
            "source_duration_sec": 3600.0,
            "transcript_key": "transcript-key",
            "storylines": storylines,
        },
    )
    monkeypatch.setattr(
        auto_story_worker, "_source_context",
        lambda _job: {"cfg": SimpleNamespace()},
    )
    created = []

    def create_plan(_state, _context, arc, story_id):
        created.append((arc["id"], story_id))
        if arc["id"] == "arc-1":
            raise ValueError("No valid schedule after automatic pacing fallbacks.")
        return {
            "story_id": story_id,
            "storyline": arc,
            "recipe": {"beats": []},
            "validation": {"estimated_runtime_sec": 60.0},
        }

    rendered = []

    def render_stories(_job_id, _job, state):
        rendered.extend(state["stories"])
        return {"stories": state["stories"], "output_files": ["story.mp4"]}

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(auto_story_worker, "_create_story_plan", create_plan)
    monkeypatch.setattr(
        auto_story_worker, "_render_stories_sync", render_stories
    )
    monkeypatch.setattr(worker, "run_serialized_call", run_inline)

    asyncio.run(auto_story_worker._run_stage(
        job_id,
        "generation",
        auto_story_worker._create_selected_stories_sync,
        ["arc-1", "arc-3"],
    ))

    saved_job = store.get_job(job_id)
    assert created == [("arc-1", "story-1"), ("arc-3", "story-2")]
    assert [story["story_id"] for story in rendered] == ["story-2"]
    assert saved_job["status"] == JobStatus.COMPLETED.value
    assert saved_job["auto_story"]["phase"] == "completed"
    assert saved_job["auto_story"]["output_files"] == ["story.mp4"]
    assert saved_job["auto_story"]["failed_stories"] == [{
        "storyline_id": "arc-1",
        "story_id": "story-1",
        "title": "Story 1",
        "reason": "No valid schedule after automatic pacing fallbacks.",
    }]
    assert "1 unbuildable storyline(s) were skipped" in (
        saved_job["auto_story"]["message"]
    )


def test_failed_story_retry_preserves_other_story_outputs(api, monkeypatch, tmp_path):
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    completed_story = {
        "story_id": "story-2",
        "storyline": {"id": "arc-2", "title": "Successful story"},
        "validation": {"estimated_runtime_sec": 60.0},
        "actual_runtime_sec": 60.0,
        "manifest_file": "story-2_manifest.json",
        "manifest": {"video_file": "story-2_video.mp4"},
        "recipe_filename": "story-2_recipe.json",
        "output_files": [{"filename": "story-2_video.mp4"}],
    }
    state = {
        "source_id": "episode-test",
        "source_name": "Episode",
        "source_duration_sec": 360.0,
        "selected_story_count": 2,
        "storylines": [
            {"id": "arc-1", "title": "Retry story"},
            completed_story["storyline"],
        ],
        "stories": [completed_story],
        "failed_stories": [{
            "storyline_id": "arc-1",
            "story_id": "story-1",
            "title": "Retry story",
            "reason": "Initial render failed.",
        }],
        "output_files": [{"filename": "story-2_video.mp4"}],
    }
    store.update_job(job_id, auto_story=state)
    cfg = SimpleNamespace(auto_story_target_duration=60)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "story-1_partial.tmp").write_text("old partial", encoding="utf-8")
    monkeypatch.setattr(
        auto_story_worker,
        "_source_context",
        lambda _job: {"source_id": "episode-test", "cfg": cfg, "output_dir": str(output_dir)},
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    monkeypatch.setattr(
        auto_story_worker,
        "_create_story_plan",
        lambda _state, _context, arc, story_id, planning_revision=0: {
            "story_id": story_id,
            "storyline": arc,
            "validation": {"estimated_runtime_sec": 60.0},
            "recipe_filename": f"{story_id}_recipe.json",
        },
    )

    def render(_job_id, planned, *_args):
        return {
            **planned,
            "actual_runtime_sec": 60.0,
            "manifest_file": "story-1_manifest.json",
            "manifest": {"video_file": "story-1_video.mp4"},
            "output_files": [{"filename": "story-1_video.mp4"}],
        }, []

    manifests = []
    monkeypatch.setattr(auto_story_worker, "_render_story_sync", render)
    monkeypatch.setattr(pipeline, "_write_json", lambda path, value: manifests.append(value))

    result = auto_story_worker._retry_failed_story_sync(job_id, "arc-1")

    assert [story["storyline"]["id"] for story in result["stories"]] == ["arc-1", "arc-2"]
    assert result["stories"][1] == completed_story
    assert result["failed_stories"] == []
    assert (output_dir / "story-1_partial.tmp").exists() is False
    assert manifests[0]["story_count"] == 2
    assert result["output_files"] == [
        {"filename": "story-1_video.mp4"},
        {"filename": "story-2_video.mp4"},
        {"filename": "story_plan_manifest.json", "download_url": f"/api/outputs/{job_id}/story_plan_manifest.json"},
    ]


def test_failed_story_retry_endpoint_validates_checkpoint_and_queues_retry(api, monkeypatch):
    client, project_dir = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    state = {
        "phase": "completed",
        "source_id": "episode-test",
        "transcript_key": "cache-key",
        "storylines": [{"id": "arc-1", "title": "Failed story"}],
        "failed_stories": [{
            "storyline_id": "arc-1",
            "story_id": "story-1",
            "title": "Failed story",
            "reason": "Render failed.",
        }],
        "stories": [{"story_id": "story-2", "storyline": {"id": "arc-2"}}],
    }
    store.update_job(job_id, status=JobStatus.COMPLETED.value, auto_story=state)
    source_path = project_dir / "uploads" / "uploaded_episode.mp4"
    source_path.write_bytes(b"episode")
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [{"word": "hi"}])
    submitted = []

    async def fake_retry(submitted_job_id, storyline_id):
        submitted.append((submitted_job_id, storyline_id))

    monkeypatch.setattr(auto_story_worker, "submit_failed_story_retry", fake_retry)

    response = client.post(f"/api/auto-story/{job_id}/failed-stories/arc-1/retry")

    assert response.status_code == 200, response.text
    assert submitted == [(job_id, "arc-1")]
    assert response.json()["status"] == JobStatus.ANALYZING.value
    assert response.json()["auto_story"]["retry_origin_status"] == JobStatus.COMPLETED.value
    assert response.json()["auto_story"]["story_retry_counts"] == {"arc-1": 1}


def test_cancelling_failed_story_retry_restores_completed_checkpoint(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        status=JobStatus.ANALYZING.value,
        auto_story={
            "phase": "retrying_story",
            "retry_origin_status": JobStatus.COMPLETED.value,
            "retrying_storyline_id": "arc-1",
            "stories": [{"story_id": "story-2"}],
        },
    )

    def cancel_retry(*_args):
        raise worker.JobCancelled()

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(auto_story_worker, "_check_cancelled", lambda _job_id: None)
    monkeypatch.setattr(worker, "run_serialized_call", run_inline)

    asyncio.run(auto_story_worker._run_stage(job_id, "retry_story", cancel_retry))

    saved_job = store.get_job(job_id)
    assert saved_job["status"] == JobStatus.COMPLETED.value
    assert saved_job["auto_story"]["stories"] == [{"story_id": "story-2"}]
    assert saved_job["auto_story"]["phase"] == "completed"
    assert "retry_origin_status" not in saved_job["auto_story"]
    assert saved_job["cancel_requested"] is False


def test_optional_review_stops_after_validated_plans(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "source_id": "episode-test",
            "source_duration_sec": 600.0,
            "transcript_key": "transcript-key",
            "storylines": [{"id": "arc-1", "title": "Story"}],
        },
    )
    monkeypatch.setattr(
        auto_story_worker,
        "_source_context",
        lambda _job: {"cfg": SimpleNamespace()},
    )
    monkeypatch.setattr(
        auto_story_worker,
        "_create_story_plan",
        lambda _state, _context, arc, story_id: {
            "story_id": story_id,
            "storyline": arc,
            "recipe": {"beats": []},
            "validation": {"estimated_runtime_sec": 60.0},
        },
    )
    monkeypatch.setattr(
        auto_story_worker,
        "_render_stories_sync",
        lambda *_args: pytest.fail("Review mode must stop before rendering."),
    )

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(worker, "run_serialized_call", run_inline)
    asyncio.run(auto_story_worker._run_stage(
        job_id,
        "generation",
        auto_story_worker._create_selected_stories_sync,
        ["arc-1"],
        True,
    ))

    saved_job = store.get_job(job_id)
    assert saved_job["status"] == JobStatus.AWAITING_RECIPE.value
    assert saved_job["auto_story"]["phase"] == "awaiting_recipe"
    assert saved_job["auto_story"]["stories"][0]["story_id"] == "story-1"
    assert saved_job["auto_story"]["validation"]["valid_story_count"] == 1


def test_all_unbuildable_stories_return_to_selection_instead_of_failing(api, monkeypatch):
    _, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    store.update_job(
        job_id,
        auto_story={
            "discovery_revision": 0,
            "source_id": "episode-test",
            "source_duration_sec": 3600.0,
            "transcript_key": "transcript-key",
            "storylines": [
                {"id": "arc-failed", "title": "Unschedulable story"}
            ],
        },
    )
    monkeypatch.setattr(
        auto_story_worker, "_source_context",
        lambda _job: {"cfg": SimpleNamespace()},
    )

    def reject_plan(*_args):
        raise ValueError("No feasible beat schedule.")

    async def run_inline(_job_id, function, *args):
        return function(*args)

    monkeypatch.setattr(auto_story_worker, "_create_story_plan", reject_plan)
    monkeypatch.setattr(worker, "run_serialized_call", run_inline)

    asyncio.run(auto_story_worker._run_stage(
        job_id,
        "generation",
        auto_story_worker._create_selected_stories_sync,
        ["arc-failed"],
    ))

    saved_job = store.get_job(job_id)
    assert saved_job["status"] == JobStatus.AWAITING_STORYLINE.value
    assert saved_job["error"] is None
    assert saved_job["auto_story"]["discovery_revision"] == 1
    assert saved_job["auto_story"]["selected_storyline_ids"] == []
    assert saved_job["auto_story"]["failed_stories"][0]["title"] == (
        "Unschedulable story"
    )


def test_complete_story_plan_validation_and_render_are_single_approval(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    stories = [
        {
            "story_id": f"story-{index}",
            "storyline": {"id": f"arc-{index}", "title": f"Story {index}"},
            "recipe": {"beats": []},
            "recipe_filename": f"story-{index}.json",
            "validation": {"beat_count": 8, "estimated_runtime_sec": 60.0},
        }
        for index in range(1, 5)
    ]
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_recipe",
            "stories": stories,
            "validation": {"valid_story_count": 4},
        },
        status=JobStatus.AWAITING_RECIPE.value,
    )
    submitted = []
    render_submissions = []

    async def fake_validate(_job_id, submitted_stories):
        submitted.extend(submitted_stories)
        return {
            "stories": stories,
            "validation": {
                "story_count": 4,
                "valid_story_count": 4,
                "estimated_runtime_sec": 240.0,
            },
        }

    async def fake_render(_job_id):
        render_submissions.append(_job_id)

    monkeypatch.setattr(auto_story_worker, "validate_stories", fake_validate)
    monkeypatch.setattr(auto_story_worker, "submit_render", fake_render)

    response = client.put(
        f"/api/auto-story/{job_id}/recipe",
        json={
            "stories": [
                {"story_id": story["story_id"], "recipe": story["recipe"]}
                for story in stories
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert len(submitted) == 4
    assert response.json()["auto_story"]["validation"]["valid_story_count"] == 4

    rendered = client.post(f"/api/auto-story/{job_id}/render")
    assert rendered.status_code == 200, rendered.text
    assert render_submissions == [job_id]


def test_reconstruct_story_only_targets_a_story_in_the_reviewed_plan(api, monkeypatch):
    client, _ = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    stories = [
        {
            "story_id": f"story-{index}",
            "storyline": {"id": f"arc-{index}", "title": f"Story {index}"},
            "validation": {"estimated_runtime_sec": 60.0},
        }
        for index in range(1, 5)
    ]
    store.update_job(
        job_id,
        auto_story={
            "phase": "awaiting_recipe",
            "stories": stories,
            "validation": {"valid_story_count": 4},
            "reconstruction_counts": {},
        },
        status=JobStatus.AWAITING_RECIPE.value,
    )
    submitted = []

    async def fake_reconstruct(_job_id, story_id, revision):
        submitted.append((story_id, revision))

    monkeypatch.setattr(
        auto_story_worker, "submit_reconstruction", fake_reconstruct
    )

    missing = client.post(
        f"/api/auto-story/{job_id}/stories/not-in-plan/reconstruct"
    )
    assert missing.status_code == 404

    response = client.post(
        f"/api/auto-story/{job_id}/stories/story-3/reconstruct"
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == JobStatus.ANALYZING.value
    assert response.json()["auto_story"]["reconstruction_counts"]["story-3"] == 1
    assert submitted == [("story-3", 1)]


def test_worker_validates_every_recipe_in_the_complete_plan(api, monkeypatch):
    _, tmp_path = api
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    stories = [
        {
            "story_id": f"story-{index}",
            "storyline": {"id": f"arc-{index}", "title": f"Story {index}"},
            "recipe": {
                "source_id": "episode-test",
                "storyline": {"id": f"arc-{index}"},
                "beats": [{"beat_id": 1}],
            },
            "recipe_filename": f"story-{index}.json",
            "validation": {"estimated_runtime_sec": 60.0},
        }
        for index in range(1, 5)
    ]
    state = {
        "source_id": "episode-test",
        "source_duration_sec": 3600.0,
        "stories": stories,
    }
    store.update_job(job_id, auto_story=state)
    cfg = SimpleNamespace(
        auto_story_target_duration=60,
        auto_story_min_beat_sec=3.5,
        auto_story_max_beat_sec=6.0,
        auto_story_min_gap_sec=30.0,
        auto_story_min_episode_span_ratio=0.4,
        auto_story_padding_sec=0.18,
        auto_story_crossfade_ms=60,
    )
    monkeypatch.setattr(
        auto_story_worker,
        "_source_context",
        lambda _job: {"cfg": cfg, "output_dir": str(tmp_path)},
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    validated_ids = []

    def fake_validate(recipe, arc, *_args):
        validated_ids.append(arc["id"])
        return {
            "beats": [{
                "start_word_id": "W1",
                "end_word_id": "W2",
                "dialogue_start_sec": 10.0,
                "dialogue_end_sec": 14.0,
                "text": arc["title"],
                "speaker_ids": [],
            }],
            "estimated_runtime_sec": 60.0,
            "episode_span_sec": 500.0,
        }

    monkeypatch.setattr(pipeline, "validate_review_recipe", fake_validate)
    monkeypatch.setattr(
        pipeline, "_build_review_timing_options", lambda *_args: {}
    )

    result = auto_story_worker._validate_stories_sync(
        job_id,
        [
            {"story_id": story["story_id"], "recipe": story["recipe"]}
            for story in stories
        ],
    )

    assert validated_ids == [f"arc-{index}" for index in range(1, 5)]
    assert result["validation"]["valid_story_count"] == 4
    assert len(result["stories"]) == 4
    assert all(
        (tmp_path / story["recipe_filename"]).is_file()
        for story in result["stories"]
    )


def test_story_span_requirement_scales_to_a_shorter_storyline(api):
    cfg = SimpleNamespace(auto_story_min_episode_span_ratio=0.4)

    ratio = auto_story_worker._story_span_ratio(
        cfg,
        {"start_sec": 1200.0, "end_sec": 1800.0},
        3600.0,
    )

    assert ratio == pytest.approx(0.4 * 600 / 3600)
    assert 3600 * ratio == pytest.approx(240.0)


def test_reconstructing_one_story_preserves_the_other_plans(api, monkeypatch, tmp_path):
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    stories = [
        {
            "story_id": f"story-{index}",
            "storyline": {
                "id": f"arc-{index}",
                "title": f"Story {index}",
                "source_ranges": [],
            },
            "recipe": {"beats": []},
            "validation": {"estimated_runtime_sec": 60.0},
        }
        for index in range(1, 5)
    ]
    state = {
        "source_id": "episode-test",
        "source_duration_sec": 3600.0,
        "transcript_key": "transcript-key",
        "stories": stories,
    }
    store.update_job(job_id, auto_story=state)
    context = {
        "cfg": SimpleNamespace(
            auto_story_target_duration=60,
            auto_story_min_gap_sec=30.0,
        ),
        "auto_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "outputs"),
    }
    monkeypatch.setattr(
        auto_story_worker, "_source_context", lambda _job: context
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    discovery = {}

    def fake_discover(*_args, **kwargs):
        discovery.update(kwargs)
        return [{
            "id": "suggested-id",
            "title": "Replacement story",
            "summary": "A new setup. A different payoff.",
            "source_ranges": [],
        }]

    monkeypatch.setattr(pipeline, "_discover_storylines", fake_discover)
    monkeypatch.setattr(pipeline, "_write_json", lambda *_args: None)
    monkeypatch.setattr(
        auto_story_worker,
        "_create_story_plan",
        lambda _state, _context, arc, story_id: {
            "story_id": story_id,
            "storyline": arc,
            "recipe": {"beats": []},
            "validation": {"estimated_runtime_sec": 60.0},
        },
    )

    result = auto_story_worker._reconstruct_story_sync(
        job_id, "story-2", 1
    )

    assert [story["storyline"]["title"] for story in result["stories"]] == [
        "Story 1", "Replacement story", "Story 3", "Story 4",
    ]
    assert result["stories"][1]["storyline"]["id"] == "story-2-reconstruction-1"
    assert len(discovery["excluded_storylines"]) == 4
    assert discovery["count"] == 1


def test_multi_story_render_continues_after_one_story_fails(api, monkeypatch, tmp_path):
    job_id = store.create_job(
        upload_filename="uploaded_episode.mp4",
        source="local",
        config={"auto_story_mode": True},
    )
    stories = [
        {
            "story_id": f"story-{index}",
            "storyline": {"id": f"arc-{index}", "title": f"Story {index}"},
            "recipe": {"beats": []},
            "recipe_filename": f"story-{index}_recipe.json",
            "validation": {"estimated_runtime_sec": 60.0},
        }
        for index in range(1, 5)
    ]
    state = {
        "source_id": "episode-test",
        "source_name": "Episode",
        "source_duration_sec": 3600.0,
        "validation": {"valid_story_count": 4},
        "stories": stories,
    }
    store.update_job(job_id, auto_story=state)
    cfg = SimpleNamespace(
        auto_story_target_duration=60,
        auto_story_min_beat_sec=3.5,
        auto_story_max_beat_sec=6.0,
        auto_story_min_gap_sec=30.0,
        auto_story_min_episode_span_ratio=0.4,
        auto_story_padding_sec=0.18,
        auto_story_crossfade_ms=60,
    )
    monkeypatch.setattr(
        auto_story_worker,
        "_source_context",
        lambda _job: {
            "source_id": "episode-test",
            "cfg": cfg,
            "source_registry": {},
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "outputs"),
        },
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: [])
    monkeypatch.setattr(
        pipeline,
        "validate_review_recipe",
        lambda _recipe, arc, *_args: {
            "source_id": "episode-test",
            "storyline": arc,
            "target_duration_sec": 60,
            "estimated_runtime_sec": 60.0,
            "settings": {},
            "beats": [],
        },
    )
    prefixes = []

    def fake_render(*_args, output_prefix=None):
        prefixes.append(output_prefix)
        if output_prefix.startswith("story-1_"):
            raise RuntimeError("Synthetic render failure.")
        return (
            str(tmp_path / f"{output_prefix}_story_montage.mp4"),
            str(tmp_path / f"{output_prefix}_story_montage.ass"),
            60.0,
        )

    monkeypatch.setattr(pipeline, "_render_montage", fake_render)

    result = auto_story_worker._render_stories_sync(
        job_id, store.get_job(job_id), state
    )

    assert len(set(prefixes)) == 4
    assert len([
        item for item in result["output_files"]
        if item["filename"].endswith(".mp4")
    ]) == 3
    assert result["output_files"][-1]["filename"] == "story_plan_manifest.json"
    assert (tmp_path / "outputs" / "story_plan_manifest.json").is_file()
    plan_manifest = json.loads(
        (tmp_path / "outputs" / "story_plan_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan_manifest["story_count"] == 3
    assert plan_manifest["selected_story_count"] == 4
    assert plan_manifest["failed_stories"][0]["title"] == "Story 1"
    saved_state = store.get_job(job_id)["auto_story"]
    assert saved_state["rendered_story_count"] == 3
    assert saved_state["failed_stories"][0]["reason"] == (
        "Rendering failed: RuntimeError: Synthetic render failure."
    )
    assert len(saved_state["output_files"]) == len(result["output_files"]) - 1
