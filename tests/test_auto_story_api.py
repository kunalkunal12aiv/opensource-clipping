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
    assert job["auto_story"]["story_count"] == 0
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


def test_storyline_selection_and_rediscovery_are_available_after_discovery(api, monkeypatch):
    client, _ = api
    storylines = [
        {"id": f"arc-{index}", "title": f"Story {index}"}
        for index in range(1, 4)
    ]
    job_ids = []
    for revision in (2, 0):
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

    async def fake_select(job_id, storyline_ids):
        selected.append((job_id, storyline_ids))

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
    assert selected == [(job_ids[0], ["arc-1", "arc-3"])]

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
            "cfg": cfg,
            "auto_cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "outputs"),
        },
    )
    monkeypatch.setattr(auto_story_worker, "_load_words", lambda _state: words)
    monkeypatch.setattr(
        pipeline,
        "_select_beats",
        lambda *_args: ({
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
        config={"auto_story_mode": True},
    )
    store.update_job(job_id, auto_story={"phase": "queued"})
    context = {
        "cfg": SimpleNamespace(
            auto_story_target_duration=60,
            auto_story_min_gap_sec=30.0,
        ),
        "source_id": "episode-test",
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
    ):
        progress_callback(50, 100)
        phase_callback("diarization")
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
    assert "Choose which ones to create" in job["auto_story"]["message"]
    assert discovery_options["count"] is None
    assert job["auto_story"]["transcript_key"] == "key"
    assert 25.0 in progress_updates
    assert 45 in progress_updates


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

    def select_beats(*_args, candidate_round):
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
    assert created == [("arc-1", "story-1"), ("arc-3", "story-1")]
    assert [story["story_id"] for story in rendered] == ["story-1"]
    assert saved_job["status"] == JobStatus.COMPLETED.value
    assert saved_job["auto_story"]["phase"] == "completed"
    assert saved_job["auto_story"]["output_files"] == ["story.mp4"]
    assert saved_job["auto_story"]["failed_stories"] == [{
        "storyline_id": "arc-1",
        "title": "Story 1",
        "reason": "No valid schedule after automatic pacing fallbacks.",
    }]
    assert "1 unbuildable storyline(s) were skipped" in (
        saved_job["auto_story"]["message"]
    )


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
