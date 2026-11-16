"""Background stages for the dashboard-guided Auto Story workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from urllib.parse import quote

from clipping import auto_story as pipeline
from scripts.studio_logging import log_event

from . import config_adapter, store, worker
from .models import JobStatus

_tasks: dict[str, asyncio.Task] = {}
_TOTAL_STEPS = 4


def _job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None or not job.get("config", {}).get("auto_story_mode"):
        raise ValueError("Auto Story job not found.")
    return job


def get_state(job_id: str) -> dict:
    return dict(_job(job_id).get("auto_story") or {})


def _build_config(job: dict):
    settings = job.get("config", {})
    payload = {
        "source": "local",
        "upload_filename": job.get("upload_filename"),
        "ai_provider": settings.get("ai_provider", "gemini"),
        "target_duration_sec": settings.get("target_duration_sec", 60),
        "auto_story_mode": True,
        "auto_story_whisper_model": settings.get("auto_story_whisper_model", "small"),
        "auto_story_whisper_device": settings.get("auto_story_whisper_device", "cpu"),
        "auto_story_whisper_compute_type": settings.get(
            "auto_story_whisper_compute_type", "int8"
        ),
        "whisper_model": settings.get("auto_story_whisper_model", "small"),
        "whisper_device": settings.get("auto_story_whisper_device", "cpu"),
        "whisper_compute_type": settings.get("auto_story_whisper_compute_type", "int8"),
    }
    return config_adapter.build_config_from_payload(
        payload,
        job["id"],
        env_overrides=worker.get_settings_env(),
    )


def _source_context(job: dict) -> dict:
    cfg = _build_config(job)
    upload_filename = job.get("upload_filename")
    if not upload_filename:
        raise ValueError("Auto Story requires an uploaded episode.")
    video_path = config_adapter.resolve_upload_path(upload_filename)
    if not os.path.isfile(video_path):
        raise FileNotFoundError("Uploaded episode was not found. Upload it again.")

    source_id = "episode-" + hashlib.sha256(
        upload_filename.encode("utf-8")
    ).hexdigest()[:12]
    source = {
        "id": source_id,
        "name": job.get("config", {}).get("source_name") or upload_filename,
        "platform": "local",
        "local_path": video_path,
    }
    from clipping.story import source_manager

    cache_dir = source_manager.get_cache_dir(
        str(config_adapter.PROJECT_ROOT / "outputs")
    )
    auto_cache_dir = os.path.join(
        cache_dir, "auto_story", pipeline._safe_source_name(source_id)
    )
    os.makedirs(auto_cache_dir, exist_ok=True)
    return {
        "cfg": cfg,
        "source_id": source_id,
        "source": source,
        "source_registry": {source_id: source},
        "video_path": video_path,
        "cache_dir": cache_dir,
        "auto_cache_dir": auto_cache_dir,
        "output_dir": cfg.auto_story_output_dir,
    }


def _load_words(state: dict) -> list[dict]:
    source_id = state.get("source_id")
    transcript_key = state.get("transcript_key")
    if not source_id or not transcript_key:
        raise RuntimeError("Cached transcript details are missing; restart the workflow.")
    cache_dir = os.path.join(
        str(config_adapter.PROJECT_ROOT / "outputs"),
        "story_cache",
        "auto_story",
        pipeline._safe_source_name(source_id),
    )
    cached = pipeline._read_cache(
        os.path.join(cache_dir, "combined_transcript.json"), transcript_key
    )
    if cached is None or not isinstance(cached.get("words"), list):
        raise RuntimeError("Cached episode transcript is missing; start a new workflow.")
    return cached["words"]


def _set_progress(
    job_id: str,
    status: JobStatus,
    step: str,
    step_number: int,
    message: str,
    percent: float,
) -> None:
    store.set_status(job_id, status)
    store.update_progress(
        job_id,
        step=step,
        step_number=step_number,
        total_steps=_TOTAL_STEPS,
        message=message,
        percent=percent,
    )


def _update_state(job_id: str, **updates) -> dict:
    state = get_state(job_id)
    state.update(updates)
    store.update_job(job_id, auto_story=state)
    return state


def _check_cancelled(job_id: str) -> None:
    if store.is_cancel_requested(job_id):
        raise worker.JobCancelled()


def _mark_cancelled(job_id: str) -> None:
    state = get_state(job_id)
    state.update(phase="cancelled", message="Auto Story cancelled.")
    store.update_job(
        job_id,
        status=JobStatus.CANCELLED.value,
        error=None,
        auto_story=state,
    )
    store.update_progress(
        job_id, "cancelled", 0, _TOTAL_STEPS, "Auto Story cancelled.", 0
    )
    log_event("auto_story_cancelled", component="auto_story", job_id=job_id)


def _mark_failed(job_id: str, exc: Exception) -> None:
    state = get_state(job_id)
    state.update(phase="failed", message=str(exc))
    store.update_job(
        job_id,
        status=JobStatus.FAILED.value,
        error=f"{type(exc).__name__}: {exc}",
        auto_story=state,
    )
    store.update_progress(
        job_id, "error", 0, _TOTAL_STEPS, f"{type(exc).__name__}: {exc}", 0
    )
    log_event(
        "auto_story_failed",
        component="auto_story",
        job_id=job_id,
        level="ERROR",
        error=exc,
    )


def _discover_sync(job_id: str) -> dict:
    job = _job(job_id)
    context = _source_context(job)
    cfg = context["cfg"]
    pipeline._provider_model(cfg)
    _check_cancelled(job_id)

    _set_progress(
        job_id,
        JobStatus.TRANSCRIBING,
        "transcribe",
        1,
        "Transcribing the episode and identifying speakers. This can take a while.",
        10,
    )

    last_progress_percent = 10

    def report_transcription_progress(completed: float, total: float) -> None:
        nonlocal last_progress_percent
        if total <= 0:
            return
        fraction = min(1.0, max(0.0, completed / total))
        percent = min(40, 10 + int(fraction * 30))
        if percent <= last_progress_percent:
            return
        last_progress_percent = percent
        store.update_progress(
            job_id,
            "transcribe",
            1,
            _TOTAL_STEPS,
            f"Transcribing episode: {round(fraction * 100)}% of audio processed.",
            float(percent),
        )

    def report_pipeline_phase(phase: str) -> None:
        if phase == "diarization":
            store.update_progress(
                job_id,
                "speaker_diarization",
                1,
                _TOTAL_STEPS,
                "Transcription complete. Identifying speakers with PyAnnote.",
                45,
            )

    duration = pipeline._probe_duration(context["video_path"])
    words, transcript_key = pipeline._load_or_create_transcript(
        context["video_path"],
        context["source_id"],
        context["auto_cache_dir"],
        cfg,
        duration,
        progress_callback=report_transcription_progress,
        phase_callback=report_pipeline_phase,
    )
    _check_cancelled(job_id)

    _set_progress(
        job_id,
        JobStatus.ANALYZING,
        "storyline_discovery",
        2,
        "Finding three possible storylines in the transcript.",
        60,
    )
    storylines = pipeline._discover_storylines(
        words,
        duration,
        transcript_key,
        context["auto_cache_dir"],
        context["output_dir"],
        cfg,
    )
    return {
        "source_id": context["source_id"],
        "source_name": context["source"]["name"],
        "source_duration_sec": duration,
        "transcript_key": transcript_key,
        "storylines": storylines,
    }


def _select_beats_sync(job_id: str, storyline_id: str) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    cfg = context["cfg"]
    arc = next(
        (item for item in state.get("storylines", []) if item["id"] == storyline_id),
        None,
    )
    if arc is None:
        raise ValueError("Choose one of the storylines shown for this episode.")
    words = _load_words(state)
    crossfade_sec = cfg.auto_story_crossfade_ms / 1000
    settings = {
        "target_duration_sec": cfg.auto_story_target_duration,
        "min_beat_sec": cfg.auto_story_min_beat_sec,
        "max_beat_sec": cfg.auto_story_max_beat_sec,
        "min_gap_sec": cfg.auto_story_min_gap_sec,
        "min_episode_span_ratio": cfg.auto_story_min_episode_span_ratio,
        "padding_sec": cfg.auto_story_padding_sec,
        "crossfade_sec": crossfade_sec,
    }
    response, cache_key = pipeline._select_beats(
        arc,
        words,
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        cfg.auto_story_min_gap_sec,
        cfg.auto_story_min_episode_span_ratio,
        state["source_duration_sec"],
        cfg.auto_story_padding_sec,
        crossfade_sec,
        state["transcript_key"],
        context["auto_cache_dir"],
        cfg,
    )
    review_path = os.path.join(
        context["output_dir"], f"story_recipe_review_{cache_key[:12]}.json"
    )
    if os.path.exists(review_path):
        with open(review_path, "r", encoding="utf-8") as file:
            recipe = json.load(file)
    else:
        recipe = pipeline._build_review_recipe(
            response, arc, words, state["source_id"], settings
        )
        pipeline._write_json(review_path, recipe)
    return {
        "selected_storyline_id": storyline_id,
        "recipe": recipe,
        "timing_options": pipeline._build_review_timing_options(
            recipe, arc, words
        ),
        "recipe_filename": os.path.basename(review_path),
    }


def _validate_recipe_sync(job_id: str, recipe: dict) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    arc = next(
        (
            item for item in state.get("storylines", [])
            if item.get("id") == state.get("selected_storyline_id")
        ),
        None,
    )
    if arc is None:
        raise ValueError("Select a storyline before reviewing its recipe.")
    words = _load_words(state)
    cfg = context["cfg"]
    filename = state.get("recipe_filename")
    if not filename:
        raise RuntimeError("Recipe review file is missing.")
    pipeline._write_json(os.path.join(context["output_dir"], filename), recipe)
    validated = pipeline.validate_review_recipe(
        recipe,
        arc,
        words,
        state["source_id"],
        state["source_duration_sec"],
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        cfg.auto_story_min_gap_sec,
        cfg.auto_story_min_episode_span_ratio,
        cfg.auto_story_padding_sec,
        cfg.auto_story_crossfade_ms / 1000,
    )
    normalized_beats = []
    for submitted, accepted in zip(recipe["beats"], validated["beats"]):
        normalized = dict(submitted)
        for field in (
            "start_word_id",
            "end_word_id",
            "dialogue_start_sec",
            "dialogue_end_sec",
            "text",
            "speaker_ids",
        ):
            normalized[field] = accepted[field]
        normalized_beats.append(normalized)
    normalized_recipe = {**recipe, "beats": normalized_beats}
    pipeline._write_json(
        os.path.join(context["output_dir"], filename), normalized_recipe
    )
    return {
        "recipe": normalized_recipe,
        "validation": {
            "beat_count": len(validated["beats"]),
            "estimated_runtime_sec": validated["estimated_runtime_sec"],
            "episode_span_sec": validated["episode_span_sec"],
        },
    }


def _render_sync(job_id: str) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    cfg = context["cfg"]
    arc = next(
        (
            item for item in state.get("storylines", [])
            if item.get("id") == state.get("selected_storyline_id")
        ),
        None,
    )
    if arc is None or not state.get("recipe"):
        raise ValueError("A reviewed recipe is required before rendering.")
    words = _load_words(state)
    validated = pipeline.validate_review_recipe(
        state["recipe"],
        arc,
        words,
        state["source_id"],
        state["source_duration_sec"],
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        cfg.auto_story_min_gap_sec,
        cfg.auto_story_min_episode_span_ratio,
        cfg.auto_story_padding_sec,
        cfg.auto_story_crossfade_ms / 1000,
    )
    _check_cancelled(job_id)
    _set_progress(
        job_id,
        JobStatus.RENDERING,
        "render",
        4,
        "Rendering the reviewed montage and captions.",
        85,
    )
    video_path, captions_path, actual_runtime = pipeline._render_montage(
        validated,
        words,
        context["source_registry"],
        context["cache_dir"],
        context["output_dir"],
        cfg,
    )
    manifest_path = os.path.join(context["output_dir"], "story_recipe_manifest.json")
    manifest = {
        "schema_version": 1,
        "source_id": state["source_id"],
        "source_name": state["source_name"],
        "storyline": validated["storyline"],
        "target_duration_sec": validated["target_duration_sec"],
        "estimated_runtime_sec": validated["estimated_runtime_sec"],
        "actual_runtime_sec": actual_runtime,
        "settings": validated["settings"],
        "video_file": os.path.basename(video_path),
        "captions_file": os.path.basename(captions_path),
        "recipe_file": state["recipe_filename"],
        "beats": validated["beats"],
    }
    pipeline._write_json(manifest_path, manifest)
    filenames = [
        os.path.basename(video_path),
        os.path.basename(captions_path),
        os.path.basename(manifest_path),
        state["recipe_filename"],
    ]
    return {
        "manifest": manifest,
        "actual_runtime_sec": actual_runtime,
        "output_files": [
            {
                "filename": filename,
                "download_url": (
                    f"/api/outputs/{quote(job_id, safe='')}/"
                    f"{quote(filename, safe='')}"
                ),
            }
            for filename in filenames
        ],
    }


async def _run_stage(job_id: str, stage: str, function, *args) -> None:
    try:
        _check_cancelled(job_id)
        log_event(
            "auto_story_stage_started",
            component="auto_story",
            job_id=job_id,
            stage=stage,
        )
        result = await worker.run_serialized_call(job_id, function, job_id, *args)
        _check_cancelled(job_id)
        if stage == "discovery":
            state = _update_state(job_id, **result, phase="awaiting_storyline")
            store.update_job(job_id, status=JobStatus.AWAITING_STORYLINE.value)
            store.update_progress(
                job_id,
                "awaiting_storyline",
                2,
                _TOTAL_STEPS,
                "Choose the storyline you want to build.",
                65,
            )
        elif stage == "selection":
            _update_state(job_id, **result, phase="awaiting_recipe")
            store.update_job(job_id, status=JobStatus.AWAITING_RECIPE.value)
            store.update_progress(
                job_id,
                "awaiting_recipe",
                3,
                _TOTAL_STEPS,
                "Review the proposed dialogue beats, then validate them.",
                75,
            )
        else:
            state = _update_state(job_id, **result, phase="completed")
            store.update_job(
                job_id,
                status=JobStatus.COMPLETED.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id, "done", _TOTAL_STEPS, _TOTAL_STEPS,
                "Auto Story montage is ready to download.", 100,
            )
        log_event(
            "auto_story_stage_completed",
            component="auto_story",
            job_id=job_id,
            stage=stage,
        )
    except worker.JobCancelled:
        _mark_cancelled(job_id)
    except Exception as exc:
        _mark_failed(job_id, exc)


def _start(job_id: str, stage: str, function, *args) -> None:
    current = _tasks.get(job_id)
    if current is not None and not current.done():
        raise ValueError("An Auto Story step is already running.")
    task = asyncio.create_task(_run_stage(job_id, stage, function, *args))
    _tasks[job_id] = task

    def forget(done_task: asyncio.Task) -> None:
        if _tasks.get(job_id) is done_task:
            _tasks.pop(job_id, None)

    task.add_done_callback(forget)


async def submit_discovery(job_id: str) -> None:
    _start(job_id, "discovery", _discover_sync)


async def submit_selection(job_id: str, storyline_id: str) -> None:
    _start(job_id, "selection", _select_beats_sync, storyline_id)


async def submit_render(job_id: str) -> None:
    _start(job_id, "render", _render_sync)


async def validate_recipe(job_id: str, recipe: dict) -> dict:
    try:
        _check_cancelled(job_id)
        result = await worker.run_serialized_call(
            job_id, _validate_recipe_sync, job_id, recipe
        )
        _check_cancelled(job_id)
        return result
    except worker.JobCancelled:
        _mark_cancelled(job_id)
        raise


def cancel(job_id: str) -> bool:
    job = _job(job_id)
    if job.get("status") in store.TERMINAL_STATES:
        return False
    if not store.request_cancel(job_id):
        return False
    state = get_state(job_id)
    state.update(phase="cancel_requested", message="Cancellation requested.")
    store.update_job(job_id, auto_story=state)
    if job_id not in _tasks and not worker.is_job_active(job_id):
        _mark_cancelled(job_id)
    return True


async def cancel_all() -> None:
    for job_id in list(_tasks):
        try:
            cancel(job_id)
        except ValueError:
            continue
