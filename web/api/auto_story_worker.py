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

    state = get_state(job_id)
    discovery_revision = int(state.get("discovery_revision", 0))
    _set_progress(
        job_id,
        JobStatus.ANALYZING,
        "storyline_discovery",
        2,
        "Finding the distinct story arcs this episode can support.",
        60,
    )
    storylines = pipeline._discover_storylines(
        words,
        duration,
        transcript_key,
        context["auto_cache_dir"],
        context["output_dir"],
        cfg,
        count=None,
        target_duration_sec=cfg.auto_story_target_duration,
        min_gap_sec=cfg.auto_story_min_gap_sec,
        cache_variant=f"discovery-{discovery_revision}",
    )
    return {
        "source_id": context["source_id"],
        "source_name": context["source"]["name"],
        "source_duration_sec": duration,
        "transcript_key": transcript_key,
        "storylines": storylines,
        "stories": [],
        "story_count": len(storylines),
        "discovery_revision": discovery_revision,
    }


def _create_story_plan(
    state: dict,
    context: dict,
    arc: dict,
    story_id: str,
) -> dict:
    cfg = context["cfg"]
    words = _load_words(state)
    crossfade_sec = cfg.auto_story_crossfade_ms / 1000
    story_span_ratio = _story_span_ratio(cfg, arc, state["source_duration_sec"])
    configured_gap = cfg.auto_story_min_gap_sec
    schedules = (
        (configured_gap, story_span_ratio),
        (configured_gap, story_span_ratio),
        (min(configured_gap, max(5.0, configured_gap * 0.5)), story_span_ratio * 0.5),
    )
    last_error = None
    for candidate_round, (min_gap, min_span_ratio) in enumerate(schedules):
        settings = {
            "target_duration_sec": cfg.auto_story_target_duration,
            "min_beat_sec": cfg.auto_story_min_beat_sec,
            "max_beat_sec": cfg.auto_story_max_beat_sec,
            "min_gap_sec": min_gap,
            "min_episode_span_ratio": min_span_ratio,
            "padding_sec": cfg.auto_story_padding_sec,
            "crossfade_sec": crossfade_sec,
        }
        try:
            response, cache_key = pipeline._select_beats(
                arc,
                words,
                cfg.auto_story_target_duration,
                cfg.auto_story_min_beat_sec,
                cfg.auto_story_max_beat_sec,
                min_gap,
                min_span_ratio,
                state["source_duration_sec"],
                cfg.auto_story_padding_sec,
                crossfade_sec,
                state["transcript_key"],
                context["auto_cache_dir"],
                cfg,
                candidate_round=candidate_round,
            )
            recipe_path = os.path.join(
                context["output_dir"],
                f"{story_id}_recipe_{cache_key[:12]}.json",
            )
            if os.path.exists(recipe_path):
                with open(recipe_path, "r", encoding="utf-8") as file:
                    recipe = json.load(file)
            else:
                recipe = pipeline._build_review_recipe(
                    response, arc, words, state["source_id"], settings
                )
            validated = pipeline.validate_review_recipe(
                recipe,
                arc,
                words,
                state["source_id"],
                state["source_duration_sec"],
                settings["target_duration_sec"],
                settings["min_beat_sec"],
                settings["max_beat_sec"],
                settings["min_gap_sec"],
                settings["min_episode_span_ratio"],
                settings["padding_sec"],
                settings["crossfade_sec"],
            )
            recipe = _normalize_review_recipe(recipe, validated)
            pipeline._write_json(recipe_path, recipe)
            return {
                "story_id": story_id,
                "storyline": arc,
                "recipe": recipe,
                "settings": settings,
                "timing_options": pipeline._build_review_timing_options(
                    recipe, arc, words
                ),
                "recipe_filename": os.path.basename(recipe_path),
                "validation": {
                    "beat_count": len(validated["beats"]),
                    "estimated_runtime_sec": validated["estimated_runtime_sec"],
                    "episode_span_sec": validated["episode_span_sec"],
                },
            }
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Could not schedule story {arc['title']!r} after three candidate "
        f"attempts, including a {schedules[-1][0]:.1f}-second pacing fallback: "
        f"{last_error}"
    ) from last_error


def _story_span_ratio(cfg, arc: dict, episode_duration: float) -> float:
    configured_ratio = cfg.auto_story_min_episode_span_ratio
    arc_start = float(arc.get("start_sec", 0.0))
    arc_end = float(arc.get("end_sec", episode_duration))
    arc_span = arc_end - arc_start
    if episode_duration <= 0 or arc_span <= 0:
        return configured_ratio
    return configured_ratio * min(1.0, arc_span / episode_duration)


def _aggregate_story_validation(stories: list[dict]) -> dict:
    return {
        "story_count": len(stories),
        "valid_story_count": sum(bool(item.get("validation")) for item in stories),
        "estimated_runtime_sec": sum(
            float((item.get("validation") or {}).get("estimated_runtime_sec", 0))
            for item in stories
        ),
        "stories": [
            {
                "story_id": item["story_id"],
                "title": item["storyline"]["title"],
                **(item.get("validation") or {}),
            }
            for item in stories
        ],
    }


def _create_selected_stories_sync(
    job_id: str,
    storyline_ids: list[str],
) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    storylines = state.get("storylines", [])
    by_id = {arc["id"]: arc for arc in storylines}
    if not storyline_ids or any(story_id not in by_id for story_id in storyline_ids):
        raise ValueError("Select one or more storylines from the discovered list.")
    if len(set(storyline_ids)) != len(storyline_ids):
        raise ValueError("Selected storylines must be unique.")

    stories = []
    failed_stories = []
    for index, storyline_id in enumerate(storyline_ids, 1):
        _check_cancelled(job_id)
        arc = by_id[storyline_id]
        store.update_progress(
            job_id,
            "story_planning",
            3,
            _TOTAL_STEPS,
            f"Creating story {index} of {len(storyline_ids)}: {arc['title']}.",
            60 + (20 * (index - 1) / len(storyline_ids)),
        )
        try:
            story = _create_story_plan(
                state,
                context,
                arc,
                f"story-{len(stories) + 1}",
            )
        except ValueError as exc:
            failed_stories.append({
                "storyline_id": storyline_id,
                "title": arc["title"],
                "reason": str(exc),
            })
            state = _update_state(
                job_id,
                selected_storyline_ids=storyline_ids,
                selected_story_count=len(storyline_ids),
                stories=list(stories),
                failed_stories=list(failed_stories),
                phase="planning",
                message=(
                    f"Story {index} could not be scheduled. Continuing with "
                    f"the other {len(storyline_ids) - index} selected stories."
                ),
            )
            continue
        stories.append(story)
        state = _update_state(
            job_id,
            selected_storyline_ids=storyline_ids,
            selected_story_count=len(storyline_ids),
            stories=list(stories),
            failed_stories=list(failed_stories),
            story_count=len(storylines),
            validation=_aggregate_story_validation(stories),
            output_files=[],
            rendered_story_count=0,
            phase="planning",
            message=(
                f"Created and validated {index} of {len(storyline_ids)} "
                "selected stories."
            ),
        )

    _check_cancelled(job_id)
    if not stories:
        failures = "; ".join(
            f"{item['title']}: {item['reason']}" for item in failed_stories
        )
        raise ValueError(
            f"None of the {len(storyline_ids)} selected stories could be "
            f"scheduled. {failures}"
        )
    state = _update_state(
        job_id,
        stories=stories,
        failed_stories=failed_stories,
        selected_story_count=len(storyline_ids),
        story_count=len(storylines),
        validation=_aggregate_story_validation(stories),
        phase="rendering",
        message=(
            f"Rendering {len(stories)} stories"
            + (
                f"; skipping {len(failed_stories)} unbuildable storyline(s)."
                if failed_stories else "."
            )
        ),
    )
    return _render_stories_sync(job_id, _job(job_id), state)


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
    validated = pipeline.validate_review_recipe(
        recipe,
        arc,
        words,
        state["source_id"],
        state["source_duration_sec"],
        settings["target_duration_sec"],
        settings["min_beat_sec"],
        settings["max_beat_sec"],
        settings["min_gap_sec"],
        settings["min_episode_span_ratio"],
        settings["padding_sec"],
        settings["crossfade_sec"],
    )
    recipe = _normalize_review_recipe(recipe, validated)
    pipeline._write_json(review_path, recipe)
    return {
        "selected_storyline_id": storyline_id,
        "recipe": recipe,
        "timing_options": pipeline._build_review_timing_options(
            recipe, arc, words
        ),
        "recipe_filename": os.path.basename(review_path),
        "validation": {
            "beat_count": len(validated["beats"]),
            "estimated_runtime_sec": validated["estimated_runtime_sec"],
            "episode_span_sec": validated["episode_span_sec"],
        },
    }


def _normalize_review_recipe(recipe: dict, validated: dict) -> dict:
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
    return {**recipe, "beats": normalized_beats}


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
    normalized_recipe = _normalize_review_recipe(recipe, validated)
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


def _validate_stories_sync(job_id: str, submitted_stories: list[dict]) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    cfg = context["cfg"]
    planned_stories = state.get("stories")
    if not isinstance(planned_stories, list) or not planned_stories:
        raise ValueError("The complete story plan is missing; start a new workflow.")
    if not isinstance(submitted_stories, list) or len(submitted_stories) != len(planned_stories):
        raise ValueError(
            f"Submit all {len(planned_stories)} stories together for validation."
        )

    planned_by_id = {item.get("story_id"): item for item in planned_stories}
    submitted_by_id = {}
    for submitted in submitted_stories:
        if not isinstance(submitted, dict):
            raise ValueError("Each submitted story must include its story ID and recipe.")
        story_id = submitted.get("story_id")
        if story_id not in planned_by_id or story_id in submitted_by_id:
            raise ValueError("Submitted stories must match the complete plan exactly.")
        submitted_by_id[story_id] = submitted.get("recipe")
    if set(submitted_by_id) != set(planned_by_id):
        raise ValueError("Submitted stories must match the complete plan exactly.")

    words = _load_words(state)
    validated_stories = []
    for index, planned in enumerate(planned_stories, 1):
        story_id = planned["story_id"]
        recipe = submitted_by_id[story_id]
        if not isinstance(recipe, dict):
            raise ValueError(f"Story {index} must include a recipe object.")
        arc = planned["storyline"]
        try:
            validated = pipeline.validate_review_recipe(
                recipe,
                arc,
                words,
                state["source_id"],
                state["source_duration_sec"],
                cfg.auto_story_target_duration,
                cfg.auto_story_min_beat_sec,
                cfg.auto_story_max_beat_sec,
                planned.get("settings", {}).get(
                    "min_gap_sec", cfg.auto_story_min_gap_sec
                ),
                planned.get("settings", {}).get(
                    "min_episode_span_ratio",
                    _story_span_ratio(cfg, arc, state["source_duration_sec"]),
                ),
                cfg.auto_story_padding_sec,
                cfg.auto_story_crossfade_ms / 1000,
            )
        except ValueError as exc:
            raise ValueError(
                f"Story {index} ({arc['title']}) needs changes: {exc}"
            ) from exc
        normalized_recipe = _normalize_review_recipe(recipe, validated)
        pipeline._write_json(
            os.path.join(context["output_dir"], planned["recipe_filename"]),
            normalized_recipe,
        )
        validated_stories.append({
            **planned,
            "recipe": normalized_recipe,
            "timing_options": pipeline._build_review_timing_options(
                normalized_recipe, arc, words
            ),
            "validation": {
                "beat_count": len(validated["beats"]),
                "estimated_runtime_sec": validated["estimated_runtime_sec"],
                "episode_span_sec": validated["episode_span_sec"],
            },
        })

    return {
        "stories": validated_stories,
        "validation": _aggregate_story_validation(validated_stories),
    }


def _reconstruct_story_sync(
    job_id: str,
    story_id: str,
    revision: int,
) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    context = _source_context(job)
    stories = state.get("stories")
    story_index = next(
        (
            index for index, item in enumerate(stories or [])
            if item.get("story_id") == story_id
        ),
        None,
    )
    if story_index is None:
        raise ValueError("Choose one of the stories in the current plan.")

    words = _load_words(state)
    existing_arcs = [
        item["storyline"] for item in stories
    ] + list(state.get("reconstructed_storylines", []))
    alternatives = pipeline._discover_storylines(
        words,
        state["source_duration_sec"],
        state["transcript_key"],
        context["auto_cache_dir"],
        context["output_dir"],
        context["cfg"],
        count=1,
        excluded_storylines=existing_arcs,
        cache_variant=f"reconstruction-{story_id}-{revision}",
        output_filename=f"storyline_reconstruction_{story_id}_{revision}.json",
        target_duration_sec=context["cfg"].auto_story_target_duration,
        min_gap_sec=context["cfg"].auto_story_min_gap_sec,
    )
    arc = dict(alternatives[0])
    arc["id"] = f"{story_id}-reconstruction-{revision}"
    replacement = _create_story_plan(state, context, arc, story_id)
    stories = list(stories)
    reconstructed_storylines = [
        *state.get("reconstructed_storylines", []),
        stories[story_index]["storyline"],
    ]
    stories[story_index] = replacement
    storylines = [item["storyline"] for item in stories]
    pipeline._write_json(
        os.path.join(context["output_dir"], "storyline_discovery.json"),
        {
            "source_duration_sec": state["source_duration_sec"],
            "story_count": len(storylines),
            "storylines": storylines,
        },
    )
    return {
        "stories": stories,
        "storylines": storylines,
        "reconstructed_storylines": reconstructed_storylines,
        "validation": _aggregate_story_validation(stories),
        "reconstructing_story_id": None,
        "message": f"Story {story_index + 1} has been reconstructed and checked.",
    }


def _render_story_sync(
    job_id: str,
    story: dict,
    story_index: int,
    story_count: int,
    state: dict,
    context: dict,
    cfg,
    words: list[dict],
) -> tuple[dict, list[dict]]:
    arc = story["storyline"]
    recipe = story.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError(f"Story {story_index} ({arc['title']}) is missing its recipe.")
    settings = story.get("settings") or {}
    validated = pipeline.validate_review_recipe(
        recipe,
        arc,
        words,
        state["source_id"],
        state["source_duration_sec"],
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        settings.get("min_gap_sec", cfg.auto_story_min_gap_sec),
        settings.get(
            "min_episode_span_ratio",
            _story_span_ratio(cfg, arc, state["source_duration_sec"]),
        ),
        cfg.auto_story_padding_sec,
        cfg.auto_story_crossfade_ms / 1000,
    )
    _set_progress(
        job_id,
        JobStatus.RENDERING,
        "render",
        4,
        f"Rendering story {story_index} of {story_count}: {arc['title']}.",
        85 + (14 * (story_index - 1) / story_count),
    )
    title_slug = pipeline._safe_source_name(arc["title"]).lower()
    output_prefix = f"{story['story_id']}_{title_slug}"
    video_path, captions_path, actual_runtime = pipeline._render_montage(
        validated,
        words,
        context["source_registry"],
        context["cache_dir"],
        context["output_dir"],
        cfg,
        output_prefix=output_prefix,
    )
    manifest_path = os.path.join(
        context["output_dir"], f"{output_prefix}_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "story_id": story["story_id"],
        "source_id": state["source_id"],
        "source_name": state["source_name"],
        "storyline": validated["storyline"],
        "target_duration_sec": validated["target_duration_sec"],
        "estimated_runtime_sec": validated["estimated_runtime_sec"],
        "actual_runtime_sec": actual_runtime,
        "settings": validated["settings"],
        "video_file": os.path.basename(video_path),
        "captions_file": os.path.basename(captions_path),
        "recipe_file": story["recipe_filename"],
        "beats": validated["beats"],
    }
    pipeline._write_json(manifest_path, manifest)
    filenames = [
        os.path.basename(video_path),
        os.path.basename(captions_path),
        os.path.basename(manifest_path),
        story["recipe_filename"],
    ]
    files = [
        {
            "filename": filename,
            "download_url": (
                f"/api/outputs/{quote(job_id, safe='')}/"
                f"{quote(filename, safe='')}"
            ),
        }
        for filename in filenames
    ]
    return {
        **story,
        "manifest": manifest,
        "manifest_file": os.path.basename(manifest_path),
        "actual_runtime_sec": actual_runtime,
        "output_files": files,
    }, files


def _render_stories_sync(job_id: str, job: dict, state: dict) -> dict:
    context = _source_context(job)
    cfg = context["cfg"]
    stories = state["stories"]
    if not state.get("validation") or any(
        not item.get("validation") for item in stories
    ):
        raise ValueError("Validate the complete story plan before rendering.")
    words = _load_words(state)
    rendered_stories = []
    output_files = []
    failed_stories = list(state.get("failed_stories", []))
    count = len(stories)

    def output_file(filename: str) -> dict:
        return {
            "filename": filename,
            "download_url": (
                f"/api/outputs/{quote(job_id, safe='')}/"
                f"{quote(filename, safe='')}"
            ),
        }

    for index, story in enumerate(stories, 1):
        _check_cancelled(job_id)
        try:
            rendered_story, files = _render_story_sync(
                job_id,
                story,
                index,
                count,
                state,
                context,
                cfg,
                words,
            )
        except worker.JobCancelled:
            raise
        except Exception as exc:
            arc = story["storyline"]
            failed_stories.append({
                "storyline_id": arc.get("id", story["story_id"]),
                "title": arc["title"],
                "reason": f"Rendering failed: {type(exc).__name__}: {exc}",
            })
            log_event(
                "auto_story_story_render_failed",
                component="auto_story",
                job_id=job_id,
                level="ERROR",
                error=exc,
            )
            _update_state(
                job_id,
                stories=[*rendered_stories, *stories[index:]],
                failed_stories=list(failed_stories),
                output_files=list(output_files),
                rendered_story_count=len(rendered_stories),
                phase="rendering",
                message=(
                    f"Story {index} failed to render. Continuing with the "
                    f"other {count - index} planned stories."
                ),
            )
            continue
        output_files.extend(files)
        rendered_stories.append(rendered_story)
        _update_state(
            job_id,
            stories=[
                *rendered_stories,
                *stories[index:],
            ],
            failed_stories=list(failed_stories),
            output_files=list(output_files),
            rendered_story_count=len(rendered_stories),
            phase="rendering",
            message=(
                f"Rendered story {index} of {count}: {arc['title']}."
            ),
        )

    if not rendered_stories:
        failures = "; ".join(
            f"{item['title']}: {item['reason']}" for item in failed_stories
        )
        raise ValueError(f"No selected stories could be rendered. {failures}")

    plan_manifest = {
        "schema_version": 1,
        "source_id": state["source_id"],
        "source_name": state["source_name"],
        "story_count": len(rendered_stories),
        "selected_story_count": state.get(
            "selected_story_count", len(state.get("selected_storyline_ids", stories))
        ),
        "failed_stories": failed_stories,
        "target_duration_sec_per_story": cfg.auto_story_target_duration,
        "stories": [
            {
                "story_id": story["story_id"],
                "storyline": story["storyline"],
                "target_duration_sec": cfg.auto_story_target_duration,
                "estimated_runtime_sec": story["validation"]["estimated_runtime_sec"],
                "actual_runtime_sec": story["actual_runtime_sec"],
                "manifest_file": story["manifest_file"],
                "video_file": story["manifest"]["video_file"],
                "recipe_file": story["recipe_filename"],
            }
            for story in rendered_stories
        ],
    }
    plan_manifest_path = os.path.join(
        context["output_dir"], "story_plan_manifest.json"
    )
    pipeline._write_json(plan_manifest_path, plan_manifest)
    output_files.append(output_file(os.path.basename(plan_manifest_path)))
    return {
        "stories": rendered_stories,
        "failed_stories": failed_stories,
        "output_files": output_files,
        "manifest": plan_manifest,
    }


def _render_sync(job_id: str) -> dict:
    job = _job(job_id)
    state = get_state(job_id)
    if state.get("stories"):
        return _render_stories_sync(job_id, job, state)
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
            count = len(result.get("storylines", []))
            state = _update_state(
                job_id,
                **result,
                phase="awaiting_storyline",
                message=(
                    f"Found {count} storylines. Choose which ones to create."
                    if count
                    else "No clear storylines found. Rerun story discovery."
                ),
            )
            store.update_job(
                job_id,
                status=JobStatus.AWAITING_STORYLINE.value,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "awaiting_storyline",
                2,
                _TOTAL_STEPS,
                state["message"],
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
        elif stage == "generation":
            state = _update_state(job_id, **result, phase="completed")
            skipped_count = len(state.get("failed_stories", []))
            rendered_count = len(state.get("stories", []))
            completion_message = (
                f"Rendered {rendered_count} selected stories; "
                f"{skipped_count} unbuildable storyline(s) were skipped."
                if skipped_count
                else "All selected stories are ready to download."
            )
            state["message"] = completion_message
            store.update_job(
                job_id,
                status=JobStatus.COMPLETED.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "done",
                _TOTAL_STEPS,
                _TOTAL_STEPS,
                completion_message,
                100,
            )
        elif stage == "reconstruction":
            state = _update_state(
                job_id,
                **result,
                phase="awaiting_recipe",
            )
            state.pop("reconstructing_story_id", None)
            store.update_job(
                job_id,
                status=JobStatus.AWAITING_RECIPE.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "awaiting_recipe",
                3,
                _TOTAL_STEPS,
                state.get("message", "The reconstructed story is ready for review."),
                80,
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
                (
                    "All Auto Story montages are ready to download."
                    if state.get("stories")
                    else "Auto Story montage is ready to download."
                ),
                100,
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
        if stage == "discovery":
            state = get_state(job_id)
            state.update(
                phase="awaiting_storyline",
                message=f"Story discovery failed: {exc}. You can rerun discovery.",
            )
            store.update_job(
                job_id,
                status=JobStatus.AWAITING_STORYLINE.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "awaiting_storyline",
                2,
                _TOTAL_STEPS,
                state["message"],
                60,
            )
            log_event(
                "auto_story_discovery_failed",
                component="auto_story",
                job_id=job_id,
                level="ERROR",
                error=exc,
            )
        elif (
            stage == "generation"
            and not get_state(job_id).get("stories")
            and get_state(job_id).get("failed_stories")
        ):
            state = get_state(job_id)
            state.update(
                phase="awaiting_storyline",
                discovery_revision=int(state.get("discovery_revision", 0)) + 1,
                selected_storyline_ids=[],
                selected_story_count=0,
                message=(
                    "No selected story produced an output. The failed arcs are "
                    "unchecked; select other stories or rerun discovery."
                ),
            )
            store.update_job(
                job_id,
                status=JobStatus.AWAITING_STORYLINE.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "awaiting_storyline",
                2,
                _TOTAL_STEPS,
                state["message"],
                65,
            )
            log_event(
                "auto_story_generation_retryable",
                component="auto_story",
                job_id=job_id,
                level="WARNING",
            )
        elif stage == "reconstruction":
            state = get_state(job_id)
            state.pop("reconstructing_story_id", None)
            state.update(
                phase="awaiting_recipe",
                message=f"Could not reconstruct this story: {exc}",
            )
            store.update_job(
                job_id,
                status=JobStatus.AWAITING_RECIPE.value,
                error=None,
                auto_story=state,
            )
            store.update_progress(
                job_id,
                "awaiting_recipe",
                3,
                _TOTAL_STEPS,
                state["message"],
                75,
            )
            log_event(
                "auto_story_reconstruction_failed",
                component="auto_story",
                job_id=job_id,
                level="ERROR",
                error=exc,
            )
        else:
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


async def submit_storyline_selection(
    job_id: str,
    storyline_ids: list[str],
) -> None:
    _start(
        job_id,
        "generation",
        _create_selected_stories_sync,
        storyline_ids,
    )


async def submit_rediscovery(job_id: str) -> None:
    _start(job_id, "discovery", _discover_sync)


async def submit_reconstruction(
    job_id: str,
    story_id: str,
    revision: int,
) -> None:
    _start(
        job_id,
        "reconstruction",
        _reconstruct_story_sync,
        story_id,
        revision,
    )


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


async def validate_stories(job_id: str, stories: list[dict]) -> dict:
    try:
        _check_cancelled(job_id)
        result = await worker.run_serialized_call(
            job_id, _validate_stories_sync, job_id, stories
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
