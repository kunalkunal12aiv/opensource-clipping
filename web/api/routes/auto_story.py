"""Dashboard endpoints for the staged Auto Story-Recipe workflow."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from .. import auto_story_projects, auto_story_worker, config_adapter, store, worker
from ..models import (
    AutoStoryCreateRequest,
    AutoStoryRecipeRequest,
    AutoStorySelectionRequest,
    AutoStorySpeakerNamesRequest,
    AutoStorylineSelectionRequest,
    JobResponse,
    JobStatus,
)
from ..hardware import cuda_available
from ..readiness import get_readiness
from .jobs import _job_to_response

router = APIRouter(prefix="/api/auto-story", tags=["auto-story"])


def _require_auto_story_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None or not job.get("config", {}).get("auto_story_mode"):
        raise HTTPException(status_code=404, detail="Auto Story job not found.")
    return job


def _required_key(key_name: str) -> bool:
    env = worker.get_settings_env()
    return bool(env.get(key_name) or os.environ.get(key_name))


@router.post("", status_code=201, response_model=JobResponse)
async def create_auto_story(req: AutoStoryCreateRequest) -> JobResponse:
    """Create an Auto Story job and start transcript/storyline discovery."""
    try:
        upload_path = config_adapter.resolve_upload_path(req.upload_filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not os.path.isfile(upload_path):
        raise HTTPException(
            status_code=400,
            detail="Episode upload was not found. Select and upload the video again.",
        )

    provider = req.ai_provider.value
    provider_key = "GOOGLE_API_KEY" if provider == "gemini" else "NVIDIA_API_KEY"
    if not _required_key(provider_key):
        raise HTTPException(
            status_code=400,
            detail=f"Add {provider_key} in Studio Settings before starting Auto Story.",
        )
    if req.diarization_enabled and not _required_key("HF_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="Add HF_TOKEN in Studio Settings and accept the PyAnnote model terms before starting Auto Story.",
        )

    readiness = await asyncio.to_thread(get_readiness)
    if not readiness["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Renderer backend is not ready. Check /api/ready before starting Auto Story.",
        )

    use_cuda = await asyncio.to_thread(cuda_available)
    config = {
        "auto_story_mode": True,
        "workflow_mode": req.workflow_mode,
        "target_duration_sec": req.target_duration_sec,
        "min_gap_sec": req.min_gap_sec,
        "diarization_speakers": req.diarization_speakers,
        "diarization_enabled": req.diarization_enabled,
        "ai_provider": provider,
        "source_name": req.source_name,
        "auto_story_whisper_model": req.whisper_model,
        "auto_story_whisper_device": "cuda" if use_cuda else "cpu",
        "auto_story_whisper_compute_type": "float16" if use_cuda else "int8",
    }
    job_id = store.create_job(
        upload_filename=req.upload_filename,
        source="local",
        config=config,
    )
    store.update_job(
        job_id,
        auto_story={
            "phase": "queued",
            "target_duration_sec": req.target_duration_sec,
            "workflow_mode": req.workflow_mode,
            "story_count": 0,
            "source_name": req.source_name,
            "storylines": [],
            "stories": [],
            "discovery_revision": 0,
            "recipe": None,
            "validation": None,
            "output_files": [],
        },
    )
    try:
        await auto_story_worker.submit_discovery(job_id)
    except ValueError as exc:
        store.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=str(exc),
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.get("/{job_id}/project")
async def download_auto_story_project(job_id: str):
    """Export a self-contained project archive for backup or migration."""
    _require_auto_story_job(job_id)
    archive_path, download_name = await asyncio.to_thread(
        auto_story_projects.export_project, job_id
    )
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=download_name,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.post("/projects/import", status_code=201, response_model=JobResponse)
async def import_auto_story_project(file: UploadFile = File(...)) -> JobResponse:
    """Restore an exported project, including its source, cache, and renders."""
    job_id = await auto_story_projects.import_project(file)
    return _job_to_response(store.get_job(job_id))


@router.get("/projects")
async def list_auto_story_projects(
    offset: int = Query(0, ge=0, le=1_000_000),
    limit: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=180),
) -> dict:
    """Search and page compact project summaries without copying job payloads."""
    search = q.strip().casefold()
    matching = store.list_auto_story_project_summaries()
    if search:
        matching = [
            project for project in matching
            if any(
                search in str(project.get(key) or "").casefold()
                for key in ("source_name", "upload_filename", "id", "status")
            )
        ]
    total = len(matching)
    projects = matching[offset:offset + limit]
    return {
        "projects": projects,
        "total": total,
        "offset": offset,
        "limit": limit,
        "storage": store.get_persistence_status(),
    }


@router.get("/{job_id}", response_model=JobResponse)
async def get_auto_story(job_id: str) -> JobResponse:
    """Return Auto Story progress and the current review state."""
    job = _require_auto_story_job(job_id)
    return _job_to_response(job)


@router.get("/{job_id}/source")
async def stream_auto_story_source(job_id: str):
    """Stream the uploaded episode for transcript-turn audio review."""
    job = _require_auto_story_job(job_id)
    try:
        source_path = config_adapter.resolve_upload_path(job.get("upload_filename"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Episode source is unavailable.") from exc
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=404, detail="Episode source is unavailable.")
    media_type = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".flv": "video/x-flv",
        ".ts": "video/mp2t",
    }.get(os.path.splitext(source_path)[1].lower(), "application/octet-stream")
    return FileResponse(source_path, media_type=media_type)


@router.get("/{job_id}/diagnostics")
async def download_auto_story_diagnostics(job_id: str):
    """Download safe stage/cache diagnostics for support and troubleshooting."""
    job = _require_auto_story_job(job_id)
    state = job.get("auto_story") or {}
    config = job.get("config") or {}
    source_available = False
    source_size = None
    try:
        source_path = config_adapter.resolve_upload_path(job.get("upload_filename"))
        if os.path.isfile(source_path):
            source_available = True
            source_size = os.path.getsize(source_path)
    except (TypeError, ValueError, OSError):
        pass

    output_dir = Path(config_adapter.PROJECT_ROOT) / "outputs" / job_id
    outputs = []
    output_bytes = 0
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            if path.is_file() and not path.name.startswith("."):
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                outputs.append({"filename": path.name, "size_bytes": size})
                output_bytes += size

    cache_files = {"transcript": False, "diarization": False, "combined": False}
    source_id = state.get("source_id")
    valid_source_id = (
        isinstance(source_id, str)
        and re.fullmatch(r"episode-[a-f0-9]{24}", source_id)
    )
    if valid_source_id:
        cache_dir = (
            Path(config_adapter.PROJECT_ROOT)
            / "outputs" / "story_cache" / "auto_story" / source_id
        )
        cache_files = {
            "transcript": (cache_dir / "transcript_words.json").is_file(),
            "diarization": (cache_dir / "speaker_turns.json").is_file(),
            "combined": (cache_dir / "combined_transcript.json").is_file(),
        }

    error_type = state.get("diagnostic_error_type")
    if not isinstance(error_type, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", error_type):
        error_type = None
    cache_usage = state.get("analysis_cache")
    if not isinstance(cache_usage, dict):
        cache_usage = {}
    stage_runs = state.get("stage_runs")
    if not isinstance(stage_runs, list):
        stage_runs = []
    payload = {
        "format": "opensource-clipping-auto-story-diagnostics",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "status": job.get("status"),
        "job_store": store.get_persistence_status(),
        "phase": state.get("phase"),
        "last_failed_stage": state.get("last_failed_stage"),
        "error_type": error_type,
        "settings": {
            "ai_provider": config.get("ai_provider"),
            "target_duration_sec": config.get("target_duration_sec"),
            "min_gap_sec": config.get("min_gap_sec"),
            "diarization_speakers": config.get("diarization_speakers"),
            "whisper_model": config.get("auto_story_whisper_model"),
            "whisper_device": config.get("auto_story_whisper_device"),
            "whisper_compute_type": config.get("auto_story_whisper_compute_type"),
        },
        "source": {
            "available": source_available,
            "size_bytes": source_size,
            "content_id": source_id if valid_source_id else None,
        },
        "analysis_cache": {
            "usage": {
                key: value
                for key, value in cache_usage.items()
                if key in {"transcript", "diarization", "combined"}
                and isinstance(value, str)
                and value in {"reused", "migrated", "created", "missing"}
            },
            "files_available": cache_files,
        },
        "stage_runs": [
            {
                "stage": run.get("stage"),
                "elapsed_ms": run.get("elapsed_ms"),
                "outcome": run.get("outcome"),
            }
                for run in stage_runs[-50:]
            if isinstance(run, dict)
            and isinstance(run.get("stage"), str)
            and type(run.get("elapsed_ms")) in {int, float}
            and run.get("outcome") in {"completed", "failed", "cancelled"}
        ],
        "results": {
            "storyline_count": len(state.get("storylines") or []),
            "successful_story_count": len(state.get("stories") or []),
            "failed_story_count": len(state.get("failed_stories") or []),
            "output_count": len(outputs),
            "output_bytes": output_bytes,
            "outputs": sorted(outputs, key=lambda item: item["filename"]),
        },
        "privacy": "Transcript text, API keys, raw provider errors, and local filesystem paths are omitted.",
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="auto-story-{job_id}-diagnostics.json"'},
    )


@router.post("/{job_id}/failed-stories/{storyline_id}/retry", response_model=JobResponse)
async def retry_failed_story(job_id: str, storyline_id: str) -> JobResponse:
    """Retry only one failed storyline while preserving successful renders."""
    job = _require_auto_story_job(job_id)
    origin_status = job.get("status")
    if origin_status not in {
        JobStatus.COMPLETED.value,
        JobStatus.AWAITING_STORYLINE.value,
    }:
        raise HTTPException(status_code=409, detail="A failed story can only be retried at a project checkpoint.")
    state = dict(job.get("auto_story") or {})
    failed = next(
        (item for item in state.get("failed_stories", []) if item.get("storyline_id") == storyline_id),
        None,
    )
    if failed is None or not any(
        item.get("id") == storyline_id for item in state.get("storylines", [])
    ):
        raise HTTPException(status_code=404, detail="This storyline is not marked as failed in the project.")
    try:
        source_path = config_adapter.resolve_upload_path(job.get("upload_filename"))
        if not os.path.isfile(source_path):
            raise ValueError("source missing")
        await asyncio.to_thread(auto_story_worker._load_words, state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The source or saved transcript is unavailable. Import a complete project backup to retry this story.",
        ) from exc

    retry_counts = dict(state.get("story_retry_counts") or {})
    retry_counts[storyline_id] = int(retry_counts.get(storyline_id, 0)) + 1
    state.update(
        story_retry_counts=retry_counts,
        retrying_storyline_id=storyline_id,
        retry_origin_status=origin_status,
        phase="retrying_story",
        message=f"Retrying {failed.get('title') or 'the failed storyline'}; completed stories will stay available.",
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.ANALYZING.value,
        error=None,
        cancel_requested=False,
    )
    store.update_progress(
        job_id,
        "story_retry",
        3,
        4,
        state["message"],
        70,
    )
    try:
        await auto_story_worker.submit_failed_story_retry(job_id, storyline_id)
    except ValueError as exc:
        state.pop("retrying_storyline_id", None)
        state.pop("retry_origin_status", None)
        state.update(
            phase="completed" if origin_status == JobStatus.COMPLETED.value else "awaiting_storyline",
            message="The failed story retry could not start. Try again from this project.",
        )
        store.update_job(job_id, auto_story=state, status=origin_status)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/retry-render", response_model=JobResponse)
async def retry_auto_story_render(job_id: str) -> JobResponse:
    """Resume rendering from a saved, validated story plan after a failure."""
    job = _require_auto_story_job(job_id)
    if job.get("status") not in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
        raise HTTPException(status_code=409, detail="Only a failed or cancelled project can retry rendering.")
    state = dict(job.get("auto_story") or {})
    stories = state.get("stories") or []
    has_multi_story_plan = bool(stories) and bool(state.get("validation")) and all(
        item.get("validation") for item in stories
    )
    has_legacy_plan = (
        isinstance(state.get("recipe"), dict)
        and bool(state.get("validation"))
        and bool(state.get("selected_storyline_id"))
    )
    if not (has_multi_story_plan or has_legacy_plan):
        raise HTTPException(status_code=409, detail="This project has no complete validated render plan to resume.")
    try:
        context = await asyncio.to_thread(auto_story_worker._source_context, job)
        if context.get("source_id") != state.get("source_id"):
            raise ValueError("source changed")
        await asyncio.to_thread(auto_story_worker._load_words, state)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The original episode or saved transcript is unavailable. Restore a complete project backup before retrying rendering.",
        ) from exc

    state.update(
        phase="rendering",
        message="Resuming rendering from the saved plan and completed story outputs.",
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.RENDERING.value,
        error=None,
        cancel_requested=False,
        restart_recovery=None,
    )
    store.update_progress(
        job_id, "render", 4, 4, state["message"], 85,
    )
    try:
        await auto_story_worker.submit_render(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/storyline", response_model=JobResponse)
async def select_storyline(
    job_id: str, req: AutoStorylineSelectionRequest
) -> JobResponse:
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_STORYLINE.value:
        raise HTTPException(status_code=409, detail="This job is not waiting for storyline selection.")
    storylines = (job.get("auto_story") or {}).get("storylines", [])
    if not any(item.get("id") == req.storyline_id for item in storylines):
        raise HTTPException(status_code=422, detail="Choose one of the storylines shown for this episode.")

    state = dict(job.get("auto_story") or {})
    state.update(
        phase="beat_selection",
        selected_storyline_id=req.storyline_id,
        recipe=None,
        timing_options=None,
        validation=None,
        message="Selecting dialogue beats for the chosen storyline.",
    )
    store.update_job(
        job_id, auto_story=state, status=JobStatus.ANALYZING.value
    )
    store.update_progress(
        job_id, "beat_selection", 3, 4,
        "Selecting dialogue beats. This may take a few minutes.", 70,
    )
    try:
        await auto_story_worker.submit_selection(job_id, req.storyline_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/storylines/select", response_model=JobResponse)
async def select_storylines(
    job_id: str, req: AutoStorySelectionRequest
) -> JobResponse:
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_STORYLINE.value:
        raise HTTPException(
            status_code=409,
            detail="Storylines can only be selected after discovery.",
        )
    state = dict(job.get("auto_story") or {})
    available_ids = {
        item.get("id") for item in state.get("storylines", [])
    }
    if not set(req.storyline_ids).issubset(available_ids):
        raise HTTPException(
            status_code=422,
            detail="Select only storylines shown for this episode.",
        )
    state.update(
        phase="creating_stories",
        selected_storyline_ids=req.storyline_ids,
        selected_story_count=len(req.storyline_ids),
        review_before_render=req.review_before_render,
        stories=[],
        failed_stories=[],
        validation=None,
        output_files=[],
        rendered_story_count=0,
        message=(
            f"Creating and rendering {len(req.storyline_ids)} selected stories."
        ),
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.ANALYZING.value,
        error=None,
    )
    store.update_progress(
        job_id,
        "story_selection",
        3,
        4,
        state["message"],
        65,
    )
    try:
        await auto_story_worker.submit_storyline_selection(
            job_id, req.storyline_ids, req.review_before_render
        )
    except ValueError as exc:
        state.update(
            phase="awaiting_storyline",
            message="Could not start story creation. Please select again.",
        )
        store.update_job(
            job_id,
            auto_story=state,
            status=JobStatus.AWAITING_STORYLINE.value,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/storylines/rediscover", response_model=JobResponse)
async def rediscover_storylines(job_id: str) -> JobResponse:
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_STORYLINE.value:
        raise HTTPException(
            status_code=409,
            detail="Story discovery can only be rerun before selecting stories.",
        )
    state = dict(job.get("auto_story") or {})
    revision = int(state.get("discovery_revision", 0)) + 1
    state.update(
        phase="storyline_discovery",
        discovery_revision=revision,
        storylines=[],
        story_count=0,
        selected_storyline_ids=[],
        selected_story_count=0,
        failed_stories=[],
        message="Rerunning story discovery with a fresh AI request.",
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.ANALYZING.value,
        error=None,
    )
    store.update_progress(
        job_id,
        "storyline_discovery",
        2,
        4,
        state["message"],
        55,
    )
    try:
        await auto_story_worker.submit_rediscovery(job_id)
    except ValueError as exc:
        state.update(
            phase="awaiting_storyline",
            message="Could not rerun story discovery. Please try again.",
        )
        store.update_job(
            job_id,
            auto_story=state,
            status=JobStatus.AWAITING_STORYLINE.value,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.put("/{job_id}/speakers", response_model=JobResponse)
async def update_speaker_names(
    job_id: str, req: AutoStorySpeakerNamesRequest
) -> JobResponse:
    """Save friendly speaker labels and rerun discovery against the saved transcript."""
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_STORYLINE.value:
        raise HTTPException(status_code=409, detail="Speaker names can only be changed before selecting stories.")
    state = dict(job.get("auto_story") or {})
    try:
        raw_words = await asyncio.to_thread(auto_story_worker._load_uncorrected_words, state)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="Saved transcript is unavailable; retry analysis first.") from exc
    known = {word.get("speaker_id") for word in raw_words if word.get("speaker_id")}
    speaker_names = req.speaker_names if req.speaker_names is not None else (state.get("speaker_names") or {})
    if not set(speaker_names).issubset(known):
        raise HTTPException(status_code=422, detail="Speaker labels must match speakers detected in this episode.")
    corrections = list(state.get("speaker_corrections") or [])
    merges = req.speaker_merges if req.speaker_merges is not None else (state.get("speaker_merges") or {})
    try:
        merges = auto_story_worker._normalize_speaker_merges(merges, known)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        auto_story_worker._apply_speaker_corrections(raw_words, state)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="Saved speaker edits do not match this transcript; restore a valid project backup.") from exc
    if req.speaker_corrections is not None:
        for correction in req.speaker_corrections:
            start = int(correction["start_word_id"][1:]) - 1
            end = int(correction["end_word_id"][1:]) - 1
            if (
                start < 0 or end < start or end >= len(raw_words)
                or raw_words[start].get("word_id") != correction["start_word_id"]
                or raw_words[end].get("word_id") != correction["end_word_id"]
                or end - start > 2000
            ):
                raise HTTPException(status_code=422, detail="Speaker correction must use a valid transcript range of at most 2,001 words.")
            target = correction.get("speaker_id")
            if target is not None and target not in known:
                raise HTTPException(status_code=422, detail="Choose a detected speaker label or Unassigned.")
            corrections = [
                item for item in corrections
                if not (
                    item.get("start_word_id") == correction["start_word_id"]
                    and item.get("end_word_id") == correction["end_word_id"]
                )
            ]
            corrections.append(correction)
        correction_word_count = sum(
            int(item["end_word_id"][1:]) - int(item["start_word_id"][1:]) + 1
            for item in corrections
        )
        if len(corrections) > 1000 or correction_word_count > auto_story_worker._MAX_CORRECTED_TRANSCRIPT_WORDS:
            raise HTTPException(status_code=422, detail="This project has reached its speaker edit limit. Start a fresh Auto Story job from this episode; its transcript cache can be reused.")
    candidate_state = {
        **state,
        "speaker_corrections": corrections,
        "speaker_merges": merges,
    }
    try:
        auto_story_worker._apply_speaker_corrections(raw_words, candidate_state)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state.update(
        speaker_names=speaker_names,
        speaker_corrections=corrections,
        speaker_merges=merges,
        phase="storyline_discovery",
        discovery_revision=int(state.get("discovery_revision", 0)) + 1,
        storylines=[],
        story_count=0,
        message="Speaker names saved. Finding storylines again using those labels.",
    )
    store.update_job(job_id, auto_story=state, status=JobStatus.ANALYZING.value, error=None)
    store.update_progress(job_id, "storyline_discovery", 2, 4, state["message"], 55)
    try:
        await auto_story_worker.submit_rediscovery(job_id)
    except ValueError as exc:
        state.update(
            phase="awaiting_storyline",
            message="Speaker edits were saved, but discovery could not start. Please retry discovery.",
        )
        store.update_job(
            job_id,
            auto_story=state,
            status=JobStatus.AWAITING_STORYLINE.value,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.get("/{job_id}/speakers/transcript")
async def get_speaker_transcript(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
) -> dict:
    """Return a paged transcript grouped into editable speaker turns."""
    job = _require_auto_story_job(job_id)
    state = dict(job.get("auto_story") or {})
    try:
        raw_words = await asyncio.to_thread(auto_story_worker._load_uncorrected_words, state)
        words = await asyncio.to_thread(
            auto_story_worker._apply_speaker_corrections, raw_words, state
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="Saved transcript is unavailable; retry analysis first.") from exc
    turns, total = auto_story_worker._page_speaker_turns(words, offset, limit)
    return {
        "turns": turns,
        "offset": offset,
        "limit": limit,
        "total": total,
        "speaker_ids": sorted({word["speaker_id"] for word in raw_words if word.get("speaker_id")}),
        "has_more": offset + limit < total,
    }


@router.post("/{job_id}/retry-discovery", response_model=JobResponse)
async def retry_discovery_from_saved_analysis(job_id: str) -> JobResponse:
    """Retry post-transcription discovery while reusing saved transcript and speakers."""
    job = _require_auto_story_job(job_id)
    if job.get("status") not in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
        raise HTTPException(status_code=409, detail="Only a failed or cancelled project can be retried here.")
    try:
        source_path = config_adapter.resolve_upload_path(job.get("upload_filename"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="The episode source is unavailable. Import a project backup to restore it.") from exc
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=409, detail="The episode source is unavailable. Import a project backup to restore it.")

    state = dict(job.get("auto_story") or {})
    revision = int(state.get("discovery_revision", 0)) + 1
    state.update(
        phase="storyline_discovery",
        discovery_revision=revision,
        storylines=[],
        story_count=0,
        selected_storyline_ids=[],
        selected_story_count=0,
        stories=[],
        failed_stories=[],
        validation=None,
        recipe=None,
        output_files=[],
        error=None,
        message="Retrying story discovery from the saved transcript and speaker analysis.",
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.ANALYZING.value,
        error=None,
        cancel_requested=False,
    )
    store.update_progress(job_id, "storyline_discovery", 2, 4, state["message"], 55)
    try:
        await auto_story_worker.submit_rediscovery(job_id)
    except ValueError as exc:
        store.update_job(job_id, status=JobStatus.FAILED.value, error=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/review-again", response_model=JobResponse)
async def reopen_completed_project_for_review(job_id: str) -> JobResponse:
    """Reopen a completed multi-story project when its source/cache are present."""
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail="Only completed projects can reopen plan review.")
    state = dict(job.get("auto_story") or {})
    stories = state.get("stories")
    if not isinstance(stories, list) or not stories or any(
        not isinstance(story, dict) or not isinstance(story.get("recipe"), dict)
        for story in stories
    ):
        raise HTTPException(status_code=409, detail="This project has no editable multi-story plan.")
    try:
        source_path = config_adapter.resolve_upload_path(job.get("upload_filename"))
        if not os.path.isfile(source_path):
            raise ValueError("episode source missing")
        await asyncio.to_thread(auto_story_worker._load_words, state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The source transcript is unavailable. Import a complete project backup to edit this plan.",
        ) from exc
    state.update(
        phase="awaiting_recipe",
        message="Review the saved dialogue timings, validate your edits, then render again.",
    )
    store.update_job(
        job_id,
        status=JobStatus.AWAITING_RECIPE.value,
        auto_story=state,
        error=None,
        cancel_requested=False,
    )
    store.update_progress(
        job_id,
        "awaiting_recipe",
        3,
        4,
        state["message"],
        80,
    )
    return _job_to_response(store.get_job(job_id))


@router.put("/{job_id}/recipe", response_model=JobResponse)
async def validate_recipe(
    job_id: str, req: AutoStoryRecipeRequest
) -> JobResponse:
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_RECIPE.value:
        raise HTTPException(status_code=409, detail="This job is not waiting for recipe review.")
    state = dict(job.get("auto_story") or {})
    if req.stories is not None:
        submitted_stories = [
            item.model_dump(mode="json") for item in req.stories
        ]
        submitted_by_id = {
            item["story_id"]: item["recipe"] for item in submitted_stories
        }
        state["stories"] = [
            {
                **story,
                "recipe": submitted_by_id.get(story.get("story_id"), story.get("recipe")),
            }
            for story in state.get("stories", [])
        ]
        state.update(
            validation=None,
            phase="awaiting_recipe",
            message="Checking every story in the plan.",
        )
    else:
        submitted_stories = None
        state.update(
            recipe=req.recipe,
            validation=None,
            phase="awaiting_recipe",
            message="Checking the reviewed recipe.",
        )
    store.update_job(job_id, auto_story=state)
    try:
        if submitted_stories is not None:
            result = await auto_story_worker.validate_stories(
                job_id, submitted_stories
            )
        else:
            result = await auto_story_worker.validate_recipe(
                job_id, req.recipe
            )
    except ValueError as exc:
        label = "Plan" if submitted_stories is not None else "Recipe"
        state.update(message=f"{label} needs changes: {exc}")
        store.update_job(job_id, auto_story=state)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except worker.JobCancelled as exc:
        raise HTTPException(status_code=409, detail="Auto Story has been cancelled.") from exc

    state = dict((store.get_job(job_id) or {}).get("auto_story") or {})
    state.update(
        result,
        phase="awaiting_recipe",
        message=(
            "Every story is valid. Review the complete plan and approve rendering."
            if submitted_stories is not None
            else "Recipe validated. Approve rendering when ready."
        ),
    )
    store.update_job(job_id, auto_story=state)
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/stories/{story_id}/reconstruct", response_model=JobResponse)
async def reconstruct_story(job_id: str, story_id: str) -> JobResponse:
    job = _require_auto_story_job(job_id)
    if job.get("status") != JobStatus.AWAITING_RECIPE.value:
        raise HTTPException(
            status_code=409,
            detail="Stories can only be reconstructed while reviewing the plan.",
        )
    state = dict(job.get("auto_story") or {})
    stories = state.get("stories")
    if not isinstance(stories, list) or not any(
        item.get("story_id") == story_id for item in stories
    ):
        raise HTTPException(status_code=404, detail="Story not found in this plan.")
    if not state.get("validation") or any(
        not item.get("validation") for item in stories
    ):
        raise HTTPException(
            status_code=409,
            detail="Validate the complete story plan before reconstructing a story.",
        )

    counts = dict(state.get("reconstruction_counts") or {})
    revision = int(counts.get(story_id, 0)) + 1
    counts[story_id] = revision
    state.update(
        reconstruction_counts=counts,
        reconstructing_story_id=story_id,
        phase="reconstructing",
        message="Reconstructing one story; the other plans will be preserved.",
    )
    store.update_job(
        job_id,
        auto_story=state,
        status=JobStatus.ANALYZING.value,
        error=None,
    )
    store.update_progress(
        job_id,
        "story_reconstruction",
        3,
        4,
        f"Reconstructing {story_id} without changing the other stories.",
        70,
    )
    try:
        await auto_story_worker.submit_reconstruction(
            job_id, story_id, revision
        )
    except ValueError as exc:
        state.pop("reconstructing_story_id", None)
        state.update(
            phase="awaiting_recipe",
            message="The story could not be reconstructed.",
        )
        store.update_job(
            job_id,
            auto_story=state,
            status=JobStatus.AWAITING_RECIPE.value,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.post("/{job_id}/render", response_model=JobResponse)
async def render_auto_story(job_id: str) -> JobResponse:
    job = _require_auto_story_job(job_id)
    state = job.get("auto_story") or {}
    has_story_plan = bool(state.get("stories"))
    plan_is_valid = bool(state.get("validation"))
    if has_story_plan:
        plan_is_valid = plan_is_valid and all(
            item.get("validation") for item in state["stories"]
        )
    if job.get("status") != JobStatus.AWAITING_RECIPE.value or not plan_is_valid:
        raise HTTPException(
            status_code=409,
            detail=(
                "Validate the complete story plan before approving rendering."
                if has_story_plan
                else "Validate the reviewed recipe before approving rendering."
            ),
        )
    state = dict(state)
    state.update(
        phase="rendering",
        message=(
            "Rendering all approved stories."
            if has_story_plan
            else "Rendering the approved montage."
        ),
    )
    store.update_job(
        job_id, auto_story=state, status=JobStatus.RENDERING.value
    )
    store.update_progress(
        job_id,
        "render",
        4,
        4,
        (
            "Rendering each approved story and its captions."
            if has_story_plan
            else "Rendering the approved montage and captions."
        ),
        85,
    )
    try:
        await auto_story_worker.submit_render(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_to_response(store.get_job(job_id))


@router.delete("/{job_id}")
async def cancel_auto_story(job_id: str) -> dict:
    _require_auto_story_job(job_id)
    if not auto_story_worker.cancel(job_id):
        raise HTTPException(status_code=409, detail="Auto Story job is already finished.")
    return {
        "id": job_id,
        "cancel_requested": True,
        "message": "Cancellation requested; the current model operation will finish safely.",
    }
