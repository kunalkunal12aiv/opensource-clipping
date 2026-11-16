"""Dashboard endpoints for the staged Auto Story-Recipe workflow."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException

from .. import auto_story_worker, config_adapter, store, worker
from ..models import (
    AutoStoryCreateRequest,
    AutoStoryRecipeRequest,
    AutoStorySelectionRequest,
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
    if not _required_key("HF_TOKEN"):
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
        "target_duration_sec": req.target_duration_sec,
        "ai_provider": provider,
        "source_name": req.source_name,
        "auto_story_whisper_model": "small",
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


@router.get("/{job_id}", response_model=JobResponse)
async def get_auto_story(job_id: str) -> JobResponse:
    """Return Auto Story progress and the current review state."""
    job = _require_auto_story_job(job_id)
    return _job_to_response(job)


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
            job_id, req.storyline_ids
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
