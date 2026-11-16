"""
web.api.models — Pydantic schemas for request/response validation.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    AWAITING_STORYLINE = "awaiting_storyline"
    AWAITING_RECIPE = "awaiting_recipe"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourcePlatform(str, enum.Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    GDRIVE = "gdrive"
    LOCAL = "local"


class AspectRatio(str, enum.Enum):
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"
    RATIO_4_5 = "4:5"


class FontStyle(str, enum.Enum):
    DEFAULT = "DEFAULT"
    STORYTELLER = "STORYTELLER"
    HORMOZI = "HORMOZI"
    CINEMATIC = "CINEMATIC"


class FaceDetector(str, enum.Enum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"


class AIProvider(str, enum.Enum):
    GEMINI = "gemini"
    NVIDIA = "nvidia"


class WhisperDevice(str, enum.Enum):
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Job Creation Request
# ---------------------------------------------------------------------------

class JobCreateRequest(BaseModel):
    """Payload to create a new clipping job."""

    # Source
    url: Optional[str] = Field(None, description="Video URL to process")
    upload_filename: Optional[str] = Field(None, description="Filename of an uploaded video")
    source: SourcePlatform = Field(SourcePlatform.YOUTUBE, description="Source platform")
    reuse_job_id: Optional[str] = Field(
        None, pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Existing Job ID to reuse its downloads and JSON",
    )

    # Main settings
    clips: int = Field(7, ge=1, le=30, description="Number of clips to generate")
    ratio: AspectRatio = Field(AspectRatio.RATIO_9_16, description="Output aspect ratio")
    source_height: str = Field("max", description="Source download max height")
    render_height: str = Field("1080", description="Target output height")

    # Content & Hook
    words_per_sub: int = Field(5, ge=1, le=15)
    hook_duration: int = Field(3, ge=1, le=10)
    use_broll: bool = True
    use_hook_glitch: bool = True
    use_auto_bgm: bool = True
    use_karaoke_effect: bool = True
    use_split_screen: bool = False
    use_camera_switch: bool = False
    no_subs: bool = False

    # Hook V2
    hook_v2: bool = False
    hook_v2_items: int = Field(3, ge=2, le=6)
    no_segment_trim: bool = False
    silence_trim: bool = False

    # Subtitle & Typography
    font_style: FontStyle = FontStyle.HORMOZI

    # Whisper
    whisper_model: str = "small"
    whisper_device: WhisperDevice = WhisperDevice.CPU
    whisper_compute_type: str = "int8"
    use_dlp_subs: bool = False

    # AI
    ai_provider: AIProvider = AIProvider.GEMINI
    gemini_model: str = "gemini-3-flash-preview"
    face_detector: FaceDetector = FaceDetector.MEDIAPIPE
    load_gemini_json: bool = False

    @model_validator(mode="before")
    @classmethod
    def default_cpu_compute_type(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = data.copy()
        if data.get("whisper_device") == WhisperDevice.CPU:
            data.setdefault("whisper_compute_type", "int8")
        return data

    @model_validator(mode="after")
    def reject_cpu_float16(self) -> "JobCreateRequest":
        if (
            self.whisper_device == WhisperDevice.CPU
            and self.whisper_compute_type == "float16"
        ):
            raise ValueError("Whisper float16 compute type requires a GPU; use int8 or float32 on CPU.")
        return self

    @field_validator("upload_filename")
    @classmethod
    def validate_upload_filename(cls, value: Optional[str]) -> Optional[str]:
        """Only accept the basename returned by the upload endpoint."""
        if value is not None and (
            not value or value in {".", ".."}
            or any(ch in value for ch in '/\\:\x00')
        ):
            raise ValueError("upload_filename must be a filename, not a path")
        return value


class AutoStoryCreateRequest(BaseModel):
    """Create a dashboard-guided Auto Story workflow from an uploaded episode."""

    upload_filename: str = Field(..., description="Filename returned by /api/upload")
    source_name: str = Field("Episode", min_length=1, max_length=180)
    target_duration_sec: int = Field(60, ge=60, le=120)
    story_count: Optional[int] = Field(
        None, ge=1, le=8, description="Legacy field; story count is discovered by AI."
    )
    ai_provider: AIProvider = AIProvider.GEMINI

    @field_validator("upload_filename")
    @classmethod
    def validate_upload_filename(cls, value: str) -> str:
        if not value or value in {".", ".."} or any(ch in value for ch in '/\\:\x00'):
            raise ValueError("upload_filename must be a filename, not a path")
        return value


class AutoStorylineSelectionRequest(BaseModel):
    storyline_id: str = Field(..., min_length=1, max_length=80)


class AutoStorySelectionRequest(BaseModel):
    storyline_ids: list[str] = Field(..., min_length=1, max_length=8)

    @model_validator(mode="after")
    def reject_duplicate_storylines(self) -> "AutoStorySelectionRequest":
        if len(set(self.storyline_ids)) != len(self.storyline_ids):
            raise ValueError("Selected storylines must be unique.")
        return self


class AutoStoryStoryRecipeRequest(BaseModel):
    story_id: str = Field(..., min_length=1, max_length=80)
    recipe: dict


class AutoStoryRecipeRequest(BaseModel):
    recipe: Optional[dict] = None
    stories: Optional[list[AutoStoryStoryRecipeRequest]] = None

    @model_validator(mode="after")
    def require_one_recipe_payload(self) -> "AutoStoryRecipeRequest":
        if (self.recipe is None) == (self.stories is None):
            raise ValueError("Provide either one recipe or the complete stories plan.")
        return self


# ---------------------------------------------------------------------------
# Job Progress Event
# ---------------------------------------------------------------------------

class JobProgressEvent(BaseModel):
    """Single progress event for SSE streaming."""
    step: str
    step_number: int
    total_steps: int
    message: str
    percent: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Clip Detail
# ---------------------------------------------------------------------------

class ClipDetail(BaseModel):
    """Metadata for a single rendered clip."""
    rank: int
    viral_score: Optional[int] = None
    title: Optional[str] = None
    title_en: Optional[str] = None
    filename: str
    duration: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    download_url: str
    thumbnail_url: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Job Response
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """Full job detail for API responses."""
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    url: Optional[str] = None
    upload_filename: Optional[str] = None
    source: SourcePlatform = SourcePlatform.YOUTUBE
    config: dict = Field(default_factory=dict)
    progress: Optional[JobProgressEvent] = None
    clips: list[ClipDetail] = Field(default_factory=list)
    error: Optional[str] = None
    cancel_requested: bool = False
    log: list[str] = Field(default_factory=list)
    auto_story: Optional[dict] = None


class JobListResponse(BaseModel):
    """Paginated job list."""
    jobs: list[JobResponse]
    total: int


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsRequest(BaseModel):
    """Settings update payload."""
    google_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    # Defaults
    default_clips: Optional[int] = None
    default_ratio: Optional[AspectRatio] = None
    default_font_style: Optional[FontStyle] = None
    default_whisper_model: Optional[str] = None
    default_whisper_device: Optional[WhisperDevice] = None
    default_ai_provider: Optional[AIProvider] = None


class SettingsResponse(BaseModel):
    """Current settings (keys are masked)."""
    google_api_key_set: bool = False
    pexels_api_key_set: bool = False
    hf_token_set: bool = False
    nvidia_api_key_set: bool = False
    default_clips: int = 7
    default_ratio: str = "9:16"
    default_font_style: str = "HORMOZI"
    default_whisper_model: str = "small"
    default_whisper_device: str = "cpu"
    default_ai_provider: str = "gemini"
    gpu_available: bool = False


class SystemHealthResponse(BaseModel):
    """System health check."""
    status: str = "ok"
    version: str
    gpu_available: bool
    ffmpeg_available: bool
    jobs_running: int
    jobs_queued: int
