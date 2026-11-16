#!/usr/bin/env python3
"""
Shared Video Post-Production Engine - Configuration

Common configuration defaults shared across HotClip and opensource-clipping adapters.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import json
from pathlib import Path


@dataclass
class VideoMetadata:
    """Normalized video metadata extracted from source files."""
    path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    bit_rate: Optional[int] = None
    pixel_format: Optional[str] = None
    has_video: bool = True
    has_audio: bool = True

    @classmethod
    def from_hotclip(cls, file_dict: dict) -> 'VideoMetadata':
        """Create VideoMetadata from HotClip checkpoint.file schema."""
        return cls(
            path=file_dict.get('path', ''),
            duration_sec=file_dict.get('durationSec', 0.0),
            width=file_dict.get('width', 0),
            height=file_dict.get('height', 0),
            fps=file_dict.get('fps', 0.0),
            video_codec=file_dict.get('videoCodec', 'unknown'),
            audio_codec=file_dict.get('audioCodec', 'unknown'),
            bit_rate=file_dict.get('bitRate'),
            pixel_format=file_dict.get('pixelFormat'),
            has_video=file_dict.get('hasVideo', True),
            has_audio=file_dict.get('hasAudio', True),
        )

    @classmethod
    def from_ffmpeg_probe(cls, probe_data: dict) -> 'VideoMetadata':
        """Create VideoMetadata from ffprobe JSON output."""
        video_stream = None
        audio_stream = None

        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = stream

        return cls(
            path=probe_data.get('filepath', ''),
            duration_sec=float(probe_data.get('format', {}).get('duration', 0)),
            width=video_stream.get('width', 0) if video_stream else 0,
            height=video_stream.get('height', 0) if video_stream else 0,
            fps=video_stream.get('r_frame_rate', '0/1').split('/')[-1] if video_stream else 0.0,
            video_codec=video_stream.get('codec_name', 'unknown') if video_stream else 'unknown',
            audio_codec=audio_stream.get('codec_name', 'unknown') if audio_stream else 'unknown',
            bit_rate=int(probe_data.get('format', {}).get('bit_rate', 0)) if probe_data.get('format', {}).get('bit_rate') else None,
            pixel_format=video_stream.get('pix_fmt') if video_stream else None,
            has_video=video_stream is not None,
            has_audio=audio_stream is not None,
        )


@dataclass
class Word:
    """Normalized word-level transcript data."""
    text: str
    start: float
    end: float
    timing_source: Optional[str] = None

    def to_hotclip(self) -> dict:
        """Convert to HotClip word format (camelCase keys)."""
        return {
            'text': self.text,
            'startSec': self.start,
            'endSec': self.end,
            'timingSource': self.timing_source or 'normalized'
        }

    def to_osc(self) -> dict:
        """Convert to opensource-clipping word format (PascalCase keys)."""
        return {
            'word': self.text,
            'start': self.start,
            'end': self.end
        }


@dataclass
class Segment:
    """Normalized transcript segment with word-level timing."""
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_hotclip(cls, seg_dict: dict) -> 'Segment':
        """Create from HotClip segment format."""
        words = [
            Word(
                text=w.get('text', ''),
                start=w.get('startSec', 0),
                end=w.get('endSec', 0),
                timing_source=w.get('timingSource')
            )
            for w in seg_dict.get('words', [])
        ]
        return cls(
            id=seg_dict.get('id', 0),
            start=seg_dict.get('startSec', 0),
            end=seg_dict.get('endSec', 0),
            text=seg_dict.get('text', ''),
            words=words
        )

    @classmethod
    def from_osc(cls, seg_dict: dict, idx: int = 0) -> 'Segment':
        """Create from opensource-clipping segment format."""
        words = [
            Word(
                text=w.get('word', ''),
                start=w.get('start', 0),
                end=w.get('end', 0)
            )
            for w in seg_dict.get('words', [])
        ]
        return cls(
            id=seg_dict.get('id', idx),
            start=seg_dict.get('start', 0),
            end=seg_dict.get('end', 0),
            text=seg_dict.get('text', seg_dict.get('words_text', '')),
            words=words
        )


@dataclass
class Transcript:
    """Normalized transcript containing segments and language info."""
    language: str
    engine: str
    segments: list[Segment]
    duration_sec: float = 0.0

    @classmethod
    def from_hotclip(cls, transcript_dict: dict) -> 'Transcript':
        """Create from HotClip transcript cache format."""
        segments = [
            Segment.from_hotclip(seg)
            for seg in transcript_dict.get('segments', [])
        ]
        duration = segments[-1].end if segments else 0
        return cls(
            language=transcript_dict.get('language', 'en'),
            engine=transcript_dict.get('engine', 'unknown'),
            segments=segments,
            duration_sec=duration
        )

    @classmethod
    def from_osc(cls, data_segmen: list, engine: str = 'faster-whisper', language: str = 'en') -> 'Transcript':
        """Create from opensource-clipping data_segmen format."""
        segments = [
            Segment.from_osc(seg, idx=i)
            for i, seg in enumerate(data_segmen)
        ]
        duration = segments[-1].end if segments else 0
        return cls(
            language=language,
            engine=engine,
            segments=segments,
            duration_sec=duration
        )

    def to_hotclip(self) -> dict:
        """Convert to HotClip transcript format."""
        return {
            'language': self.language,
            'engine': self.engine,
            'durationSec': self.duration_sec,
            'segments': [
                {
                    'id': seg.id,
                    'startSec': seg.start,
                    'endSec': seg.end,
                    'text': seg.text,
                    'words': [w.to_hotclip() for w in seg.words]
                }
                for seg in self.segments
            ]
        }

    def to_osc(self) -> list:
        """Convert to opensource-clipping data_segmen format."""
        return [
            {
                'start': seg.start,
                'end': seg.end,
                'text': seg.text,
                'words': [w.to_osc() for w in seg.words]
            }
            for seg in self.segments
        ]


@dataclass
class ClipCandidate:
    """Normalized clip candidate with title, hook, timing."""
    id: int
    start_sec: float
    end_sec: float
    title: str
    hook: str
    score: int
    score_dims: Optional[dict] = None
    keywords: Optional[list[str]] = None

    @classmethod
    def from_hotclip(cls, candidate: dict) -> 'ClipCandidate':
        """Create from HotClip candidate format."""
        return cls(
            id=candidate.get('id', 0),
            start_sec=candidate.get('startSec', 0),
            end_sec=candidate.get('endSec', 0),
            title=candidate.get('title', ''),
            hook=candidate.get('hook', ''),
            score=candidate.get('score', 0),
            score_dims=candidate.get('scoreDims'),
            keywords=candidate.get('keywords', [])
        )

    @classmethod
    def from_osc(cls, data: dict, idx: int = 0) -> 'ClipCandidate':
        """Create from opensource-clipping AI analysis format."""
        return cls(
            id=data.get('rank', idx),
            start_sec=data.get('start_time', 0),
            end_sec=data.get('end_time', 0),
            title=data.get('title_inggris', data.get('title_indonesia', '')),
            hook=data.get('description_hook', data.get('alasan', '')),
            score=data.get('viral_score', 0),
            keywords=data.get('keyword_tags', [])
        )


@dataclass
class VideoProject:
    """Normalized project containing video, transcript, and candidates."""
    video: VideoMetadata
    transcript: Transcript
    candidates: list[ClipCandidate] = field(default_factory=list)
    name: str = ''
    source_platform: str = 'unknown'
    source_url: Optional[str] = None

    def to_hotclip_json(self) -> dict:
        """Convert to HotClip project JSON format for saving."""
        candidates = [
            {
                'id': c.id,
                'startSec': c.start_sec,
                'endSec': c.end_sec,
                'title': c.title,
                'hook': c.hook,
                'score': c.score,
                'scoreDims': c.score_dims,
                'keywords': c.keywords,
                'recommended': True,
                'gate': 'publish'
            }
            for c in self.candidates
        ]

        return {
            'version': 1,
            'id': self._generate_id(),
            'name': self.name,
            'source': {
                'path': self.video.path,
                'size': 0,  # Would need original file stats
                'mtimeMs': 0  # Would need original file stats
            },
            'createdAt': self._iso_timestamp(),
            'updatedAt': self._iso_timestamp(),
            'lastOpenedAt': self._iso_timestamp(),
            'checkpoint': {
                'file': {
                    'path': self.video.path,
                    'durationSec': self.video.duration_sec,
                    'hasVideo': self.video.has_video,
                    'hasAudio': self.video.has_audio,
                    'width': self.video.width,
                    'height': self.video.height,
                    'fps': self.video.fps,
                    'videoCodec': self.video.video_codec,
                    'audioCodec': self.video.audio_codec,
                    'bitRate': self.video.bit_rate,
                    'pixelFormat': self.video.pixel_format
                },
                'transcript': self.transcript.to_hotclip(),
                'candidates': candidates,
                'selected': [c.id for c in self.candidates] if self.candidates else [],
                'focusedId': self.candidates[0].id if self.candidates else 0,
                'stats': {
                    'funnel': None,
                    'vision': None,
                    'emotion': None,
                    'danmaku': None,
                    'voice': None,
                    'reference': None,
                    'referenceError': None
                },
                'diarize': False,
                'referencePath': None,
                'paramsDirty': False,
                'savedAt': self._iso_timestamp(),
                'editHistory': []
            }
        }

    def to_osc_format(self) -> dict:
        """Convert to opensource-clipping compatible format."""
        return {
            'video_path': self.video.path,
            'transcript': self.transcript.to_osc(),
            'candidates': [
                {
                    'rank': c.id,
                    'title_inggris': c.title,
                    'description_hook': c.hook,
                    'start_time': c.start_sec,
                    'end_time': c.end_sec,
                    'viral_score': c.score,
                    'keyword_tags': c.keywords or []
                }
                for c in self.candidates
            ]
        }

    def _generate_id(self) -> str:
        """Generate a unique project ID."""
        import uuid
        return str(uuid.uuid4())

    def _iso_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# Default configuration values (mirroring opensource-clipping defaults)
DEFAULT_CONFIG = {
    'font': {
        'name': 'Montserrat',
        'size': 80,
        'outline': 3,
        'shadow': 2.5,
        'primary_color': '&H00FFFFFF',
        'outline_color': '&H00000000',
        'back_color': '&H80000000'
    },
    'captions': {
        'position': 950,  # Y position in pixels
        'align': 2,       # Bottom center
        'margin_v': 50,   # Vertical margin
        'margin_h': 20,   # Horizontal margin
        'karaoke_outline': 3,
        'basic_outline': 0.2
    },
    'video': {
        'quality_preset': 'medium',  # fast, medium, slow, slower, veryslow
        'codec': 'libx265',          # or 'libx264' for broader compatibility
        'crf': 28,                   # Quality (lower = better)
        'pix_fmt': 'yuv420p'
    }
}