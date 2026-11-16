#!/usr/bin/env python3
"""
HotClip Input Adapter

Reads HotClip .hotclip project files and transcript cache files,
normalizes to the shared VideoProject format.

HotClip stores:
- Project metadata (name, source video) in .hotclip JSON files
- Transcript data (word-level timestamps) in checkpoint.transcript or transcript-cache/*.json
- AI clip candidates in checkpoint.candidates
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

# Import from videopostengine core module
sys.path.insert(0, '/c/Users/user/videopostengine')
from core.config import VideoProject, VideoMetadata, Transcript, ClipCandidate


class HotClipAdapter:
    """
    Adapter for loading HotClip project files into the shared engine format.

    Supports two input types:
    1. Full .hotclip project files (recommended) - contains video metadata, transcript, and candidates
    2. Transcript cache files (.hotclip-transcript.json) - only transcript data
    """

    def __init__(self, project_path: Union[str, Path]):
        """
        Initialize adapter with path to HotClip project or transcript cache.

        Args:
            project_path: Path to .hotclip file or transcript-cache JSON file
        """
        self.project_path = Path(project_path)
        self._hotclip_data: Optional[dict] = None
        self._transcript_cache_path: Optional[Path] = None

    def load(self) -> VideoProject:
        """
        Load and normalize the HotClip project.

        Returns:
            VideoProject with normalized video, transcript, and candidates

        Raises:
            FileNotFoundError: If project file doesn't exist
            ValueError: If file format is invalid
        """
        # Load project file if it's a .hotclip file
        if self.project_path.suffix == '.hotclip' or self.project_path.name.endswith('.hotclip'):
            self._hotclip_data = self._load_json(self.project_path)
            source_video = self._hotclip_data.get('source', {}).get('path', '')
            transcript_data = self._hotclip_data.get('checkpoint', {}).get('transcript', {})
            candidates_data = self._hotclip_data.get('checkpoint', {}).get('candidates', [])
            project_name = self._hotclip_data.get('name', '')
        else:
            # Assume it's a transcript cache file
            self._transcript_cache_path = self.project_path
            self._hotclip_data = self._load_json(self.project_path)
            transcript_data = self._hotclip_data
            source_video = ''
            candidates_data = []
            project_name = self._transcript_cache_path.stem.replace('-transcript', '')

        # Build video metadata
        video_path = self._resolve_video_path(source_video)
        video_meta = self._extract_video_metadata(video_path)

        # Build transcript
        transcript = self._extract_transcript(transcript_data)

        # Extract transcript cache if video has no metadata in main file
        if not video_meta.path:
            transcript_cache_path = self._find_transcript_cache()
            if transcript_cache_path:
                cache_data = self._load_json(transcript_cache_path)
                transcript = self._extract_transcript(cache_data)

        # Build candidates
        candidates = [
            ClipCandidate.from_hotclip(c)
            for c in candidates_data
        ]

        return VideoProject(
            video=video_meta,
            transcript=transcript,
            candidates=candidates,
            name=project_name
        )

    def _load_json(self, path: Path) -> dict:
        """Load JSON file with error handling."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"HotClip file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in HotClip file {path}: {e}")

    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata from HotClip checkpoint.file or ffprobe."""
        if not self._hotclip_data:
            return VideoMetadata(
                path=video_path,
                duration_sec=0,
                width=0,
                height=0,
                fps=0.0,
                video_codec='unknown',
                audio_codec='unknown'
            )

        file_info = self._hotclip_data.get('checkpoint', {}).get('file', {})

        if file_info:
            # Use HotClip's stored metadata
            return VideoMetadata.from_hotclip(file_info)

        # Fallback to ffprobe if file exists
        if os.path.exists(video_path):
            return self._probe_video(video_path)

        return VideoMetadata(
            path=video_path,
            duration_sec=0,
            width=0,
            height=0,
            fps=0,
            video_codec='unknown',
            audio_codec='unknown'
        )

    def _probe_video(self, video_path: str) -> VideoMetadata:
        """Probe video file using ffprobe."""
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path],
                capture_output=True,
                text=True,
                check=True
            )
            probe_data = json.loads(result.stdout)
            return VideoMetadata.from_ffmpeg_probe(probe_data)
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            return VideoMetadata(
                path=video_path,
                duration_sec=0,
                width=0,
                height=0,
                fps=0,
                video_codec='unknown',
                audio_codec='unknown'
            )

    def _extract_transcript(self, transcript_data: dict) -> Transcript:
        """Extract and normalize transcript from HotClip format."""
        return Transcript.from_hotclip(transcript_data)

    def _resolve_video_path(self, video_path: str) -> str:
        """Resolve video path (handle Windows paths)."""
        if not video_path:
            return ''

        # Handle Windows-style paths with backslashes
        video_path = video_path.replace('\\\\', '\\')

        if os.path.exists(video_path):
            return video_path

        # Try to find video in HotClip source directories
        hotclip_sources = [
            Path.home() / 'Videos' / 'HotClip',
            Path('C:/Users/user/Videos/HotClip'),
        ]

        video_name = Path(video_path).name
        for source_dir in hotclip_sources:
            potential = source_dir / video_name
            if potential.exists():
                return str(potential)

        return video_path

    def _find_transcript_cache(self) -> Optional[Path]:
        """Find transcript cache file for current project."""
        if not self._hotclip_data:
            return None

        project_id = self._hotclip_data.get('id')
        if not project_id:
            return None

        # Look for transcript cache in HotClip cache directory
        cache_dir = Path.home() / 'AppData' / 'Roaming' / 'hotclip' / 'transcript-cache'
        if cache_dir.exists():
            for cache_file in cache_dir.glob('*.json'):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    if cache_data.get('id') == project_id:
                        return cache_file
                except (json.JSONDecodeError, IOError):
                    continue

        return None

    def get_candidates_by_score(self, reverse: bool = True) -> list[ClipCandidate]:
        """Get clip candidates sorted by score."""
        if not self._hotclip_data:
            return []

        candidates_data = self._hotclip_data.get('checkpoint', {}).get('candidates', [])
        candidates = [ClipCandidate.from_hotclip(c) for c in candidates_data]
        return sorted(candidates, key=lambda c: c.score, reverse=reverse)

    def get_selected_candidates(self) -> list[ClipCandidate]:
        """Get only the user-selected clip candidates."""
        if not self._hotclip_data:
            return []

        selected_ids = self._hotclip_data.get('checkpoint', {}).get('selected', [])
        candidates_data = self._hotclip_data.get('checkpoint', {}).get('candidates', [])

        selected = [
            ClipCandidate.from_hotclip(c)
            for c in candidates_data
            if c.get('id') in selected_ids
        ]
        return sorted(selected, key=lambda c: selected_ids.index(c.id) if c.id in selected_ids else len(selected_ids))


def load_hotclip_project(project_path: str) -> VideoProject:
    """
    Convenience function to load a HotClip project.

    Args:
        project_path: Path to .hotclip file

    Returns:
        Normalized VideoProject
    """
    adapter = HotClipAdapter(project_path)
    return adapter.load()


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hotclip_adapter.py <project.hotclip>")
        sys.exit(1)

    project = load_hotclip_project(sys.argv[1])

    print(f"Project: {project.name}")
    print(f"Video: {project.video.path}")
    print(f"Duration: {project.video.duration_sec:.2f}s")
    print(f"Transcript segments: {len(project.transcript.segments)}")
    print(f"Candidates: {len(project.candidates)}")

    for i, c in enumerate(project.candidates[:3]):
        print(f"  [{i+1}] {c.title} - Score: {c.score}")