#!/usr/bin/env python3
"""
OpenSource-Clipping Input Adapter

Reads opensource-clipping engine outputs and normalizes to the shared VideoProject format.

Key data files:
- gemini_response.json (AI analysis with clip suggestions, typography plan, B-roll)
- engine.transcribe_video() output (word-level transcript segments)
- render_manifest.json (rendered clip results)
- metadata_preview.json (platform metadata)
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

# Import from videopostengine core module
sys.path.insert(0, '/c/Users/user/videopostengine')
from core.config import VideoProject, VideoMetadata, Transcript, ClipCandidate


class OSClipAdapter:
    """
    Adapter for loading opensource-clipping outputs into the shared engine format.

    Supports loading from:
    1. Output directory (contains gemini_response.json, transcript data)
    2. Individual engine data_segmen list
    3. gemini_response.json with AI analysis
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize adapter with path to opensource-clipping output directory.

        Args:
            output_dir: Path to outputs/timestamp/ directory containing render artifacts
        """
        self.output_dir = Path(output_dir)
        self._gemini_data: Optional[list] = None
        self._transcript_data: Optional[list] = None

    def load(self) -> VideoProject:
        """
        Load and normalize the opensource-clipping project.

        Returns:
            VideoProject with normalized video, transcript, and candidates
        """
        # Load AI analysis data
        gemini_path = self.output_dir / 'gemini_response.json'
        if gemini_path.exists():
            self._gemini_data = self._load_json(gemini_path)

        # Load transcript (either from separate file or from engine output)
        self._transcript_data = self._load_transcript()

        # Build video metadata by probing source video
        video_path = self._find_source_video()
        video_meta = self._probe_video(video_path) if video_path else VideoMetadata(
            path='', duration_sec=0, width=0, height=0, fps=0,
            video_codec='unknown', audio_codec='unknown'
        )

        # Build transcript
        transcript = self._extract_transcript()

        # Build candidates from AI analysis
        candidates = self._extract_candidates()

        # Get project name from output directory
        project_name = self.output_dir.name

        return VideoProject(
            video=video_meta,
            transcript=transcript,
            candidates=candidates,
            name=project_name
        )

    def load_from_engine_data(self, data_segmen: list, ai_analysis: Optional[list] = None) -> VideoProject:
        """
        Load directly from engine data (useful for programmatic use).

        Args:
            data_segmen: Word-level transcript segments from engine.transcribe_video()
            ai_analysis: Optional AI analysis results from analyze_with_ai()

        Returns:
            VideoProject with normalized transcript
        """
        self._transcript_data = data_segmen
        self._gemini_data = ai_analysis or []

        return VideoProject(
            video=VideoMetadata(
                path='',
                duration_sec=0,
                width=0,
                height=0,
                fps=0,
                video_codec='unknown',
                audio_codec='unknown'
            ),
            transcript=self._extract_transcript(),
            candidates=self._extract_candidates(),
            name='engine-output'
        )

    def _load_json(self, path: Path) -> Union[list, dict]:
        """Load JSON file with error handling."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return [] if path.name == 'gemini_response.json' else {}

    def _load_transcript(self) -> list:
        """Load transcript data segments."""
        # Try to find transcript in expected locations
        transcript_paths = [
            self.output_dir / 'data_segmen.json',  # Custom export
            self.output_dir / 'transcript.json',    # Alternative
        ]

        # Also check combined_transcript.json in story cache
        if not any(p.exists() for p in transcript_paths):
            story_cache = self.output_dir.parent.parent / 'story_cache' / self.output_dir.name
            if story_cache.exists():
                transcript_paths.append(story_cache / 'combined_transcript.json')

        for path in transcript_paths:
            if path.exists():
                data = self._load_json(path)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and 'words' in data:
                    # Convert combined_transcript format
                    return self._convert_combined_transcript(data)

        return []

    def _convert_combined_transcript(self, data: dict) -> list:
        """Convert combined_transcript.json format to data_segmen format."""
        words = data.get('words', [])
        segments = []

        # Group words into segments (every ~20 words or by sentence boundary)
        if words:
            chunk_size = 20
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                start = chunk[0].get('start', 0)
                end = chunk[-1].get('end', 0)
                text = ''.join(w.get('word', '') for w in chunk)

                segments.append({
                    'start': start,
                    'end': end,
                    'text': text,
                    'words': chunk
                })

        return segments

    def _find_source_video(self) -> Optional[str]:
        """Find source video file in output directory."""
        video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.webm']

        for ext in video_extensions:
            potential = self.output_dir / f'video_asli{ext}'
            if potential.exists():
                return str(potential)

        # Check outputs directory one level up
        parent = self.output_dir.parent
        for ext in video_extensions:
            for video_file in parent.rglob(f'*{ext}'):
                if 'video_asli' in video_file.name:
                    return str(video_file)

        return None

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

    def _extract_transcript(self) -> Transcript:
        """Extract and normalize transcript from OSC format."""
        if not self._transcript_data:
            return Transcript(language='en', engine='faster-whisper', segments=[], duration_sec=0)

        segments = []
        for i, seg_dict in enumerate(self._transcript_data):
            segment = {
                'id': seg_dict.get('id', i),
                'start': seg_dict.get('start', 0),
                'end': seg_dict.get('end', 0),
                'text': seg_dict.get('text', ''),
                'words': []
            }

            # Convert OSC word format to normalized Word
            for word_dict in seg_dict.get('words', []):
                segment['words'].append({
                    'word': word_dict.get('word', ''),
                    'start': word_dict.get('start', 0),
                    'end': word_dict.get('end', 0)
                })

            segments.append(segment)

        # Calculate duration
        duration = max((s['end'] for s in segments), default=0)

        return Transcript.from_osc(segments)

    def _extract_candidates(self) -> list[ClipCandidate]:
        """Extract clip candidates from AI analysis."""
        if not self._gemini_data:
            return []

        candidates = []
        for item in self._gemini_data:
            candidate = ClipCandidate.from_osc(item, idx=item.get('rank', 0))
            candidates.append(candidate)

        return candidates

    def get_ai_analysis(self) -> Optional[dict]:
        """Get the full AI analysis response for access to typography plan, B-roll, etc."""
        if self._gemini_data:
            return self._gemini_data[0] if isinstance(self._gemini_data, list) else self._gemini_data
        return None

    def get_typography_plan(self) -> list[dict]:
        """Extract typography plan from AI analysis for motion graphics."""
        ai_data = self.get_ai_analysis()
        if ai_data:
            return ai_data.get('typography_plan', [])
        return []

    def get_broll_suggestions(self) -> list[dict]:
        """Extract B-roll suggestions from AI analysis."""
        ai_data = self.get_ai_analysis()
        if ai_data:
            return ai_data.get('broll_list', [])
        return []

    def get_rendered_clips(self) -> list[dict]:
        """Load render manifest for status of completed clips."""
        manifest_path = self.output_dir / 'render_manifest.json'
        if manifest_path.exists():
            return self._load_json(manifest_path)
        return []

    def get_platform_metadata(self) -> dict:
        """Load platform-specific metadata (TikTok, YouTube, etc.)."""
        metadata_path = self.output_dir / 'metadata_preview.json'
        if metadata_path.exists():
            return self._load_json(metadata_path)
        return {}


def load_osc_output(output_dir: str) -> VideoProject:
    """
    Convenience function to load an opensource-clipping output directory.

    Args:
        output_dir: Path to outputs/timestamp/ directory

    Returns:
        Normalized VideoProject
    """
    adapter = OSClipAdapter(output_dir)
    return adapter.load()


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python osc_adapter.py <outputs/timestamp/>")
        sys.exit(1)

    project = load_osc_output(sys.argv[1])

    print(f"Project: {project.name}")
    print(f"Video: {project.video.path}")
    print(f"Duration: {project.video.duration_sec:.2f}s")
    print(f"Transcript segments: {len(project.transcript.segments)}")
    print(f"Candidates: {len(project.candidates)}")

    for i, c in enumerate(project.candidates[:3]):
        print(f"  [{i+1}] {c.title} - Score: {c.score}")