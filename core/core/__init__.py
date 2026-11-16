#!/usr/bin/env python3
"""
Video Post-Production Engine - Core

Core data models and configuration for the shared video post-production engine.
"""

from .config import (
    VideoMetadata,
    Word,
    Segment,
    Transcript,
    ClipCandidate,
    VideoProject,
    DEFAULT_CONFIG,
)

__all__ = [
    'VideoMetadata',
    'Word',
    'Segment',
    'Transcript',
    'ClipCandidate',
    'VideoProject',
    'DEFAULT_CONFIG',
]
__version__ = '0.1.0'