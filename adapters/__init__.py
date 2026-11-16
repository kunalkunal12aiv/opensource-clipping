#!/usr/bin/env python3
"""
Video Post-Production Engine Adapters

Input adapters for importing projects from external clipping systems:
- HotClip: Electron-based clipping app (.hotclip files)
- opensource-clipping: Python-based AI auto-clipper

All adapters normalize data to the shared VideoProject format.
"""

from .hotclip_adapter import HotClipAdapter, load_hotclip_project
from .osc_adapter import OSClipAdapter, load_osc_output

__all__ = [
    'HotClipAdapter',
    'load_hotclip_project',
    'OSClipAdapter',
    'load_osc_output',
]
__version__ = '0.1.0'