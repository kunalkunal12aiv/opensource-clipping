"""
clipping.story.assembler — Scene Trim, Concat & Render for Story Clip

Handles:
  - Trimming individual scenes from cached source videos (FFmpeg)
  - Concatenating scenes with transitions
  - Rendering hook and highlight outputs separately
"""

import os
import subprocess
import json
import math


# ==============================================================================
# FFMPEG HELPERS
# ==============================================================================

def _run_ffmpeg(cmd: list[str], label: str = "") -> None:
    """Run an FFmpeg command, raise on failure."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(
            f"FFmpeg gagal{' (' + label + ')' if label else ''}: {stderr_text[:500]}"
        ) from e


def trim_scene(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    reencode: bool = False,
) -> str:
    """
    Trim a segment from a video file using FFmpeg.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    output_path : str
        Path for the trimmed output.
    reencode : bool
        If True, re-encode for frame-accurate cuts. If False, use
        stream copy for speed (may have slight inaccuracy at GOP boundaries).

    Returns
    -------
    str
        Path to the trimmed output file.
    """
    duration = end - start

    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

    _run_ffmpeg(cmd, label=f"trim {start:.1f}-{end:.1f}")
    return output_path


# ==============================================================================
# SCENE NORMALIZATION (for concat compatibility)
# ==============================================================================

def _normalize_scene_segment(
    input_path: str,
    output_path: str,
    target_width: int = 1080,
    target_height: int = 1920,
    target_fps: int = 30,
    crop_to_fill: bool = False,
) -> str:
    """
    Re-encode a scene segment to a consistent format so all segments can be
    safely concatenated (same resolution, frame rate, pixel format, codec).

    Parameters
    ----------
    input_path : str
        Path to the trimmed scene segment.
    output_path : str
        Path for the normalized output.
    target_width, target_height : int
        Target resolution. Scenes are scaled + padded (letterbox/pillarbox)
        to fit without stretching.
    target_fps : int
        Target frame rate.

    Returns
    -------
    str
        Path to the normalized output file.
    """
    if crop_to_fill:
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height}:(iw-ow)/2:(ih-oh)/2,"
            f"fps={target_fps},format=yuv420p"
        )
    else:
        # Preserve the existing fit-and-pad behavior for tracked scenes and
        # layouts where the full source frame should remain visible.
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={target_fps},format=yuv420p"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-shortest",
        output_path,
    ]

    _run_ffmpeg(cmd, label="normalize")
    return output_path


# ==============================================================================
# CONCAT SCENES
# ==============================================================================

def concat_scenes(
    scene_paths: list[str],
    output_path: str,
    transition: str = "cut",
    fade_duration: float = 0.5,
    scene_durations: list[float] | None = None,
) -> str:
    """
    Concatenate multiple scene segments into a single video.

    Parameters
    ----------
    scene_paths : list[str]
        Ordered list of normalized scene file paths.
    output_path : str
        Path for the concatenated output.
    transition : str
        Transition type between scenes. Currently supports:
        ``"cut"`` (hard cut) and ``"crossfade"`` (dissolve).
    fade_duration : float
        Audio/video crossfade duration in seconds.
    scene_durations : list[float] | None
        Optional durations corresponding to ``scene_paths``. Used to place
        crossfades at clip boundaries without probing each rendered segment.
        More transitions can be added later.

    Returns
    -------
    str
        Path to the concatenated output.
    """
    if not scene_paths:
        raise ValueError("Tidak ada scene untuk di-concat.")

    if len(scene_paths) == 1:
        # Single scene, just copy
        import shutil
        shutil.copy2(scene_paths[0], output_path)
        return output_path

    if transition == "crossfade":
        return _concat_with_crossfade(
            scene_paths, output_path, fade_duration, scene_durations
        )

    # Default: hard cut using concat demuxer
    return _concat_hard_cut(scene_paths, output_path)


def _concat_hard_cut(scene_paths: list[str], output_path: str) -> str:
    """Concatenate using FFmpeg concat demuxer (lossless for same-format files)."""
    # Write concat list file
    list_path = output_path + ".concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in scene_paths:
            # FFmpeg concat demuxer requires forward slashes and escaped quotes
            safe_path = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]

    _run_ffmpeg(cmd, label="concat_hard_cut")

    # Clean up list file
    if os.path.exists(list_path):
        os.remove(list_path)

    return output_path


def _concat_with_crossfade(
    scene_paths: list[str],
    output_path: str,
    fade_duration: float = 0.5,
    scene_durations: list[float] | None = None,
) -> str:
    """
    Concatenate with crossfade transitions between scenes using xfade filter.
    All inputs must already be normalized to the same resolution/fps.
    """
    if not math.isfinite(fade_duration) or fade_duration <= 0:
        raise ValueError("Crossfade duration must be greater than zero.")
    if scene_durations is not None and len(scene_durations) != len(scene_paths):
        raise ValueError("scene_durations must match the number of scene_paths.")
    if scene_durations is None:
        scene_durations = [
            _probe_media_duration(path) for path in scene_paths
        ]
    if any(
        not math.isfinite(duration) or duration <= fade_duration
        for duration in scene_durations
    ):
        raise ValueError("Each scene must be longer than the crossfade duration.")

    def make_command(
        left_path: str,
        right_path: str,
        offset: float,
        output: str,
        label: str,
    ) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", left_path,
            "-i", right_path,
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={fade_duration:.3f}:"
            f"offset={offset:.3f}[v];"
            f"[0:a][1:a]acrossfade=d={fade_duration:.3f}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac",
            output,
        ]
        _run_ffmpeg(cmd, label=label)

    if len(scene_paths) == 2:
        # Simple case: 2 inputs with 1 xfade
        make_command(
            scene_paths[0],
            scene_paths[1],
            scene_durations[0] - fade_duration,
            output_path,
            "crossfade",
        )
        return output_path

    # For 3+ scenes, chain each transition at the accumulated output boundary.
    import shutil
    temp_dir = output_path + "_xfade_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        current = scene_paths[0]
        current_duration = scene_durations[0]
        for i in range(1, len(scene_paths)):
            temp_out = os.path.join(temp_dir, f"xfade_{i}.mp4")
            make_command(
                current,
                scene_paths[i],
                current_duration - fade_duration,
                temp_out,
                f"crossfade_{i}",
            )
            current_duration += scene_durations[i] - fade_duration
            current = temp_out

        shutil.copy2(current, output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return output_path


def _probe_media_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Unable to read media duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Media duration must be positive for {path}")
    return duration


# ==============================================================================
# HOOK TEXT OVERLAY
# ==============================================================================

def _add_text_overlay(
    video_path: str,
    text: str,
    output_path: str,
    font_size: int = 48,
    font_color: str = "white",
    box_color: str = "black@0.5",
    position: str = "center",
) -> str:
    """
    Add a text overlay on top of a video using FFmpeg drawtext filter.

    Parameters
    ----------
    video_path : str
        Input video path.
    text : str
        Text to overlay.
    output_path : str
        Output video path.
    position : str
        One of ``"center"``, ``"bottom"``, ``"top"``.
    """
    # Escape special chars for FFmpeg drawtext
    escaped = text.replace("'", "'\\''").replace(":", "\\:")

    if position == "center":
        xy = "x=(w-text_w)/2:y=(h-text_h)/2"
    elif position == "bottom":
        xy = "x=(w-text_w)/2:y=h-text_h-60"
    else:
        xy = "x=(w-text_w)/2:y=60"

    vf = (
        f"drawtext=text='{escaped}':"
        f"fontsize={font_size}:fontcolor={font_color}:"
        f"borderw=3:bordercolor=black:"
        f"box=1:boxcolor={box_color}:boxborderw=10:"
        f"{xy}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]

    _run_ffmpeg(cmd, label="text_overlay")
    return output_path


# ==============================================================================
# MAIN ASSEMBLY FUNCTIONS
# ==============================================================================

def _get_target_dims(ratio: str) -> tuple[int, int]:
    """Return (width, height) for a given aspect ratio string."""
    ratio_map = {
        "9:16": (1080, 1920),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
        "3:4": (1080, 1440),
        "4:5": (1080, 1350),
    }
    return ratio_map.get(ratio, (1080, 1920))


def assemble_hook(
    clip_config: dict,
    source_registry: dict[str, dict],
    cache_dir: str,
    output_dir: str,
    ratio: str = "9:16",
) -> str | None:
    """
    Assemble the hook segment for a single clip.

    Parameters
    ----------
    clip_config : dict
        A single clip entry from the recipe.
    source_registry : dict
        Source registry from loader.
    cache_dir : str
        Path to the source cache directory.
    output_dir : str
        Directory to write the output hook file.
    ratio : str
        Target aspect ratio.

    Returns
    -------
    str or None
        Path to the rendered hook video, or None if assembly failed.
    """
    cid = clip_config["clip_id"]
    hook = clip_config["hook"]
    scenes = hook.get("scenes", [])

    print(f"\n   🎣 Assembling hook_{cid}...")

    target_w, target_h = _get_target_dims(ratio)
    temp_dir = os.path.join(output_dir, f"_temp_hook_{cid}")
    os.makedirs(temp_dir, exist_ok=True)

    normalized_parts: list[str] = []

    for idx, scene in enumerate(scenes):
        sid = scene["source_id"]
        start = scene.get("start")
        end = scene.get("end")

        if start is None or end is None:
            print(f"      ⏩ Scene #{idx} ({sid}): timestamp null, skip.")
            continue

        # Resolve source path
        src = source_registry[sid]
        if src["platform"] == "local":
            video_path = src["local_path"]
        else:
            video_path = os.path.join(cache_dir, f"{sid}.mp4")

        if not os.path.exists(video_path):
            print(f"      ⚠️ Scene #{idx} ({sid}): file tidak ditemukan, skip.")
            continue

        # Trim
        trimmed = os.path.join(temp_dir, f"hook_{cid}_scene_{idx}_trim.mp4")
        trim_scene(video_path, start, end, trimmed, reencode=True)

        # Normalize
        normed = os.path.join(temp_dir, f"hook_{cid}_scene_{idx}_norm.mp4")
        _normalize_scene_segment(trimmed, normed, target_w, target_h)
        normalized_parts.append(normed)

    if not normalized_parts:
        print(f"      ❌ Hook #{cid}: tidak ada scene yang valid.")
        return None

    # Concat — hooks are always clean video (no text overlay, no subs)
    output_path = os.path.join(output_dir, f"hook_{cid}.mp4")
    concat_scenes(normalized_parts, output_path, transition="cut")

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"      ✅ hook_{cid}.mp4 berhasil dirender.")
    return output_path


def assemble_highlight(
    clip_config: dict,
    source_registry: dict[str, dict],
    cache_dir: str,
    output_dir: str,
    ratio: str = "9:16",
    transition_duration: float = 0.5,
    cfg=None,
) -> str | None:
    """
    Assemble the highlight segment for a single clip.

    Parameters
    ----------
    clip_config : dict
        A single clip entry from the recipe.
    source_registry : dict
        Source registry from loader.
    cache_dir : str
        Path to the source cache directory.
    output_dir : str
        Directory to write the output highlight file.
    ratio : str
        Target aspect ratio.

    Returns
    -------
    str or None
        Path to the rendered highlight video, or None if assembly failed.
    """
    cid = clip_config["clip_id"]
    highlight = clip_config["highlight"]
    scenes = highlight.get("scenes", [])
    transition = highlight.get("transition", "cut")

    # Map user-friendly names to internal transition names
    transition_map = {
        "smooth": "crossfade",
        "crossfade": "crossfade",
        "jedag_jedug": "cut",  # TODO: implement jedag-jedug transition
        "cut": "cut",
    }
    transition_type = transition_map.get(transition, "cut")

    print(f"\n   🎬 Assembling highlight_{cid}...")

    target_w, target_h = _get_target_dims(ratio)
    temp_dir = os.path.join(output_dir, f"_temp_highlight_{cid}")
    os.makedirs(temp_dir, exist_ok=True)

    normalized_parts: list[str] = []
    scene_durations: list[float] = []

    for idx, scene in enumerate(scenes):
        sid = scene["source_id"]
        start = scene.get("start")
        end = scene.get("end")
        label = scene.get("label", "")

        if start is None or end is None:
            print(f"      ⏩ Scene #{idx} ({sid}): timestamp null, skip. [{label}]")
            continue

        # Resolve source path
        src = source_registry[sid]
        if src["platform"] == "local":
            video_path = src["local_path"]
        else:
            video_path = os.path.join(cache_dir, f"{sid}.mp4")

        if not os.path.exists(video_path):
            print(f"      ⚠️ Scene #{idx} ({sid}): file tidak ditemukan, skip.")
            continue

        duration = end - start

        # Reuse the standard clipping face tracker for vertical Auto Story
        # scenes. Wide ratios retain the standard fit-to-frame behavior.
        use_subject_tracking = bool(
            cfg is not None
            and ratio in {"9:16", "1:1", "3:4", "4:5"}
            and not getattr(cfg, "static_crop", False)
        )
        center_crop_fallback = ratio in {"9:16", "1:1", "3:4", "4:5"}
        if use_subject_tracking:
            try:
                from clipping import studio

                tracked_silent = os.path.join(
                    temp_dir, f"hl_{cid}_scene_{idx}_tracked_silent.mp4"
                )
                tracked_with_audio = os.path.join(
                    temp_dir, f"hl_{cid}_scene_{idx}_tracked.mp4"
                )
                studio.buat_video_hybrid(
                    video_path,
                    tracked_silent,
                    start,
                    end,
                    ratio,
                    cfg,
                    label=f"Auto Story scene {idx + 1}",
                )
                if not os.path.isfile(tracked_silent) or os.path.getsize(tracked_silent) == 0:
                    raise RuntimeError("Subject tracker did not produce a video scene.")
                _run_ffmpeg(
                    [
                        "ffmpeg", "-y", "-i", tracked_silent,
                        "-ss", f"{start:.3f}", "-i", video_path,
                        "-t", f"{duration:.3f}",
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                        "-shortest", tracked_with_audio,
                    ],
                    label=f"mux tracked Auto Story scene {idx + 1} audio",
                )
                if not os.path.isfile(tracked_with_audio) or os.path.getsize(tracked_with_audio) == 0:
                    raise RuntimeError("Tracked Auto Story scene did not preserve source audio.")
                trimmed = tracked_with_audio
                center_crop_fallback = False
            except Exception as exc:
                print(
                    f"      Subject tracking failed for scene {idx + 1}; "
                    f"using center framing: {type(exc).__name__}: {exc}"
                )
                trimmed = os.path.join(temp_dir, f"hl_{cid}_scene_{idx}_trim.mp4")
                trim_scene(video_path, start, end, trimmed, reencode=True)
        else:
            # Center-framed fallback for landscape output and explicit static crop.
            trimmed = os.path.join(temp_dir, f"hl_{cid}_scene_{idx}_trim.mp4")
            trim_scene(video_path, start, end, trimmed, reencode=True)

        # Normalize
        normed = os.path.join(temp_dir, f"hl_{cid}_scene_{idx}_norm.mp4")
        _normalize_scene_segment(
            trimmed,
            normed,
            target_w,
            target_h,
            crop_to_fill=center_crop_fallback,
        )
        normalized_parts.append(normed)
        scene_durations.append(duration)

        print(f"      ✅ Scene #{idx}: {sid} [{start:.1f}s - {end:.1f}s] ({duration:.1f}s) — {label}")

    if not normalized_parts:
        print(f"      ❌ Highlight #{cid}: tidak ada scene yang valid.")
        return None

    # Concat with transition
    output_path = os.path.join(output_dir, f"highlight_{cid}.mp4")
    concat_scenes(
        normalized_parts,
        output_path,
        transition=transition_type,
        fade_duration=transition_duration,
        scene_durations=scene_durations,
    )

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    total_scenes = len(normalized_parts)
    print(f"      ✅ highlight_{cid}.mp4 berhasil dirender ({total_scenes} scene, transition: {transition}).")
    return output_path
