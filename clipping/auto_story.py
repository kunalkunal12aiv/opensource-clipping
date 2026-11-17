"""Guided, cached Auto Story-Recipe workflow."""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

PROMPT_VERSION = "auto-story-recipe-v2-multi-story-planning"
BEAT_PROMPT_VERSION = "auto-story-beats-v3-retry-candidates"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
AI_TIMEOUT_SECONDS = 900
AI_REQUEST_MAX_ATTEMPTS = 3
WORD_BOUNDARY_TOLERANCE = 0.05


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp",
        dir=os.path.dirname(path), delete=False,
    )
    try:
        with temp:
            json.dump(data, temp, ensure_ascii=False, indent=2)
            temp.write("\n")
        os.replace(temp.name, path)
    finally:
        if os.path.exists(temp.name):
            os.remove(temp.name)


def _read_cache(path: str, cache_key: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        cached = json.load(file)
    if not isinstance(cached, dict):
        raise ValueError(f"Invalid cache file (expected JSON object): {path}")
    return cached if cached.get("_cache_key") == cache_key else None


def _write_cache(path: str, cache_key: str, payload: dict) -> None:
    _write_json(path, {"_cache_key": cache_key, **payload})


def _migrate_legacy_transcript_cache(
    video_path: str,
    cache_dir: str,
    source_signature: dict,
    cfg,
    transcript_key: str,
    diarization_key: str,
) -> tuple[dict | None, dict | None]:
    if os.path.basename(os.path.dirname(cache_dir)) != "auto_story":
        return None, None

    cache_root = os.path.dirname(cache_dir)
    upload_dir = os.path.dirname(os.path.abspath(video_path))
    video_size = os.path.getsize(video_path)
    migrated_transcript = None
    migrated_diarization = None

    for filename in os.listdir(upload_dir):
        candidate_path = os.path.join(upload_dir, filename)
        if (
            not os.path.isfile(candidate_path)
            or os.path.getsize(candidate_path) != video_size
        ):
            continue

        old_source_id = "episode-" + hashlib.sha256(
            filename.encode("utf-8")
        ).hexdigest()[:12]
        old_cache_dir = os.path.join(cache_root, _safe_source_name(old_source_id))
        old_transcript_path = os.path.join(old_cache_dir, "transcript_words.json")
        old_diarization_path = os.path.join(old_cache_dir, "speaker_turns.json")
        if (
            not os.path.isfile(old_transcript_path)
            and not os.path.isfile(old_diarization_path)
        ):
            continue

        candidate_signature = _file_signature(candidate_path)
        if candidate_signature != source_signature:
            continue

        stat = os.stat(candidate_path)
        legacy_signature = {
            "path": os.path.normcase(os.path.abspath(candidate_path)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        old_transcript_key = _digest({
            "schema": 1,
            "source": legacy_signature,
            "whisper": {
                "model": cfg.auto_story_whisper_model,
                "device": cfg.auto_story_whisper_device,
                "compute_type": cfg.auto_story_whisper_compute_type,
                "max_words_per_subtitle": cfg.max_kata_per_subtitle,
            },
        })
        old_diarization_key = _digest({
            "schema": 1,
            "source": legacy_signature,
            "model": DIARIZATION_MODEL,
        })
        if migrated_transcript is None and os.path.isfile(old_transcript_path):
            migrated_transcript = _read_cache(
                old_transcript_path, old_transcript_key
            )
        if migrated_diarization is None and os.path.isfile(old_diarization_path):
            migrated_diarization = _read_cache(
                old_diarization_path, old_diarization_key
            )
        if migrated_transcript is not None and migrated_diarization is not None:
            break

    if migrated_transcript is not None:
        _write_cache(
            os.path.join(cache_dir, "transcript_words.json"),
            transcript_key,
            {
                key: value
                for key, value in migrated_transcript.items()
                if key != "_cache_key"
            },
        )
    if migrated_diarization is not None:
        _write_cache(
            os.path.join(cache_dir, "speaker_turns.json"),
            diarization_key,
            {
                key: value
                for key, value in migrated_diarization.items()
                if key != "_cache_key"
            },
        )
    if migrated_transcript is not None or migrated_diarization is not None:
        print("Migrated saved Auto Story transcript/diarization cache for this video.")
    return migrated_transcript, migrated_diarization


def _safe_source_name(source_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._")
    return slug[:60] or "episode"


def _file_signature(path: str) -> dict:
    before = os.stat(path)
    content_hash = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            content_hash.update(chunk)
    after = os.stat(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("Episode changed while preparing its transcript cache.")
    return {
        "content_sha256": content_hash.hexdigest(),
        "size": before.st_size,
    }


def _probe_duration(path: str) -> float:
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
        raise RuntimeError(f"ffprobe returned an invalid duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned a non-positive duration for {path}")
    return duration


def _flatten_words(segments: list[dict]) -> list[dict]:
    words = []
    skipped_zero_duration = 0
    for segment in segments:
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = float(word["start"])
            end = float(word["end"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < start
            ):
                raise ValueError(f"Invalid word timestamps: {word!r}")
            if end == start:
                skipped_zero_duration += 1
                continue
            words.append({"word": text, "start": start, "end": end})

    if skipped_zero_duration:
        print(
            "Warning: Skipped "
            f"{skipped_zero_duration} word(s) with zero-duration timestamps."
        )

    words.sort(key=lambda word: (word["start"], word["end"]))
    for index, word in enumerate(words, 1):
        word["word_id"] = f"W{index:06d}"
    if not words:
        raise ValueError("Transcription produced no word-level timestamps.")
    return words


def _assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    result = []
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        matching = [
            turn for turn in turns
            if float(turn["start"]) <= midpoint <= float(turn["end"])
        ]
        speaker_id = matching[0]["speaker"] if matching else None
        result.append({**word, "speaker_id": speaker_id})
    return result


def _format_transcript(
    words: list[dict], speaker_names: dict[str, str] | None = None
) -> str:
    speaker_names = speaker_names or {}
    return "\n".join(
        f"{word['word_id']}|{word['start']:.3f}-{word['end']:.3f}|"
        f"{speaker_names.get(word.get('speaker_id'), word.get('speaker_id')) or 'UNKNOWN'}|{word['word']}"
        for word in words
    )


def _load_or_create_transcript(
    video_path: str,
    source_id: str,
    cache_dir: str,
    cfg,
    duration: float,
    *,
    source_signature: dict | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
    phase_callback: Callable[[str], None] | None = None,
    cache_callback: Callable[[dict[str, str]], None] | None = None,
) -> tuple[list[dict], str]:
    os.makedirs(cache_dir, exist_ok=True)
    source_signature = source_signature or _file_signature(video_path)
    whisper_settings = {
        "model": cfg.auto_story_whisper_model,
        "device": cfg.auto_story_whisper_device,
        "compute_type": cfg.auto_story_whisper_compute_type,
        "max_words_per_subtitle": cfg.max_kata_per_subtitle,
    }
    transcript_key = _digest({
        "schema": 1,
        "source": source_signature,
        "whisper": whisper_settings,
    })
    transcript_path = os.path.join(cache_dir, "transcript_words.json")
    transcript_cache = _read_cache(transcript_path, transcript_key)
    cache_usage = {
        "transcript": "reused" if transcript_cache is not None else "missing",
        "diarization": "missing",
        "combined": "missing",
    }

    diarization_key = _digest({
        "schema": 1,
        "source": source_signature,
        "model": DIARIZATION_MODEL,
        "num_speakers": getattr(cfg, "diarization_num_speakers", "auto"),
    })
    diarization_path = os.path.join(cache_dir, "speaker_turns.json")
    diarization_cache = _read_cache(diarization_path, diarization_key)
    cache_usage["diarization"] = "reused" if diarization_cache is not None else "missing"

    combined_key = _digest({
        "schema": 1,
        "transcript": transcript_key,
        "diarization": diarization_key,
    })
    combined_path = os.path.join(cache_dir, "combined_transcript.json")
    combined_cache = _read_cache(combined_path, combined_key)
    if combined_cache is not None:
        cache_usage["combined"] = "reused"
        if cache_callback is not None:
            cache_callback(dict(cache_usage))
        return combined_cache["words"], combined_key

    if transcript_cache is None or diarization_cache is None:
        migrated_transcript, migrated_diarization = _migrate_legacy_transcript_cache(
            video_path,
            cache_dir,
            source_signature,
            cfg,
            transcript_key,
            diarization_key,
        )
        if transcript_cache is None:
            transcript_cache = migrated_transcript
            if transcript_cache is not None:
                cache_usage["transcript"] = "migrated"
        if diarization_cache is None:
            diarization_cache = migrated_diarization
            if diarization_cache is not None:
                cache_usage["diarization"] = "migrated"

    if transcript_cache is not None and diarization_cache is not None:
        words = _assign_speakers(
            transcript_cache["words"], diarization_cache["speaker_turns"]
        )
        _write_cache(combined_path, combined_key, {"words": words})
        cache_usage["combined"] = "created"
        if cache_callback is not None:
            cache_callback(dict(cache_usage))
        return words, combined_key

    if cache_callback is not None:
        cache_callback(dict(cache_usage))

    if diarization_cache is None and not cfg.hf_token:
        raise RuntimeError(
            "Auto Story requires HF_TOKEN and access to "
            f"{DIARIZATION_MODEL}. Accept the model terms on Hugging Face, "
            "then set HF_TOKEN in .env. Existing valid caches can be reused "
            "without repeating these model steps."
        )

    if transcript_cache is None:
        from . import engine

        print(
            f"Transcribing episode with Faster-Whisper "
            f"{cfg.auto_story_whisper_model} ({cfg.auto_story_whisper_device}/"
            f"{cfg.auto_story_whisper_compute_type})..."
        )
        _, segments = engine.transcribe_video(
            video_path,
            max_words_per_subtitle=cfg.max_kata_per_subtitle,
            model_size=cfg.auto_story_whisper_model,
            device=cfg.auto_story_whisper_device,
            compute_type=cfg.auto_story_whisper_compute_type,
            progress_callback=progress_callback,
        )
        transcript_cache = {"words": _flatten_words(segments)}
        _write_cache(transcript_path, transcript_key, transcript_cache)
        cache_usage["transcript"] = "created"
        if cache_callback is not None:
            cache_callback(dict(cache_usage))

    if diarization_cache is None:
        from . import diarization

        if phase_callback is not None:
            phase_callback("diarization")
        temp = tempfile.NamedTemporaryFile(
            prefix="auto_story_audio_", suffix=".wav", dir=cache_dir, delete=False
        )
        audio_path = temp.name
        temp.close()
        try:
            print("Extracting audio and identifying speakers with PyAnnote...")
            diarization.extract_audio(video_path, audio_path)
            turns = diarization.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=cfg.diarization_num_speakers,
            )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        diarization_cache = {"speaker_turns": turns}
        _write_cache(diarization_path, diarization_key, diarization_cache)
        cache_usage["diarization"] = "created"
        if cache_callback is not None:
            cache_callback(dict(cache_usage))

    words = _assign_speakers(
        transcript_cache["words"], diarization_cache["speaker_turns"]
    )
    combined_cache = {
        "source_id": source_id,
        "source_duration_sec": duration,
        "word_count": len(words),
        "words": words,
    }
    _write_cache(combined_path, combined_key, combined_cache)
    cache_usage["combined"] = "created"
    if cache_callback is not None:
        cache_callback(dict(cache_usage))
    return combined_cache["words"], combined_key


def _provider_model(cfg) -> tuple[str, str]:
    if cfg.ai_provider == "gemini":
        if not cfg.api_key_gemini:
            raise RuntimeError(
                "Auto Story requires GOOGLE_API_KEY when --ai-provider gemini is selected."
            )
        return cfg.ai_provider, cfg.gemini_model
    if cfg.ai_provider == "nvidia":
        if not cfg.api_key_nvidia:
            raise RuntimeError(
                "Auto Story requires NVIDIA_API_KEY when --ai-provider nvidia is selected."
            )
        return cfg.ai_provider, cfg.nvidia_model
    raise ValueError(f"Unsupported AI provider: {cfg.ai_provider}")


def _request_json(prompt: str, schema: dict, cfg) -> dict:
    provider, model = _provider_model(cfg)
    if provider == "gemini":
        import google.genai as genai
        from google.genai import types

        client = genai.Client(
            api_key=cfg.api_key_gemini,
            vertexai=False,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=AI_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        response = None
        for attempt in range(AI_REQUEST_MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                break
            except Exception as exc:
                if attempt + 1 >= AI_REQUEST_MAX_ATTEMPTS or not _is_retryable_ai_error(exc):
                    raise
                time.sleep(2 ** attempt)
        response_text = getattr(response, "text", None)
    else:
        import requests

        response = None
        for attempt in range(AI_REQUEST_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {cfg.api_key_nvidia}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2,
                    },
                    timeout=AI_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt + 1 >= AI_REQUEST_MAX_ATTEMPTS or not _is_retryable_ai_error(exc):
                    raise
                time.sleep(2 ** attempt)
        response_text = response.json()["choices"][0]["message"]["content"]

    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError(f"{provider} returned an empty Auto Story response.")
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider} returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{provider} response must be a JSON object.")
    return result


def _is_retryable_ai_error(exc: Exception) -> bool:
    """Retry transient provider errors, never auth or request/configuration errors."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__.lower()
    return any(token in name for token in (
        "timeout", "connectionerror", "connectionreset", "serviceunavailable",
        "resourceexhausted", "deadlineexceeded", "internalservererror",
    ))


_STORYLINES_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "storylines": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "key_characters": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "source_ranges": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_word_id": {"type": "STRING"},
                                "end_word_id": {"type": "STRING"},
                            },
                            "required": ["start_word_id", "end_word_id"],
                        },
                    },
                },
                "required": [
                    "id", "title", "summary", "key_characters", "source_ranges",
                ],
            },
        }
    },
    "required": ["storylines"],
}

_BEATS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "beats": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_word_id": {"type": "STRING"},
                    "end_word_id": {"type": "STRING"},
                    "narrative_role": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": [
                    "start_word_id", "end_word_id", "narrative_role", "reason",
                ],
            },
        }
    },
    "required": ["beats"],
}


def _validate_storylines(
    response: dict,
    words: list[dict],
    expected_count: int | None = 3,
    excluded_storylines: list[dict] | None = None,
) -> list[dict]:
    arcs = response.get("storylines")
    if not isinstance(arcs, list):
        raise ValueError("Storyline Discovery must return a storylines array.")
    if expected_count is not None and len(arcs) != expected_count:
        raise ValueError(
            f"Storyline Discovery must return exactly {expected_count} storylines."
        )
    if expected_count is None and len(arcs) > 8:
        raise ValueError("Storyline Discovery may return at most eight storylines.")
    word_positions = {word["word_id"]: index for index, word in enumerate(words)}
    seen_arc_ids = set()
    seen_titles = set()
    seen_summaries = set()
    excluded_titles = {
        re.sub(r"\W+", " ", str(item.get("title", "")).casefold()).strip()
        for item in excluded_storylines or []
    }
    excluded_summaries = {
        re.sub(r"\W+", " ", str(item.get("summary", "")).casefold()).strip()
        for item in excluded_storylines or []
    }
    validated = []
    for arc in arcs:
        if not isinstance(arc, dict):
            raise ValueError("Each discovered storyline must be a JSON object.")
        arc_id = str(arc.get("id", "")).strip()
        if not arc_id or arc_id in seen_arc_ids:
            raise ValueError("Storyline IDs must be non-empty and unique.")
        seen_arc_ids.add(arc_id)
        ranges = arc.get("source_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"Storyline {arc_id} must include transcript word ranges.")
        clean_ranges = []
        prior_end = -1
        for source_range in ranges:
            if not isinstance(source_range, dict):
                raise ValueError(f"Storyline {arc_id} contains an invalid word range.")
            start_id = source_range.get("start_word_id")
            end_id = source_range.get("end_word_id")
            if (
                not isinstance(start_id, str)
                or not isinstance(end_id, str)
                or start_id not in word_positions
                or end_id not in word_positions
            ):
                raise ValueError(f"Storyline {arc_id} references an unknown transcript word.")
            start_index = word_positions[start_id]
            end_index = word_positions[end_id]
            if end_index < start_index or start_index <= prior_end:
                raise ValueError(f"Storyline {arc_id} word ranges must be ordered and non-overlapping.")
            prior_end = end_index
            clean_ranges.append({
                "start_word_id": start_id,
                "end_word_id": end_id,
                "start_sec": words[start_index]["start"],
                "end_sec": words[end_index]["end"],
            })
        arc_words = [
            word
            for source_range in clean_ranges
            for word in words[
                word_positions[source_range["start_word_id"]]:
                word_positions[source_range["end_word_id"]] + 1
            ]
        ]
        characters = arc.get("key_characters", [])
        if not isinstance(characters, list):
            raise ValueError(f"Storyline {arc_id} key_characters must be an array.")
        title = str(arc.get("title", "")).strip()
        summary = str(arc.get("summary", "")).strip()
        if not title or not summary:
            raise ValueError(f"Storyline {arc_id} must have a title and summary.")
        title_key = re.sub(r"\W+", " ", title.casefold()).strip()
        summary_key = re.sub(r"\W+", " ", summary.casefold()).strip()
        if (
            title_key in seen_titles
            or title_key in excluded_titles
            or summary_key in seen_summaries
            or summary_key in excluded_summaries
        ):
            raise ValueError(
                f"Storyline {arc_id} repeats an existing storyline."
            )
        seen_titles.add(title_key)
        seen_summaries.add(summary_key)
        validated.append({
            "id": arc_id,
            "title": title,
            "summary": summary,
            "key_characters": [str(name).strip() for name in characters if str(name).strip()],
            "source_ranges": clean_ranges,
            "start_sec": arc_words[0]["start"],
            "end_sec": arc_words[-1]["end"],
            "speaker_ids": sorted({
                word["speaker_id"] for word in arc_words if word.get("speaker_id")
            }),
        })
    return validated


def _validate_beat_candidates(response: dict, arc: dict, words: list[dict]) -> None:
    beats = response.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ValueError("Beat Selection must return a non-empty beats array.")
    positions = {word["word_id"]: index for index, word in enumerate(words)}
    eligible_ranges = [
        (positions[item["start_word_id"]], positions[item["end_word_id"]])
        for item in arc["source_ranges"]
    ]
    for index, beat in enumerate(beats, 1):
        if not isinstance(beat, dict):
            raise ValueError(f"Beat #{index} must be a JSON object.")
        start_id = beat.get("start_word_id")
        end_id = beat.get("end_word_id")
        if (
            not isinstance(start_id, str)
            or not isinstance(end_id, str)
            or start_id not in positions
            or end_id not in positions
        ):
            raise ValueError(f"Beat #{index} references an unknown transcript word.")
        start_index = positions[start_id]
        end_index = positions[end_id]
        if end_index < start_index:
            raise ValueError(f"Beat #{index} ends before it starts.")
        if not any(
            range_start <= start_index <= end_index <= range_end
            for range_start, range_end in eligible_ranges
        ):
            raise ValueError(f"Beat #{index} is outside the selected storyline ranges.")


def _schedule_beat_candidates(
    response: dict,
    arc: dict,
    words: list[dict],
    episode_duration: float,
    target: float,
    min_beat: float,
    max_beat: float,
    min_gap: float,
    min_span_ratio: float,
    padding: float,
    crossfade: float,
) -> dict:
    positions = {word["word_id"]: index for index, word in enumerate(words)}
    valid = []
    seen = set()

    for beat in response["beats"]:
        start_index = positions[beat["start_word_id"]]
        end_index = positions[beat["end_word_id"]]
        start_word = words[start_index]
        end_word = words[end_index]
        duration = end_word["end"] - start_word["start"]
        sentence_extension = (
            duration <= min(6.5, max_beat + 0.5)
            and _is_sentence_end(end_word["word"])
        )
        if duration < min_beat or (
            duration > max_beat and not sentence_extension
        ):
            continue

        identity = (beat["start_word_id"], beat["end_word_id"])
        if identity in seen:
            continue
        seen.add(identity)
        clip_start = max(0.0, start_word["start"] - padding)
        clip_end = min(episode_duration, end_word["end"] + padding)
        if clip_end <= clip_start:
            continue

        role = str(beat.get("narrative_role", "")).lower()
        role_mask = 0
        if "hook" in role:
            role_mask |= 1
        if "escal" in role:
            role_mask |= 2
        if "twist" in role or "payoff" in role:
            role_mask |= 4
        valid.append({
            "beat": beat,
            "start_index": start_index,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "clip_duration": clip_end - clip_start,
            "role_mask": role_mask,
        })

    valid.sort(key=lambda item: (item["clip_start"], item["clip_end"]))
    if len(valid) < 2:
        raise ValueError(
            "AI did not provide enough valid dialogue candidates to build a story."
        )

    required_span = episode_duration * min_span_ratio

    def matching_pairs(first_pool, last_pool):
        return [
            (first, last)
            for first in first_pool
            for last in last_pool
            if first["start_index"] < last["start_index"]
            and last["clip_start"] - first["clip_end"] >= min_gap
            and last["clip_end"] - first["clip_start"] >= required_span
        ]

    hook_candidates = [item for item in valid if item["role_mask"] & 1]
    payoff_candidates = [item for item in valid if item["role_mask"] & 4]
    pairs = matching_pairs(
        hook_candidates or valid,
        payoff_candidates or valid,
    )
    if not pairs:
        pairs = matching_pairs(valid, valid)
    if not pairs:
        raise ValueError(
            "AI dialogue candidates do not span enough of the episode with the "
            f"required {min_gap:.1f}-second gaps. Choose another storyline."
        )

    first, last = max(
        pairs,
        key=lambda pair: (
            (pair[0]["role_mask"] | pair[1]["role_mask"]).bit_count(),
            pair[1]["clip_end"] - pair[0]["clip_start"],
        ),
    )

    middle_candidates = [
        item for item in valid
        if item is not first
        and item is not last
        and item["clip_start"] - first["clip_end"] >= min_gap
        and last["clip_start"] - item["clip_end"] >= min_gap
    ]
    middle_candidates.sort(key=lambda item: (item["clip_end"], item["clip_start"]))
    compatible = []
    previous_end = first["clip_end"]
    for item in middle_candidates:
        if item["clip_start"] - previous_end >= min_gap:
            compatible.append(item)
            previous_end = item["clip_end"]

    min_count, max_count = _expected_beat_count_range(
        target, min_beat, max_beat, padding, crossfade
    )
    min_count = max(2, min_count)
    anchor_duration = first["clip_duration"] + last["clip_duration"]
    anchor_roles = first["role_mask"] | last["role_mask"]
    states = {
        (2, round(anchor_duration * 10), anchor_roles): (anchor_duration, ())
    }
    upper_runtime = target * 1.1

    for candidate_index, item in enumerate(compatible):
        updated_states = dict(states)
        for (count, _duration_bucket, roles), (duration_sum, chosen) in states.items():
            if count >= max_count:
                continue
            next_count = count + 1
            next_duration = duration_sum + item["clip_duration"]
            runtime = next_duration - (next_count - 1) * crossfade
            if runtime > upper_runtime:
                continue
            next_roles = roles | item["role_mask"]
            key = (next_count, round(next_duration * 10), next_roles)
            previous = updated_states.get(key)
            if previous is None:
                updated_states[key] = (next_duration, chosen + (candidate_index,))
            else:
                previous_runtime = previous[0] - (next_count - 1) * crossfade
                if abs(runtime - target) < abs(previous_runtime - target):
                    updated_states[key] = (next_duration, chosen + (candidate_index,))
        states = updated_states

    lower_runtime = target * 0.9
    schedules = []
    for (count, _duration_bucket, roles), (duration_sum, chosen) in states.items():
        if not min_count <= count <= max_count:
            continue
        runtime = duration_sum - (count - 1) * crossfade
        if lower_runtime <= runtime <= upper_runtime:
            schedules.append((
                (
                    (roles).bit_count(),
                    -abs(runtime - target),
                    -abs(count - (min_count + max_count) / 2),
                ),
                [first, *(compatible[index] for index in chosen), last],
            ))

    if not schedules:
        raise ValueError(
            "AI dialogue candidates could not be scheduled to meet the target "
            "runtime, beat duration, episode span, and source-time gap constraints. "
            "Choose another storyline."
        )

    _, selected = max(schedules, key=lambda item: item[0])
    selected.sort(key=lambda item: item["clip_start"])
    return {"beats": [item["beat"] for item in selected]}


def _discover_storylines(
    words: list[dict],
    duration: float,
    transcript_key: str,
    cache_dir: str,
    output_dir: str,
    cfg,
    *,
    count: int | None = 3,
    excluded_storylines: list[dict] | None = None,
    cache_variant: str = "initial",
    output_filename: str = "storyline_discovery.json",
    target_duration_sec: int | None = None,
    min_gap_sec: float | None = None,
    speaker_names: dict[str, str] | None = None,
    speaker_corrections_key: str | None = None,
) -> list[dict]:
    provider, model = _provider_model(cfg)
    exclusions = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "summary": item.get("summary"),
        }
        for item in excluded_storylines or []
    ]
    cache_key = _digest({
        "prompt_version": PROMPT_VERSION,
        "pass": 1,
        "storyline_count": count,
        "excluded_storylines": exclusions,
        "cache_variant": cache_variant,
        "target_duration_sec": target_duration_sec,
        "min_gap_sec": min_gap_sec,
        "transcript": transcript_key,
        "speaker_names": speaker_names or {},
        "speaker_corrections": speaker_corrections_key,
        "provider": provider,
        "model": model,
    })
    cache_path = os.path.join(cache_dir, f"storyline_discovery_{cache_key[:12]}.json")
    cached = _read_cache(cache_path, cache_key)
    if cached is None:
        discovery_instruction = (
            "Find as many strong, distinct, self-contained story arcs as the "
            "transcript genuinely supports, up to eight. Return fewer when fewer "
            "are supported; never pad the list with weak or duplicate arcs. Return "
            "an empty storylines array if no coherent arc is supported."
            if count is None
            else f"Find exactly {count} distinct, self-contained story arcs."
        )
        prompt = (
            f"{discovery_instruction} Do not summarize the whole episode. The arcs must have "
            "different central conflicts, character dynamics, or outcomes; their "
            "source ranges may overlap. For each arc, return a "
            "short title, exactly two concise summary sentences, key character names "
            "that are explicitly supported by dialogue, and one or more chronological "
            "source_ranges. Each range must use exact start_word_id and end_word_id "
            "values copied from the transcript. Do not invent dialogue, IDs, names, "
            "or timestamps. Prefer arcs with a clear setup, escalation, and payoff.\n\n"
            + (
                "Do not repeat or reframe these already-planned story arcs:\n"
                f"{json.dumps(exclusions, ensure_ascii=False)}\n\n"
                if exclusions else ""
            )
            + (
                f"Each arc must contain enough distributed dialogue to support "
                f"an approximately {target_duration_sec}-second montage, using "
                f"{min_gap_sec:.1f}-second minimum source gaps between clips. "
                "Prefer source ranges that span enough of the episode for the "
                "scheduler to find distinct hook, escalation, and payoff beats.\n\n"
                if target_duration_sec is not None and min_gap_sec is not None
                else ""
            )
            + f"Episode duration: {duration:.3f} seconds.\n"
            + "Speaker-labeled word transcript (ID|start-end seconds|speaker|word):\n"
            + f"{_format_transcript(words, speaker_names)}"
        )
        validation_error = None
        for correction_attempt in range(2):
            request_prompt = prompt
            if validation_error:
                request_prompt += (
                    "\n\nYour previous response failed validation: "
                    f"{validation_error}. Return corrected storylines using only "
                    "the exact word IDs from the supplied transcript."
                )
            response = _request_json(request_prompt, _STORYLINES_SCHEMA, cfg)
            candidate = {"storylines": response.get("storylines")}
            try:
                _validate_storylines(candidate, words, count, excluded_storylines)
            except ValueError as exc:
                validation_error = str(exc)
                if correction_attempt == 1:
                    raise
                continue
            cached = candidate
            break
        _write_cache(cache_path, cache_key, cached)

    storylines = _validate_storylines(
        cached, words, count, excluded_storylines
    )
    _write_json(
        os.path.join(output_dir, output_filename),
        {
            "source_duration_sec": duration,
            "story_count": len(storylines),
            "storylines": storylines,
        },
    )
    return storylines


def _expected_beat_count_range(
    target: float,
    minimum: float,
    maximum: float,
    padding: float,
    crossfade: float,
) -> tuple[int, int]:
    target_low, target_high = target * 0.9, target * 1.1
    possible = []
    for count in range(1, 100):
        shortest = count * (minimum + 2 * padding) - max(0, count - 1) * crossfade
        longest = count * (maximum + 2 * padding) - max(0, count - 1) * crossfade
        if shortest <= target_high and longest >= target_low:
            possible.append(count)
    if not possible:
        raise ValueError("No beat count can satisfy the selected duration constraints.")
    return min(possible), max(possible)


def _select_arc_words(arc: dict, words: list[dict]) -> list[dict]:
    positions = {word["word_id"]: index for index, word in enumerate(words)}
    return [
        word
        for source_range in arc["source_ranges"]
        for word in words[
            positions[source_range["start_word_id"]]:
            positions[source_range["end_word_id"]] + 1
        ]
    ]


def _select_beats(
    arc: dict,
    words: list[dict],
    target: int,
    min_beat: float,
    max_beat: float,
    min_gap: float,
    min_span_ratio: float,
    episode_duration: float,
    padding: float,
    crossfade: float,
    transcript_key: str,
    cache_dir: str,
    cfg,
    *,
    candidate_round: int = 0,
    planning_revision: int = 0,
    speaker_names: dict[str, str] | None = None,
    speaker_corrections_key: str | None = None,
) -> tuple[dict, str]:
    provider, model = _provider_model(cfg)
    settings = {
        "target_duration_sec": target,
        "min_beat_sec": min_beat,
        "max_beat_sec": max_beat,
        "min_gap_sec": min_gap,
        "min_episode_span_ratio": min_span_ratio,
        "padding_sec": padding,
        "crossfade_sec": crossfade,
    }
    cache_key = _digest({
        "prompt_version": BEAT_PROMPT_VERSION,
        "pass": 2,
        "candidate_round": candidate_round,
        "planning_revision": planning_revision,
        "transcript": transcript_key,
        "speaker_names": speaker_names or {},
        "speaker_corrections": speaker_corrections_key,
        "arc": arc,
        "settings": settings,
        "provider": provider,
        "model": model,
    })
    cache_path = os.path.join(cache_dir, f"beat_selection_{cache_key[:12]}.json")
    cached = _read_cache(cache_path, cache_key)
    if cached is None:
        min_count, max_count = _expected_beat_count_range(
            target, min_beat, max_beat, padding, crossfade
        )
        candidate_multiplier = 2 + candidate_round
        min_candidates = max(
            min_count * candidate_multiplier,
            min_count + 4 + candidate_round * 4,
        )
        max_candidates = max(
            min_candidates,
            min(
                max_count * candidate_multiplier,
                min_candidates + 16 + candidate_round * 8,
            ),
        )
        prompt = (
            "Build a candidate pool of chronological dialogue-only beats that tell "
            "the selected storyline as Hook -> Escalation -> Twist/Payoff. Return one "
            'JSON object with a "beats" array containing between '
            f"{min_candidates} and {max_candidates} candidate beats. These are "
            "alternatives, not the final cut; a deterministic scheduler will select "
            "a valid subset for automatic rendering. Spread candidates across the full episode "
            "span, include alternatives for every story phase, and avoid clustering "
            "all candidates around the same scene. Candidates may overlap or fall "
            "inside the minimum gap; the scheduler will resolve those conflicts. "
            "Each beat must identify exact "
            "start_word_id and end_word_id values from the provided transcript, and "
            "include a concise narrative_role and reason. Use only a contiguous "
            "dialogue interval from one supplied source range. Never invent or "
            "paraphrase source dialogue. The final montage needs "
            f"{min_count}-{max_count} beats. Individual dialogue intervals should be "
            f"{min_beat:.2f}-{max_beat:.2f} seconds; a sentence may finish up to "
            "0.5 seconds beyond the maximum only if the final transcript word ends "
            "with sentence punctuation, and never exceed 6.5 seconds. Return "
            "candidates chronologically and ensure the pool covers at least "
            f"{min_span_ratio:.0%} of the episode. The final scheduler enforces "
            f"{min_gap:.1f}-second source gaps. After adding "
            f"{padding:.2f}s padding to each side and applying {crossfade:.2f}s "
            "crossfades, total runtime must be within 10% of the target.\n\n"
            f"Selected storyline:\n{json.dumps(arc, ensure_ascii=False)}\n"
            f"Target duration: {target} seconds; episode duration: "
            f"{episode_duration:.3f} seconds.\n"
            "Eligible speaker-labeled transcript (ID|start-end seconds|speaker|word):\n"
            f"{_format_transcript(_select_arc_words(arc, words), speaker_names)}"
        )
        if candidate_round:
            prompt = (
                "A previous candidate pool could not satisfy the automatic schedule. "
                "Return a broader and more varied set of valid alternatives, focusing "
                "on separate scenes across the selected arc and ensuring enough clips "
                "can meet all timing and spacing constraints.\n\n"
                + prompt
            )
        response = _request_json(prompt, _BEATS_SCHEMA, cfg)
        cached = {"beats": response.get("beats")}
        _validate_beat_candidates(cached, arc, words)
        _write_cache(cache_path, cache_key, cached)
    _validate_beat_candidates(cached, arc, words)
    scheduled = _schedule_beat_candidates(
        cached,
        arc,
        words,
        episode_duration,
        target,
        min_beat,
        max_beat,
        min_gap,
        min_span_ratio,
        padding,
        crossfade,
    )
    return scheduled, cache_key


def _build_review_recipe(
    response: dict,
    arc: dict,
    words: list[dict],
    source_id: str,
    settings: dict,
) -> dict:
    beats = response.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ValueError("Beat Selection returned no beats.")
    by_id = {word["word_id"]: word for word in words}
    positions = {word["word_id"]: index for index, word in enumerate(words)}
    recipe_beats = []
    for index, beat in enumerate(beats, 1):
        if not isinstance(beat, dict):
            raise ValueError(f"Beat #{index} is not a JSON object.")
        start_id = beat.get("start_word_id")
        end_id = beat.get("end_word_id")
        if (
            not isinstance(start_id, str)
            or not isinstance(end_id, str)
            or start_id not in by_id
            or end_id not in by_id
        ):
            raise ValueError(f"Beat #{index} references an unknown transcript word.")
        start_word = by_id[start_id]
        end_word = by_id[end_id]
        start_index = positions[start_id]
        end_index = positions[end_id]
        if end_index < start_index:
            raise ValueError(f"Beat #{index} ends before it starts.")
        selected_words = words[start_index:end_index + 1]
        recipe_beats.append({
            "beat_id": index,
            "start_word_id": start_id,
            "end_word_id": end_id,
            "dialogue_start_sec": round(start_word["start"], 3),
            "dialogue_end_sec": round(end_word["end"], 3),
            "text": " ".join(word["word"] for word in selected_words),
            "speaker_ids": sorted({
                word["speaker_id"] for word in selected_words
                if word.get("speaker_id")
            }),
            "narrative_role": str(beat.get("narrative_role", "")).strip(),
            "reason": str(beat.get("reason", "")).strip(),
        })
    return {
        "schema_version": 1,
        "source_id": source_id,
        "storyline": arc,
        "target_duration_sec": settings["target_duration_sec"],
        "settings": settings,
        "beats": recipe_beats,
        "review_note": (
            "You may adjust dialogue_start_sec/dialogue_end_sec to exact word "
            "boundaries or add/remove beats. The final validation derives text, "
            "speaker IDs, and padded cut times from the cached transcript."
        ),
    }


def _build_review_timing_options(
    recipe: dict,
    arc: dict,
    words: list[dict],
    radius: int = 8,
) -> dict:
    positions = {word["word_id"]: index for index, word in enumerate(words)}
    eligible_ranges = [
        (
            positions[item["start_word_id"]],
            positions[item["end_word_id"]],
        )
        for item in arc["source_ranges"]
    ]
    max_duration = min(
        6.5,
        float(recipe.get("settings", {}).get("max_beat_sec", 6.0)) + 0.5,
    )
    options = {}

    for index, beat in enumerate(recipe.get("beats", []), 1):
        start_index = positions.get(beat.get("start_word_id"))
        end_index = positions.get(beat.get("end_word_id"))
        if start_index is None or end_index is None or end_index < start_index:
            raise ValueError(f"Beat #{index} references invalid transcript words.")

        source_range = next(
            (
                (range_start, range_end)
                for range_start, range_end in eligible_ranges
                if range_start <= start_index <= end_index <= range_end
            ),
            None,
        )
        if source_range is None:
            raise ValueError(f"Beat #{index} is outside the selected storyline.")
        range_start, range_end = source_range
        start_word = words[start_index]
        end_word = words[end_index]
        beat_options = {"start": [], "end": []}

        for option_index in range(
            max(range_start, start_index - radius),
            min(end_index, start_index + radius) + 1,
        ):
            option = words[option_index]
            duration = end_word["end"] - option["start"]
            if option_index == start_index or 0 < duration <= max_duration:
                beat_options["start"].append({
                    "time_sec": option["start"],
                    "word": option["word"],
                    "word_id": option["word_id"],
                })

        for option_index in range(
            max(start_index, end_index - radius),
            min(range_end, end_index + radius) + 1,
        ):
            option = words[option_index]
            duration = option["end"] - start_word["start"]
            if option_index == end_index or 0 < duration <= max_duration:
                beat_options["end"].append({
                    "time_sec": option["end"],
                    "word": option["word"],
                    "word_id": option["word_id"],
                })

        options[str(beat.get("beat_id", index))] = beat_options

    return options


def _nearest_word(words: list[dict], timestamp: float, endpoint: str) -> dict:
    key = "start" if endpoint == "start" else "end"
    nearest = min(words, key=lambda word: abs(word[key] - timestamp))
    if abs(nearest[key] - timestamp) > WORD_BOUNDARY_TOLERANCE:
        raise ValueError(
            f"{endpoint} time {timestamp:.3f}s is not within "
            f"{WORD_BOUNDARY_TOLERANCE:.2f}s of a cached word boundary."
        )
    return nearest


def _is_sentence_end(word: str) -> bool:
    cleaned = word.rstrip("\"')]}")
    return bool(cleaned) and cleaned[-1] in ".!?"


def validate_review_recipe(
    recipe: dict,
    arc: dict,
    words: list[dict],
    source_id: str,
    episode_duration: float,
    target_duration: float,
    min_beat_sec: float,
    max_beat_sec: float,
    min_gap_sec: float,
    min_episode_span_ratio: float,
    padding_sec: float,
    crossfade_sec: float,
) -> dict:
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object.")
    if recipe.get("source_id") != source_id:
        raise ValueError(f"Recipe source_id must remain {source_id!r}.")
    recipe_storyline = recipe.get("storyline")
    if not isinstance(recipe_storyline, dict) or recipe_storyline.get("id") != arc["id"]:
        raise ValueError("Recipe storyline ID does not match the selected storyline.")
    beats = recipe.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ValueError("Recipe must contain at least one beat.")

    word_positions = {word["word_id"]: index for index, word in enumerate(words)}
    eligible_ranges = [
        (word_positions[item["start_word_id"]], word_positions[item["end_word_id"]])
        for item in arc["source_ranges"]
    ]
    validated_beats = []
    issues = []
    for index, beat in enumerate(beats, 1):
        try:
            requested_start = float(beat["dialogue_start_sec"])
            requested_end = float(beat["dialogue_end_sec"])
            if not math.isfinite(requested_start) or not math.isfinite(requested_end):
                raise ValueError("timestamps must be finite numbers.")
            start_word = _nearest_word(words, requested_start, "start")
            end_word = _nearest_word(words, requested_end, "end")
            start_index = word_positions[start_word["word_id"]]
            end_index = word_positions[end_word["word_id"]]
            if end_index < start_index:
                raise ValueError("end time must follow start time.")
            if not any(
                range_start <= start_index <= end_index <= range_end
                for range_start, range_end in eligible_ranges
            ):
                raise ValueError("beat must stay inside one selected storyline range.")
            selected_words = words[start_index:end_index + 1]
            duration = end_word["end"] - start_word["start"]
            sentence_extension = (
                duration <= min(6.5, max_beat_sec + 0.5)
                and _is_sentence_end(end_word["word"])
            )
            if duration < min_beat_sec or (
                duration > max_beat_sec and not sentence_extension
            ):
                raise ValueError(
                    f"dialogue duration {duration:.2f}s is outside "
                    f"{min_beat_sec:.2f}-{max_beat_sec:.2f}s "
                    "(up to 0.5s extra only to finish a punctuated sentence, "
                    "maximum 6.5s)."
                )
            clip_start = max(0.0, start_word["start"] - padding_sec)
            clip_end = min(episode_duration, end_word["end"] + padding_sec)
            validated_beats.append({
                "beat_id": index,
                "start_word_id": start_word["word_id"],
                "end_word_id": end_word["word_id"],
                "dialogue_start_sec": start_word["start"],
                "dialogue_end_sec": end_word["end"],
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "duration_sec": clip_end - clip_start,
                "text": " ".join(word["word"] for word in selected_words),
                "speaker_ids": sorted({
                    word["speaker_id"] for word in selected_words
                    if word.get("speaker_id")
                }),
                "narrative_role": str(beat.get("narrative_role", "")).strip(),
                "reason": str(beat.get("reason", "")).strip(),
                "word_ids": [word["word_id"] for word in selected_words],
            })
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"Beat #{index}: {exc}")

    if not issues:
        for index in range(1, len(validated_beats)):
            previous = validated_beats[index - 1]
            current = validated_beats[index]
            gap = current["clip_start_sec"] - previous["clip_end_sec"]
            if gap < min_gap_sec:
                issues.append(
                    f"Beats #{index} and #{index + 1} are only {gap:.1f}s apart; "
                    f"minimum is {min_gap_sec:.1f}s."
                )

    if validated_beats and len(validated_beats) == len(beats):
        clip_span = (
            validated_beats[-1]["clip_end_sec"]
            - validated_beats[0]["clip_start_sec"]
        )
        required_span = episode_duration * min_episode_span_ratio
        if clip_span < required_span:
            issues.append(
                f"Episode span is {clip_span:.1f}s; at least "
                f"{required_span:.1f}s ({min_episode_span_ratio:.0%} of the episode) "
                "is required."
            )
        runtime = sum(beat["duration_sec"] for beat in validated_beats)
        runtime -= max(0, len(validated_beats) - 1) * crossfade_sec
        lower, upper = target_duration * 0.9, target_duration * 1.1
        if not lower <= runtime <= upper:
            issues.append(
                f"Estimated montage runtime is {runtime:.1f}s; expected "
                f"{lower:.1f}-{upper:.1f}s for a {target_duration}s target."
            )
    else:
        runtime = 0.0
        clip_span = 0.0

    if issues:
        raise ValueError("Story Recipe validation failed:\n- " + "\n- ".join(issues))
    return {
        "source_id": source_id,
        "storyline": arc,
        "target_duration_sec": target_duration,
        "estimated_runtime_sec": runtime,
        "episode_span_sec": clip_span,
        "settings": {
            "min_beat_sec": min_beat_sec,
            "max_beat_sec": max_beat_sec,
            "min_gap_sec": min_gap_sec,
            "min_episode_span_ratio": min_episode_span_ratio,
            "padding_sec": padding_sec,
            "crossfade_sec": crossfade_sec,
        },
        "beats": validated_beats,
    }


def _build_caption_segments(
    beats: list[dict],
    words: list[dict],
    crossfade_sec: float,
    max_words_per_segment: int,
) -> tuple[list[dict], float]:
    words_by_id = {word["word_id"]: word for word in words}
    caption_segments = []
    timeline_start = 0.0
    for beat_index, beat in enumerate(beats):
        mapped_words = []
        for word_id in beat["word_ids"]:
            word = words_by_id[word_id]
            start = timeline_start + word["start"] - beat["clip_start_sec"]
            end = timeline_start + word["end"] - beat["clip_start_sec"]
            mapped_words.append({
                "word": word["word"],
                "start": max(timeline_start, start),
                "end": max(timeline_start, end),
            })
        for offset in range(0, len(mapped_words), max_words_per_segment):
            group = mapped_words[offset:offset + max_words_per_segment]
            caption_segments.append({
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "words": group,
            })
        timeline_start += beat["duration_sec"]
        if beat_index < len(beats) - 1:
            timeline_start -= crossfade_sec
    return caption_segments, timeline_start


def _render_montage(
    validated_recipe: dict,
    words: list[dict],
    source_registry: dict,
    cache_dir: str,
    output_dir: str,
    cfg,
    output_prefix: str | None = None,
) -> tuple[str, str, float]:
    from . import studio
    from .story import assembler

    os.makedirs(output_dir, exist_ok=True)
    transition_sec = validated_recipe["settings"]["crossfade_sec"]
    clip_config = {
        "clip_id": 1,
        "title": validated_recipe["storyline"]["title"],
        "highlight": {
            "transition": "crossfade" if len(validated_recipe["beats"]) > 1 else "cut",
            "scenes": [
                {
                    "source_id": validated_recipe["source_id"],
                    "start": beat["clip_start_sec"],
                    "end": beat["clip_end_sec"],
                    "label": beat["text"],
                }
                for beat in validated_recipe["beats"]
            ],
        },
    }
    filename_prefix = f"{output_prefix}_" if output_prefix else ""
    output_path = os.path.join(
        output_dir, f"{filename_prefix}story_montage.mp4"
    )
    captions_path = os.path.join(
        output_dir, f"{filename_prefix}story_montage.ass"
    )
    caption_segments, expected_runtime = _build_caption_segments(
        validated_recipe["beats"],
        words,
        transition_sec,
        cfg.max_kata_per_subtitle,
    )
    studio.buat_file_ass(
        caption_segments,
        0.0,
        expected_runtime,
        captions_path,
        cfg.pilihan_rasio,
        cfg,
        gunakan_advanced=False,
    )

    with tempfile.TemporaryDirectory(
        prefix="_auto_story_render_", dir=output_dir
    ) as temp_dir:
        raw_path = os.path.join(temp_dir, "highlight_1.mp4")
        assembled_path = assembler.assemble_highlight(
            clip_config=clip_config,
            source_registry=source_registry,
            cache_dir=cache_dir,
            output_dir=temp_dir,
            ratio=cfg.pilihan_rasio,
            transition_duration=transition_sec,
        )
        if not assembled_path or not os.path.isfile(assembled_path):
            raise RuntimeError("Story Clip assembler did not produce the raw montage.")
        if cfg.no_subs:
            os.replace(raw_path, output_path)
        else:
            subtitle_filter = studio.build_subtitles_filter(
                captions_path, cfg.font_dir
            )
            assembler._run_ffmpeg(
                [
                    "ffmpeg", "-y", "-i", raw_path,
                    "-vf", subtitle_filter,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "copy", "-movflags", "+faststart", output_path,
                ],
                label="burn Auto Story karaoke captions",
            )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Auto Story renderer did not produce a non-empty MP4.")
    actual_runtime = _probe_duration(output_path)
    lower = validated_recipe["target_duration_sec"] * 0.9
    upper = validated_recipe["target_duration_sec"] * 1.1
    if not lower <= actual_runtime <= upper:
        raise RuntimeError(
            f"Rendered montage is {actual_runtime:.1f}s, outside the accepted "
            f"{lower:.1f}-{upper:.1f}s target range."
        )
    return output_path, captions_path, actual_runtime


def run_auto_story_pipeline(cfg, input_fn=input) -> dict | None:
    """Run Auto Story with mandatory storyline, recipe, and render reviews."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Auto Story requires an interactive terminal for storyline selection "
            "and render approval."
        )
    _provider_model(cfg)

    from .story import loader, source_manager

    source_registry = loader.load_sources(cfg.sources_json_path)
    if len(source_registry) != 1:
        raise ValueError("Auto Story v1 accepts exactly one episode source.")
    source_id, source = next(iter(source_registry.items()))
    if not isinstance(source_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,59}", source_id
    ):
        raise ValueError(
            "Auto Story source IDs must contain only letters, numbers, dots, "
            "underscores, or hyphens, and must start with a letter or number."
        )
    cache_dir = source_manager.get_cache_dir(cfg.outputs_dir)
    if source["platform"] == "local":
        video_path = source["local_path"]
    elif cfg.skip_download:
        video_path = os.path.join(cache_dir, f"{source_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Cached episode not found: {video_path}. Remove --skip-download "
                "or download the source first."
            )
    else:
        paths = source_manager.download_all_sources(
            source_registry,
            cache_dir,
            getattr(cfg, "download_source_height", "max"),
        )
        if source_id not in paths:
            raise RuntimeError(f"Episode source download failed: {source_id}")
        video_path = paths[source_id]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Episode video not found: {video_path}")

    safe_id = _safe_source_name(source_id)
    auto_cache_dir = os.path.join(cache_dir, "auto_story", safe_id)
    output_dir = os.path.join(cfg.auto_story_output_dir, safe_id)
    os.makedirs(auto_cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    duration = _probe_duration(video_path)
    words, transcript_key = _load_or_create_transcript(
        video_path, source_id, auto_cache_dir, cfg, duration
    )
    storylines = _discover_storylines(
        words, duration, transcript_key, auto_cache_dir, output_dir, cfg
    )

    print("\nStoryline Discovery - select one arc:")
    for index, arc in enumerate(storylines, 1):
        print(
            f"\n{index}. {arc['title']}\n"
            f"   {arc['summary']}\n"
            f"   Characters: {', '.join(arc['key_characters']) or 'Not specified'}\n"
            f"   Speakers: {', '.join(arc['speaker_ids']) or 'Not identified'}\n"
            f"   Source span: {arc['start_sec']:.1f}s-{arc['end_sec']:.1f}s"
        )
    selection = input_fn("\nChoose a storyline (1-3), or q to stop: ").strip().lower()
    if selection in {"q", "quit"}:
        print("Auto Story stopped before Beat Selection.")
        return None
    if selection not in {"1", "2", "3"}:
        raise ValueError("Choose storyline 1, 2, or 3.")
    arc = storylines[int(selection) - 1]

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
    beat_response, beat_cache_key = _select_beats(
        arc,
        words,
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        cfg.auto_story_min_gap_sec,
        cfg.auto_story_min_episode_span_ratio,
        duration,
        cfg.auto_story_padding_sec,
        crossfade_sec,
        transcript_key,
        auto_cache_dir,
        cfg,
    )
    review_path = os.path.join(
        output_dir, f"story_recipe_review_{beat_cache_key[:12]}.json"
    )
    if not os.path.exists(review_path):
        review_recipe = _build_review_recipe(
            beat_response, arc, words, source_id, settings
        )
        _write_json(review_path, review_recipe)

    print(
        "\nBeat Selection complete. Review or edit the recipe file, then return "
        "to this terminal:\n"
        f"  {review_path}"
    )
    input_fn("Press Enter after reviewing the recipe JSON (Ctrl+C to stop): ")
    with open(review_path, "r", encoding="utf-8") as file:
        edited_recipe = json.load(file)
    validated_recipe = validate_review_recipe(
        edited_recipe,
        arc,
        words,
        source_id,
        duration,
        cfg.auto_story_target_duration,
        cfg.auto_story_min_beat_sec,
        cfg.auto_story_max_beat_sec,
        cfg.auto_story_min_gap_sec,
        cfg.auto_story_min_episode_span_ratio,
        cfg.auto_story_padding_sec,
        crossfade_sec,
    )
    print(
        f"Validated {len(validated_recipe['beats'])} beats; estimated montage "
        f"runtime {validated_recipe['estimated_runtime_sec']:.1f}s."
    )
    approval = input_fn("Render this reviewed recipe? Type yes to continue: ")
    if approval.strip().lower() not in {"y", "yes"}:
        print("Render not approved; recipe and caches have been kept.")
        return None

    output_path, captions_path, actual_runtime = _render_montage(
        validated_recipe, words, source_registry, cache_dir, output_dir, cfg
    )
    manifest_path = os.path.join(output_dir, "story_recipe_manifest.json")
    manifest = {
        "schema_version": 1,
        "source_id": source_id,
        "source_name": source.get("name", source_id),
        "storyline": validated_recipe["storyline"],
        "target_duration_sec": validated_recipe["target_duration_sec"],
        "estimated_runtime_sec": validated_recipe["estimated_runtime_sec"],
        "actual_runtime_sec": actual_runtime,
        "settings": validated_recipe["settings"],
        "video_path": output_path,
        "captions_path": captions_path,
        "review_recipe_path": review_path,
        "beats": validated_recipe["beats"],
    }
    _write_json(manifest_path, manifest)
    print(f"\nAuto Story complete:\n  Video: {output_path}\n  Recipe manifest: {manifest_path}")
    return manifest
