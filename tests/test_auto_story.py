import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipping import auto_story
from clipping.story import assembler


def test_flatten_words_skips_zero_duration_word_and_keeps_valid_words(capsys):
    words = auto_story._flatten_words([{
        "words": [
            {"word": "Black", "start": 197.98, "end": 197.98},
            {"word": "is", "start": 198.0, "end": 198.2},
        ],
    }])

    assert [word["word"] for word in words] == ["is"]
    assert words[0]["word_id"] == "W000001"
    assert "Skipped 1 word(s) with zero-duration timestamps." in capsys.readouterr().out


def test_flatten_words_rejects_reversed_timestamps():
    with pytest.raises(ValueError, match="Invalid word timestamps"):
        auto_story._flatten_words([{
            "words": [{"word": "invalid", "start": 2.0, "end": 1.0}],
        }])


def test_transcribe_video_reports_audio_progress(monkeypatch):
    from clipping import engine

    class Progress:
        def __init__(self, total, **_kwargs):
            self.n = 0.0
            self.total = total

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    segment = SimpleNamespace(
        start=0.0,
        end=5.0,
        text="Hello",
        words=[SimpleNamespace(word="Hello", start=0.1, end=0.5)],
    )

    class Model:
        def transcribe(self, _video_path, **_kwargs):
            return [segment], SimpleNamespace(duration=10.0)

    model = Model()
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=Progress))
    progress = []
    engine.transcribe_video(
        "episode.mp4",
        whisper_model_instance=model,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(5.0, 10.0), (10.0, 10.0)]


def _episode_words():
    words = []
    beats = []
    for index in range(12):
        start = 100.0 + index * 150
        first_id = f"W{len(words) + 1:06d}"
        words.extend([
            {
                "word_id": first_id,
                "word": f"word{index}",
                "start": start,
                "end": start + 2.5,
                "speaker_id": "SPEAKER_00",
            },
            {
                "word_id": f"W{len(words) + 2:06d}",
                "word": "finished.",
                "start": start + 2.5,
                "end": start + 5.0,
                "speaker_id": "SPEAKER_00",
            },
        ])
        beats.append({
            "start_word_id": first_id,
            "end_word_id": words[-1]["word_id"],
            "narrative_role": "escalation",
            "reason": "Moves the story forward.",
        })
    arc = {
        "id": "arc_1",
        "title": "The Turning Point",
        "summary": "A test story. It reaches a clear payoff.",
        "key_characters": ["Alex"],
        "source_ranges": [{
            "start_word_id": words[0]["word_id"],
            "end_word_id": words[-1]["word_id"],
            "start_sec": words[0]["start"],
            "end_sec": words[-1]["end"],
        }],
        "start_sec": words[0]["start"],
        "end_sec": words[-1]["end"],
        "speaker_ids": ["SPEAKER_00"],
    }
    return words, arc, beats


def test_transcript_and_diarization_are_cached_separately(monkeypatch, tmp_path):
    from clipping import diarization, engine

    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"test video")
    cache_dir = tmp_path / "cache"
    calls = {"transcribe": 0, "diarize": 0}

    def fake_transcribe(*args, **kwargs):
        calls["transcribe"] += 1
        return "hello world", [{
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"word": "Hello", "start": 0.1, "end": 0.4},
                {"word": "world.", "start": 0.5, "end": 0.9},
            ],
        }]

    def fake_extract_audio(_video_path, audio_path):
        Path(audio_path).write_bytes(b"wav")
        return audio_path

    def fake_diarize(*args, **kwargs):
        calls["diarize"] += 1
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(engine, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(diarization, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(diarization, "run_diarization", fake_diarize)
    cfg = SimpleNamespace(
        auto_story_whisper_model="small",
        auto_story_whisper_device="cpu",
        auto_story_whisper_compute_type="int8",
        max_kata_per_subtitle=5,
        hf_token="test-token",
        diarization_num_speakers="auto",
    )

    first, key = auto_story._load_or_create_transcript(
        str(video_path), "episode", str(cache_dir), cfg, 30.0
    )
    cfg.hf_token = ""
    second, second_key = auto_story._load_or_create_transcript(
        str(video_path), "episode", str(cache_dir), cfg, 30.0
    )

    assert key == second_key
    assert first == second
    assert first[0]["speaker_id"] == "SPEAKER_00"
    assert calls == {"transcribe": 1, "diarize": 1}


def test_recipe_validation_enforces_span_gaps_and_runtime():
    words, arc, beats = _episode_words()
    recipe = auto_story._build_review_recipe(
        {"beats": beats},
        arc,
        words,
        "episode",
        {"target_duration_sec": 60},
    )
    assert recipe["beats"][0]["text"] == "word0 finished."
    assert recipe["beats"][0]["speaker_ids"] == ["SPEAKER_00"]
    result = auto_story.validate_review_recipe(
        recipe,
        arc,
        words,
        "episode",
        episode_duration=3600,
        target_duration=60,
        min_beat_sec=3.5,
        max_beat_sec=6.0,
        min_gap_sec=30,
        min_episode_span_ratio=0.4,
        padding_sec=0.18,
        crossfade_sec=0.06,
    )

    assert len(result["beats"]) == 12
    assert result["beats"][0]["text"] == "word0 finished."
    assert result["estimated_runtime_sec"] == pytest.approx(63.66)
    assert result["beats"][0]["clip_start_sec"] == pytest.approx(99.82)
    captions, runtime = auto_story._build_caption_segments(
        result["beats"], words, crossfade_sec=0.06, max_words_per_segment=5
    )
    assert runtime == pytest.approx(result["estimated_runtime_sec"])
    assert captions[0]["words"][0]["start"] == pytest.approx(0.18)
    assert captions[1]["words"][0]["start"] == pytest.approx(5.48)

    with pytest.raises(ValueError, match="minimum is 200.0s"):
        auto_story.validate_review_recipe(
            recipe,
            arc,
            words,
            "episode",
            episode_duration=3600,
            target_duration=60,
            min_beat_sec=3.5,
            max_beat_sec=6.0,
            min_gap_sec=200,
            min_episode_span_ratio=0.4,
            padding_sec=0.18,
            crossfade_sec=0.06,
        )

    invalid_recipe = dict(recipe)
    invalid_recipe["beats"] = [dict(recipe["beats"][0])]
    invalid_recipe["beats"][0]["dialogue_start_sec"] = float("nan")
    with pytest.raises(ValueError, match="finite numbers"):
        auto_story.validate_review_recipe(
            invalid_recipe,
            arc,
            words,
            "episode",
            episode_duration=3600,
            target_duration=60,
            min_beat_sec=3.5,
            max_beat_sec=6.0,
            min_gap_sec=30,
            min_episode_span_ratio=0.4,
            padding_sec=0.18,
            crossfade_sec=0.06,
        )


def test_storyline_and_beat_ai_passes_reuse_cached_responses(monkeypatch, tmp_path):
    words = [
        {
            "word_id": f"W{index:06d}",
            "word": f"word{index}.",
            "start": float(index),
            "end": float(index) + 0.5,
            "speaker_id": "SPEAKER_00",
        }
        for index in range(1, 5)
    ]
    calls = []

    def fake_request(prompt, schema, _cfg):
        calls.append(schema)
        if schema is auto_story._STORYLINES_SCHEMA:
            return {
                "storylines": [
                    {
                        "id": f"arc_{index}",
                        "title": f"Arc {index}",
                        "summary": "A setup. A payoff.",
                        "key_characters": ["Alex"],
                        "source_ranges": [{
                            "start_word_id": f"W{index:06d}",
                            "end_word_id": f"W{index:06d}",
                        }],
                    }
                    for index in range(1, 4)
                ]
            }
        return {
            "beats": [{
                "start_word_id": "W000001",
                "end_word_id": "W000001",
                "narrative_role": "hook",
                "reason": "Opens the story.",
            }]
        }

    monkeypatch.setattr(auto_story, "_request_json", fake_request)
    cfg = SimpleNamespace(
        ai_provider="gemini",
        api_key_gemini="test-key",
        gemini_model="test-model",
        nvidia_model="test-model",
    )
    cache_dir = str(tmp_path / "cache")
    output_dir = str(tmp_path / "output")

    first_arcs = auto_story._discover_storylines(
        words, 100, "transcript-key", cache_dir, output_dir, cfg
    )
    second_arcs = auto_story._discover_storylines(
        words, 100, "transcript-key", cache_dir, output_dir, cfg
    )
    first_beats, _ = auto_story._select_beats(
        first_arcs[0], words, 60, 3.5, 6, 30, 0.4, 3600, 0.18, 0.06,
        "transcript-key", cache_dir, cfg,
    )
    second_beats, _ = auto_story._select_beats(
        second_arcs[0], words, 60, 3.5, 6, 30, 0.4, 3600, 0.18, 0.06,
        "transcript-key", cache_dir, cfg,
    )

    assert first_arcs == second_arcs
    assert first_beats["beats"] == second_beats["beats"]
    assert calls == [auto_story._STORYLINES_SCHEMA, auto_story._BEATS_SCHEMA]


def test_auto_story_cli_defaults_to_local_cpu_and_validates_mode(monkeypatch, tmp_path):
    from clipping import config

    monkeypatch.chdir(tmp_path)
    cfg = config.build_config(["--auto-story-mode"])
    assert cfg.auto_story_mode is True
    assert cfg.auto_story_target_duration == 60
    assert cfg.auto_story_whisper_device == "cpu"
    assert cfg.auto_story_whisper_compute_type == "int8"

    with pytest.raises(SystemExit):
        config.build_config([
            "--auto-story-mode", "--story-mode", "--target-duration", "60",
        ])
    with pytest.raises(SystemExit):
        config.build_config([
            "--auto-story-mode", "--min-gap-sec", "nan",
        ])


def test_crossfade_offsets_are_at_scene_boundaries(monkeypatch, tmp_path):
    commands = []

    def fake_run_ffmpeg(command, label=""):
        commands.append((command, label))
        Path(command[-1]).write_bytes(b"rendered")

    monkeypatch.setattr(assembler, "_run_ffmpeg", fake_run_ffmpeg)
    output = tmp_path / "montage.mp4"
    assembler._concat_with_crossfade(
        ["first.mp4", "second.mp4", "third.mp4"],
        str(output),
        fade_duration=0.06,
        scene_durations=[5.0, 4.0, 3.0],
    )

    offsets = []
    for command, _ in commands:
        filter_complex = command[command.index("-filter_complex") + 1]
        offsets.append(float(re.search(r"offset=([0-9.]+)", filter_complex).group(1)))
    assert offsets[0] == pytest.approx(4.94)
    assert offsets[1] == pytest.approx(8.88)
    assert output.read_bytes() == b"rendered"
