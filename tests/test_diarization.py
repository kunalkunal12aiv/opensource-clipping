"""The shared CLI/API audio extractor must never unlink its source."""

import sys
import types

import pytest

from clipping import diarization


def test_missing_gated_model_access_reports_actionable_error(monkeypatch):
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    with pytest.raises(RuntimeError, match="accepted the model terms"):
        diarization.run_diarization("audio.wav", "hf-token")


def test_missing_segmentation_access_reports_both_gated_models(monkeypatch):
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AttributeError("'NoneType' object has no attribute 'eval'")

    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    with pytest.raises(RuntimeError, match="pyannote/segmentation-3.0"):
        diarization.run_diarization("audio.wav", "hf-token")


def test_same_path_is_rejected_before_touching_video(tmp_path, monkeypatch):
    source = tmp_path / "podcast.webm"
    source.write_bytes(b"original video")
    calls = []
    monkeypatch.setattr(diarization.subprocess, "run", lambda *a, **k: calls.append(a))

    with pytest.raises(ValueError, match="different"):
        diarization.extract_audio(str(source), str(source))

    assert source.read_bytes() == b"original video"
    assert calls == []


def test_cli_signature_and_ffmpeg_arguments_remain_compatible(tmp_path, monkeypatch):
    source = tmp_path / "podcast.mp4"
    output = tmp_path / "podcast_audio.wav"
    source.write_bytes(b"original video")
    calls = []
    monkeypatch.setattr(diarization.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    assert diarization.extract_audio(str(source), str(output)) == str(output)
    args, options = calls[0]
    assert args[args.index("-i") + 1] == str(source)
    assert args[-1] == str(output)
    assert options["check"] is True
    assert source.read_bytes() == b"original video"
