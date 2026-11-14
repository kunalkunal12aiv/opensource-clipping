from pathlib import Path
from runpy import run_path

_HELPERS = run_path(str(
    Path(__file__).resolve().parents[1] / "clipping" / "studio" / "helpers.py"
))
build_subtitles_filter = _HELPERS["build_subtitles_filter"]


def test_subtitles_filter_quotes_escaped_windows_paths():
    actual = build_subtitles_filter(
        r"C:\Users\user\outputs\clip.ass",
        r"C:\Users\user\opensource-clipping\custom_fonts",
    )

    assert actual == (
        r"subtitles=filename='C\:\\Users\\user\\outputs\\clip.ass':"
        r"fontsdir='C\:\\Users\\user\\opensource-clipping\\custom_fonts'"
    )
