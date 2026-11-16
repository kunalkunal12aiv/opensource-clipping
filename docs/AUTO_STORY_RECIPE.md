# Auto Story-Recipe Mode

Auto Story-Recipe creates editor-ready dialogue montages from **one episode
source**. The Studio plans four or five distinct narrative arcs, schedules and
validates every story before review, then renders each approved story to its own
video. The CLI remains a single-story workflow. Source moments may be reused
between different stories. Auto Story does not generate voice-over or upload
videos.

The Studio workflow is staged so rendering cannot begin without reviewing the
complete plan:

1. Faster-Whisper creates word timestamps and PyAnnote adds speaker IDs.
2. The selected AI provider proposes four or five distinct storylines.
3. The provider selects dialogue beats for every storyline. The automatic
   scheduler checks beat lengths, source-time gaps, episode span, and runtime
   before showing all stories together.
4. Review the full plan, optionally adjust word-boundary timings, or reconstruct
   one story while preserving the other stories. Validate edits to the full plan.
5. Approve once to render the stories sequentially as separate videos.

The CLI still requires choosing one of three storylines, reviewing its recipe
JSON, and explicitly approving the single montage.

## Requirements

- FFmpeg and `ffprobe` available on `PATH`.
- The compatible PyAnnote 3.3.2 / PyTorch 2.5.1 / Torchaudio 2.5.1 stack
  installed from `requirements.txt`.
- A Hugging Face token in `HF_TOKEN`, with access to
  `pyannote/speaker-diarization-3.1`. Accept the model's terms on Hugging Face.
- The key for the explicitly selected AI provider: `GOOGLE_API_KEY` for
  Gemini, or `NVIDIA_API_KEY` for NVIDIA. NVIDIA uses
  `deepseek-ai/deepseek-v4.1-flash` by default.
- A local video or a source supported by the existing Story Clip downloader.

Put provider keys and `HF_TOKEN` in the git-ignored `.env` file in the project
root. Auto Story does not fall back to a different provider if the selected key
is missing.

The Auto Story CLI defaults to Faster-Whisper `small` on `cpu` with `int8`.
The Studio selects CUDA with `float16` when its backend reports a usable NVIDIA
GPU; otherwise, it uses CPU with `int8`. PyAnnote also uses CUDA when available.
On Kaggle, enable a GPU accelerator before starting the backend. GPU use can
reduce processing time, but model downloads and initial setup still take time.

## Prepare one episode source

Create a sources JSON file with exactly one source. For a local Windows file,
save the example below as `episode.json` in the repository root (the same
folder as `main.py`) and replace `local_path` with the full path to your video.
In JSON, escape each Windows backslash as `\\`.

```json
{
  "sources": [
    {
      "id": "episode",
      "name": "Episode 1",
      "platform": "local",
      "local_path": "C:\\Videos\\episode.mp4"
    }
  ]
}
```

## Use from the local Studio dashboard

Start the local API and Vite dashboard using the two-terminal commands in the
[README local dashboard section](../README.md#-run-the-web-dashboard-locally).
Open `http://127.0.0.1:5173`, select **Auto Story** in the sidebar, and:

1. Choose and upload one episode video.
2. Choose four or five stories, a 60-, 90-, or 120-second target **per story**,
   and the AI provider.
3. Wait for transcription, storyline discovery, and automatic scheduling of
   every story. The dashboard progress bar follows transcript audio progress;
   speaker identification is a separate step and can take time.
4. Review all narrative arcs and their dialogue beats together. Each schedule
   is prevalidated for beat length, source gaps, episode span, and target
   runtime. Use **Reconstruct story** on one story to replace only that arc, or
   adjust its word-boundary timings.
5. If you edited timings, select **Check the full plan**. Then select
   **Approve & render all stories** once. The backend renders each montage
   sequentially and provides a separate MP4, caption sidecar, manifest, and
   reviewed recipe for each story, plus a combined plan manifest.

The page can be refreshed during processing or review; the dashboard job state
and cached transcript remain on the local backend. Auto Story uses the existing
dashboard upload limit of 2 GB. Configure the selected provider key and
`HF_TOKEN` in **Settings** first. The backend automatically selects CUDA when
available.

## Use the local Auto Story dashboard with Kaggle

The Kaggle notebook runs the API, while the local React dashboard provides the
Auto Story page. Start the Kaggle backend using the
[Kaggle Studio guide](./KAGGLE_STUDIO.md), with Internet and a GPU enabled.
Provide `NGROK_AUTHTOKEN`, `HF_TOKEN`, and the selected AI provider key through
Kaggle Secrets. Keep the notebook server cell running and copy its HTTPS Public
URL.

On Windows, start only the dashboard locally; do not start a local API server.
In a PowerShell terminal at the repository root, set the tunnel URL as Vite's
API proxy target:

```powershell
$env:VITE_API_PROXY_TARGET = "https://your-current-ngrok-url"
npm --prefix web\dashboard run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173` and use **Auto Story** as usual. The Vite proxy
forwards dashboard API and upload requests to Kaggle and adds ngrok's browser
warning bypass header. Do not share the tunnel URL; it grants access to the
backend. Kaggle stores uploads, caches, and renders in its temporary runtime,
so download finished outputs before stopping or losing that session. The
backend reuses transcript and diarization caches for the same uploaded source.

## Run from the CLI

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe main.py --auto-story-mode `
  --sources-json episode.json `
  --target-duration 60
```

Use `--target-duration 90` or `120` for a longer montage. The default output
folder is `outputs/auto_story_recipe/<source-id>/`. Use
`--auto-story-output-dir` to change its parent.

## Review and output

The storyline list is saved as `storyline_discovery.json`. Beat selection
creates `story_recipe_review_<cache-key>.json`; edit the dialogue start/end
times to cached word boundaries, or add/remove beats, while the CLI waits for
review. The validator derives the displayed dialogue and speaker IDs from the
cached transcript, so changes to free-form text do not change the source audio.
Invalid recipes are rejected with specific reasons; edit the same file and
rerun. Cached transcription, diarization, and AI responses are reused when
their source/model/settings keys still match.
The combined word-and-speaker transcript is stored at
`outputs/story_cache/auto_story/<source-id>/combined_transcript.json`.

After approval, the dashboard output folder contains:

- `story-1_<title>_story_montage.mp4` (and one corresponding file set per story)
  — rendered dialogue montage with karaoke captions, unless `--no-subs` was
  selected.
- `story-1_<title>_story_montage.ass` — rebased word-level captions for editor
  use.
- `story-1_<title>_manifest.json` — storyline summary, source/dialogue
  timestamps, transcript word IDs, speaker IDs, clip padding, and output paths.
- `story-1_recipe_review_<cache-key>.json` — the editable reviewed recipe.
- `story_plan_manifest.json` — the full set of stories and output files.

The CLI continues to write one `story_montage.mp4`, one ASS sidecar, and one
`story_recipe_manifest.json` in its output folder.

## v1 guardrails

- One episode source; no multi-episode or multi-source plot stitching. Each
  dashboard story is a distinct narrative arc, but different stories may reuse
  source moments.
- Beat dialogue is 3.5–6 seconds by default. A punctuated sentence can extend
  up to 0.5 seconds beyond the configured maximum, with a hard 6.5-second cap.
- Words with zero-duration timestamps from Faster-Whisper are skipped with a
  terminal warning; invalid or reversed timestamps still stop the workflow.
- Each cut has 180 ms padding by default, with a 60 ms audio/video crossfade
  between cuts. Crossfade is configurable from 50–80 ms.
- Consecutive beats must be at least 30 source seconds apart and total output
  must be within 10% of the target duration. The CLI's first-to-last span must
  cover at least 40% of the episode. For a dashboard story whose arc covers a
  shorter source span, the same ratio is applied to that arc's span. These
  checks are configurable with `--min-gap-sec` and
  `--min-episode-span-ratio`.
- Scene-boundary detection, active-speaker cropping, Demucs, TTS, and automatic
  retries for invalid AI schedules are not part of v1. In the Studio, use
  **Reconstruct story** for a fresh alternative; in the CLI, correct the review
  JSON and rerun. Existing cached responses avoid repeated provider calls
  unless reconstruction is requested.

The CLI's two AI passes call only the provider chosen with `--ai-provider`.
The Studio makes one storyline-discovery request and one beat-planning request
per story; reconstructing a story makes a fresh discovery and beat-planning
request for that replacement. Results are cached by transcript, prompt version,
model, selected storyline, and recipe settings. Tests mock transcription,
diarization, rendering, and AI; running the test suite does not use API credits.
