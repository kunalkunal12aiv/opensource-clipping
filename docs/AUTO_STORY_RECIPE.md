# Auto Story-Recipe Mode

Auto Story-Recipe creates editor-ready dialogue montages from **one episode
source**. The Studio asks AI to discover as many distinct narrative arcs as the
episode supports (up to eight), lets you choose which arcs to create, then
automatically schedules, validates, and renders each selected story separately.
The CLI remains a single-story workflow. Source moments may be reused between
different stories. Auto Story does not generate voice-over or upload videos.

The Studio asks for a story selection once, after discovery:

1. Faster-Whisper creates word timestamps and PyAnnote adds speaker IDs.
2. The selected AI provider discovers the distinct storylines the transcript
   supports. If you want another set, rerun discovery before continuing.
3. Select one or more storylines and continue. The backend builds and validates
   every selected story, requests fresh dialogue candidates when a schedule
   cannot fit, and renders the stories sequentially without further approval.

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
2. Choose a 60-, 90-, or 120-second target **per story** and the AI provider.
3. Wait for transcription and story discovery. The dashboard progress bar
   follows transcript audio progress; speaker identification is a separate step.
4. Select the storylines you want. All discovered arcs are selected initially;
   uncheck any you do not want. Choose **Rerun story discovery** for a fresh set,
   or **Create & render** to build the selected stories.
5. After selection, the backend automatically creates, validates, and renders
   each story. It retries beat-candidate generation up to three times when a
   candidate pool cannot meet the timing, spacing, span, or runtime constraints.
   The results include a separate MP4, caption sidecar, and manifest for each
   story, plus a combined plan manifest.

The page can be refreshed during processing or story selection; the dashboard job state
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

## CLI review and dashboard output

The storyline list is saved as `storyline_discovery.json`. In the CLI, beat
selection creates `story_recipe_review_<cache-key>.json`; edit the dialogue
start/end times to cached word boundaries, or add/remove beats, while the CLI
waits for review. The validator derives displayed dialogue and speaker IDs from
the cached transcript, so changes to free-form text do not change the source
audio. Invalid recipes are rejected with specific reasons; edit the same file
and rerun. Cached transcription, diarization, and AI responses are reused when
their source/model/settings keys still match. The dashboard automatically
creates and validates selected stories and does not ask users to review beats.
The combined word-and-speaker transcript is stored at
`outputs/story_cache/auto_story/<source-id>/combined_transcript.json`.

After rendering, the dashboard output folder contains:

- `story-1_<title>_story_montage.mp4` (and one corresponding file set per story)
  — rendered dialogue montage with karaoke captions, unless `--no-subs` was
  selected.
- `story-1_<title>_story_montage.ass` — rebased word-level captions for editor
  use.
- `story-1_<title>_manifest.json` — storyline summary, source/dialogue
  timestamps, transcript word IDs, speaker IDs, clip padding, and output paths.
- `story-1_recipe_<cache-key>.json` — the generated recipe manifest for the
  selected story.
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
- Scene-boundary detection, active-speaker cropping, Demucs, and TTS are not
  part of v1. The Studio retries beat-candidate generation up to three times;
  if the selected story still cannot satisfy the constraints, the job reports
  the failed story instead of rendering an invalid montage.

The CLI's two AI passes call only the provider chosen with `--ai-provider`.
The Studio makes one storyline-discovery request and one or more beat-planning
requests per selected story (up to three candidate rounds). Rerunning discovery
makes a fresh discovery request. Results are cached by transcript, prompt
version, model, selected storyline, candidate round, and recipe settings. Tests
mock transcription, diarization, rendering, and AI; running the test suite does
not use API credits.
