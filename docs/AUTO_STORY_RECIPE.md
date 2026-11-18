# Auto Story-Recipe Mode

Auto Story-Recipe creates editor-ready dialogue montages from **one episode
source**. The default **Auto Story** workflow identifies one promising narrative,
selects connected dialogue, and renders one story. **Choose a Story** remains an
advanced workflow that discovers up to eight storylines and lets you create
multiple stories. A storyline is a candidate arc; the story planner must still
assemble dialogue that makes sense without episode context. The CLI remains a
single-story workflow. Source moments may be reused between different stories.
Auto Story does not generate voice-over or upload videos.

The Studio asks for a story selection once, after discovery:

1. Faster-Whisper creates word timestamps. PyAnnote speaker identification is
   optional and can be enabled when speaker labels are useful.
2. In Auto Story mode, the selected provider identifies the strongest supported
   narrative and plans one connected story. In Choose a Story mode, it discovers
   several storylines for selection.
3. The planner favors self-contained dialogue, narrative continuity, distinct
   story information, and a strong payoff. Requested duration and spacing guide
   the edit; a valid best-effort draft can still be reviewed and exported when
   those preferences cannot be met.
4. Vertical output uses the standard face-tracked crop where possible, while
   preserving the original source audio. If tracking is unavailable, the renderer
   falls back to a centered crop.

The CLI still requires choosing one of three storylines, reviewing its recipe
JSON, and explicitly approving the single montage.

## Requirements

- FFmpeg and `ffprobe` available on `PATH`.
- The compatible PyAnnote 3.3.2 / PyTorch 2.5.1 / Torchaudio 2.5.1 stack
  installed from `requirements.txt` only when speaker diarization is enabled.
- A Hugging Face token in `HF_TOKEN`, with access to
  `pyannote/speaker-diarization-3.1`, only when speaker diarization is enabled.
  Accept the model's terms on Hugging Face. Story discovery and planning work
  without diarization; speaker labels are optional context.
- The key for the explicitly selected AI provider: `GOOGLE_API_KEY` for
  Gemini, `NVIDIA_API_KEY` for NVIDIA, or `FREELLMAPI_API_KEY` for FreeLLMAPI.
  FreeLLMAPI uses its OpenAI-compatible chat endpoint and `auto` model routing;
  configure `FREELLMAPI_BASE_URL` to a URL reachable from the backend (defaults
  to `http://127.0.0.1:3001/v1`). NVIDIA uses `deepseek-ai/deepseek-v4-pro` by
  default.
- A local video or a source supported by the existing Story Clip downloader.

Put the selected provider key in the git-ignored `.env` file in the project
root. Add `HF_TOKEN` only when speaker diarization is enabled. Auto Story does
not fall back to a different provider if the selected key is missing.

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

The start page lists saved Auto Story projects from the connected backend. Search
by project name, source filename, job ID, or status; open a project to resume or
retry it, or download its backup. The listing endpoint returns compact project
summaries rather than full job logs and progress payloads. Search runs on the
backend, and the library loads projects in pages so long histories stay responsive.

1. Choose and upload one episode video.
   Upload progress is shown and the transfer can be cancelled; the page also
   rejects files over the 2 GB server limit before uploading.
2. Choose **Auto Story** to make one story from the strongest supported arc, or
   **Choose a Story** to discover several arcs and select one or more. Choose a
   60-, 90-, or 120-second target **per story**, the preferred minimum gap
   between source moments (30 seconds by default), the AI provider, and the
   transcription model. Turn on speaker diarization only when speaker labels
   are useful; expected speaker count is available when it is enabled. **Small**
   is the balanced transcription default; lighter models trade accuracy for
   speed, and larger models need more memory and time. Device/compute remain
   automatic. Leave speaker count on **Automatic** unless you know the exact
   number of voices; forcing a count changes the diarization cache and reruns
   only that stage when the episode transcript is reusable.
3. Wait for transcription and story discovery. The dashboard progress bar
   follows transcript audio progress; optional speaker identification is a
   separate step. Auto Story proceeds directly to one best story. Choose a
   Story opens the storyline-selection step.
4. In Choose a Story mode, select the arcs to create. All discovered arcs are
   selected initially; uncheck any you do not want. Choose **Rerun story
   discovery** for a fresh set, or **Create & render** to build the selected
   stories. To make manual timing changes, enable **Let me review and adjust
   dialogue timing before rendering** before continuing. The backend then creates
   and validates the plans, opens the word-boundary timing editor, and waits for
   you to check the edited plan and approve rendering.
5. Auto Story automatically creates, validates, and renders its single story.
   Choose a Story does the same for each selected arc. With manual review
   enabled, render after reviewing and validating the edited plans. The backend
   retries beat-candidate generation with progressively broader candidates.
   The final candidate pool is also scheduled as a best-effort draft: target
   runtime, minimum gap, episode span, and preferred beat lengths no longer
   prevent export when usable dialogue candidates exist. If no candidate pool
   or renderable dialogue exists, that story is reported as failed; successful
   renders are not discarded. The results include a separate
   MP4, caption sidecar, and manifest for each rendered story, plus a combined
   plan manifest that records skipped storylines.

The storyline selection screen explains PyAnnote's `SPEAKER_00` style labels
and lets you assign optional speaker names. Saving names reruns discovery from
the saved analysis and passes those names as context to the AI. Open **Review &
correct transcript speaker turns** to inspect the word transcript in pages and
reassign a complete neighboring dialogue turn to another detected speaker or
mark it unassigned. Use **Play turn** to listen to the corresponding source
interval before changing its label; playback streams only the requested media
ranges as the browser buffers them. Save these edits before selecting stories. They are stored
in the project and applied consistently during discovery, planning, validation,
and rendering. Use **Merge into** when PyAnnote assigned two labels to the same
person; the merge applies to every matching transcript word, and the remaining
label becomes the canonical speaker. Speaker names provide context; turn
reassignment changes transcript labels locally.

During manual timing review, use **Preview beat with context** to audition the
current dialogue boundaries plus the configured padding before validating the
plan.

The page can be refreshed during processing or story selection; the dashboard job state
and cached transcript remain on the local backend. Job metadata is persisted in
`outputs/jobs.json`; rendered files are kept under `outputs/<job-id>/`, and the
uploaded source and reusable transcript/speaker analysis remain on the backend
disk. If the backend process restarts mid-stage, the job is marked failed and
use **Retry from saved transcript and speakers** to retry story discovery from
the saved analysis when the source/cache remain available. Failures while the
job is waiting for story selection can use **Rerun story discovery** directly.
Transient AI provider errors receive bounded automatic retries that honor the
provider's retry delay. If discovery still fails, choose another configured AI
provider and retry from the saved transcript; diarization and transcription do
not need to run again. Discovery also requests one corrected response when
returned story ranges fail transcript validation. A provider error is shown as
a retryable failure, not as a successful discovery with zero storylines.
The dashboard also checks whether the local job index is saving. If it reports
storage trouble, download portable backups for projects you need to keep. An
unreadable existing `jobs.json` is protected from automatic replacement; recover
the file and restart the backend before relying on local persistence again.
When a local backend restarts during transcription, diarization, or storyline
discovery, it automatically resumes discovery from the saved source and reuses
any complete transcript or speaker cache. It also resumes saved planning,
single story selection, reconstruction, failed story retries, and rendering.
Completed story files are retained and checked before the next story resumes.
You can download a recovery backup during analysis; import it after a remote
runtime reset. A backup is blocked while rendering so it cannot capture a
partially written video.
If only one selected story fails to plan or render, use **Retry** beside that
failed storyline. The retry reuses saved transcript and speaker corrections,
requests a fresh planning candidate round, preserves completed story files, and
updates the project manifest. It is available only while the project source and
saved analysis still match; cancellation returns to the prior checkpoint.
Auto Story uses the existing dashboard upload limit of 2 GB. Configure the
selected provider key in **Settings** first. Configure `HF_TOKEN` there only
when speaker diarization is enabled. The backend automatically selects CUDA
when available.

### Save or move an Auto Story project

At a safe checkpoint (storyline selection, manual plan review, failure, or
completion), choose **Download project backup**. The ZIP contains the source video, saved
transcript and speaker analysis, story plans, generated videos/captions, and
manifests. Keep it outside a temporary Kaggle runtime if it must survive a
runtime reset. Use **Restore a saved project** on the Auto Story start page to
import it into another backend; restored results and available analysis are
attached to a new job ID. Completed projects with their source and transcript
cache can use **Edit plans & render again** to reopen the saved word timings.
The restore screen shows upload progress, can cancel the transfer, and rejects
archives over 12 GB before sending them. Project archives can be up to 12 GB compressed and
16 GB expanded. Do not share an archive unless you intend to share its source
episode and generated files.

## Use the local Auto Story dashboard with Kaggle

The Kaggle notebook runs the API, while the local React dashboard provides the
Auto Story page. Start the Kaggle backend using the
[Kaggle Studio guide](./KAGGLE_STUDIO.md), with Internet and a GPU enabled.
Provide `NGROK_AUTHTOKEN` and the selected AI provider key through Kaggle
Secrets. Add `HF_TOKEN` only if you enable speaker diarization. Keep the
notebook server cell running and copy its HTTPS Public URL.

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
so download a project backup before stopping or losing that session. Within the same
backend project directory, transcript words and speaker turns are autosaved
under `outputs/story_cache/auto_story/` and reused across jobs, server restarts,
and re-uploads of byte-identical video files. Kaggle runtime reset removes this
local cache unless you save the notebook output or otherwise persist the
`outputs/story_cache/` folder.

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
creates and validates selected stories by default. The optional review choice
exposes the word-boundary timing editor for multi-story plans.
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
- Consecutive beats normally target at least 30 source seconds apart. The Studio
  first tries the configured schedule, then a wider 15-second fallback, and
  finally a best-effort schedule that treats runtime, gap, episode span, and beat
  length as preferences. The actual settings and best-effort status are recorded
  in the story manifest. A short draft remains exportable when candidates exist.
  The CLI's first-to-last span must
  cover at least 40% of the episode. For a dashboard story whose arc covers a
  shorter source span, the same ratio is applied to that arc's span. These
  checks are configurable with `--min-gap-sec` and
  `--min-episode-span-ratio`.
- Scene-boundary detection, active-speaker cropping, Demucs, and TTS are not
  part of v1. If no usable dialogue candidate pool exists for a selected story,
  the Studio records that story as failed and continues with the others. The job
  fails only when no story can produce any usable output.

Transcript and diarization caches are keyed by video content rather than the
random filename assigned to each upload. When unchanged prior upload files and
their filename-based caches are still present, Auto Story verifies the video
bytes and migrates valid transcript and speaker-turn caches automatically. If
those older files are unavailable or have changed, the next run computes the
missing stages once; later identical uploads reuse them.

The CLI's two AI passes call only the provider chosen with `--ai-provider`.
The Studio makes one storyline-discovery request and up to three beat-planning
requests per selected story. Rerunning discovery
makes a fresh discovery request. Results are cached by transcript, prompt
version, model, selected storyline, candidate round, and recipe settings. Tests
mock transcription, diarization, rendering, and AI; running the test suite does
not use API credits.
