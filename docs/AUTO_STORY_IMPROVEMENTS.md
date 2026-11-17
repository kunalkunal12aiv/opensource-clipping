# Auto Story improvement backlog

This backlog is based on the current Auto Story page, API, worker, project
archive flow, and their tests. Priorities reflect user impact and likelihood of
avoiding wasted processing time. Status is updated as items are implemented.

## P0 — remove silent waiting and lost progress

| Improvement | Why it matters | Status |
|---|---|---|
| Show byte-level progress for episode upload and project restore; allow cancelling either transfer. | Videos can be 2 GB and portable projects can be much larger. A spinner alone looks frozen and offers no recovery. | Implemented |
| Keep job polling alive after temporary network errors and back off when the user is at a decision checkpoint. | The current poll loop stops after one failed fetch and otherwise rerenders the full page every two seconds. | Implemented |
| Make failure recovery stage-aware and retry from saved transcript/speaker checkpoints. | Provider or planning failures after diarization should not force repeated Whisper/PyAnnote work. | User retries and automatic restart recovery reuse durable checkpoints; interrupted render recovery remains manual |
| Keep export/import round trips safe, content-verified, and tied to actual source/cache files. | Project backup must restore the same plan and be safe to accept as an upload. | Implemented in prior changes; continue round-trip coverage |
| Preserve review checkpoints and resume jobs after backend restarts; allow an in-progress recovery backup. | A local restart should not discard user decisions or force Whisper to repeat after diarization; remote notebook resets can remove the backend disk. | Analysis and worker stage recovery plus analysis-stage backups implemented; remote runtime persistence still requires a downloaded backup |

## P1 — improve editability and control

| Improvement | Why it matters | Status |
|---|---|---|
| Let users name speakers and reassign transcript turns; add a bulk merge action and audio preview for turns. | Names clarify labels; turn reassignment fixes local diarization errors, while larger identity merges and listening controls improve review. | Naming, reassignment, bulk identity merge, and per-turn source audio preview implemented |
| Add beat preview controls that play the selected source interval with its context padding. | Timing edits currently expose timestamps and transcript words but no quick way to hear the result. | Implemented in legacy and multi-story timing review |
  | Expose practical render controls (Whisper model/device, pacing gap, captions, and output quality) with safe defaults and server validation. | Current dashboard choices are limited to target duration and AI provider. | Expected diarization speaker count, validated Whisper model choices, and configurable 15–60 second source gap implemented; device, captions, and output quality controls remain planned |
| Add a project list with search, status, retry, export, and delete actions. | Jobs persist in JSON, but users need a library to find older work and keep outputs organized. | Server-side search, compact paged listing, status, open/retry, and export implemented; delete/retention planned |
| Let users rerun planning for one failed storyline while keeping other plans and completed outputs. | Current recovery can restart discovery or retry a whole selected set. | Implemented; retry receives fresh planning candidates and updates the project manifest |

## P2 — improve throughput and supportability

| Improvement | Why it matters | Status |
|---|---|---|
| Avoid repeated hashing/probing of the same episode within a workflow and report cache reuse explicitly. | Large sources are currently hashed again when later stages rebuild their source context. | Job-local hash reuse and transcript/diarization/combined cache reporting implemented |
| Add local diagnostics export with stage timings, cache hits, and safe error categories. | Makes real Kaggle/provider failures easier to distinguish from code failures without exposing credentials. | Implemented as a per-project safe JSON download |
| Add bounded concurrency-safe project writes and retention/cleanup controls. | Large source uploads, project archives, and output files need predictable disk usage. | Persistence health reporting and protection from overwriting unreadable job snapshots implemented; aggregate quotas and retention controls planned |
| Add live end-to-end smoke checks for configured provider, Hugging Face access, GPU, and representative video/render paths. | Unit tests mock external services and cannot prove a runtime is configured correctly. | Requires a configured runtime |

## Implemented in this pass

- Upload/restore progress and cancellation controls, with client-side size checks.
- Poll retry with exponential backoff and slower polling during user review.
- Skip React job-state updates when the persisted job revision has not changed.
- Add paged transcript review with saved per-turn speaker reassignment and unassignment; reuse those corrected labels through discovery, planning, validation, and rendering.
- Validate speaker edits against transcript contents when importing a project backup.
- Reuse a verified source content signature during later stages of the same job when the upload's size and modification time have not changed.
- Add a searchable saved-project library backed by a compact API response that omits worker logs and bulky job details.
- Retry a single failed storyline while retaining successful outputs; protect against retrying with a changed source and preserve checkpoint status on cancellation.
- Stream source audio using byte ranges and audition transcript turns at their detected timestamps before applying speaker corrections.
- Merge multiple diarization labels into one canonical person, validate mappings for cycles/unknown IDs, and apply merged identities consistently through discovery, planning, cache keys, and project restore.
- Preview edited dialogue beats from the source episode with the configured padding interval before validating and rendering.
- Report transcript, diarization, and combined-cache reuse; persist bounded stage timing history and provide a downloadable diagnostics summary that excludes transcript text, provider responses, keys, and local paths.
- Report local job-store read/write failures through health, project-library, job-response, and diagnostics APIs; protect an unreadable existing snapshot from being silently replaced.
- Build saved-project summaries directly under the store lock and search/page them server-side, avoiding full deep copies of job logs and render metadata during library loads.
- Allow automatic or explicitly specified PyAnnote speaker count; retain it in portable projects and let the existing content-addressed cache rerun diarization without repeating Whisper when only that count changes.
- Expose a validated transcription model choice and persist it through project backups, with the existing transcript cache keyed by the selected model.
