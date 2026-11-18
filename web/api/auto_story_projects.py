"""Portable, self-contained Auto Story project archives."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, UploadFile

from clipping import auto_story as pipeline

from . import config_adapter, store
from .models import JobStatus

PROJECT_ROOT = config_adapter.PROJECT_ROOT
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAX_PROJECT_UPLOAD_BYTES = 12 * 1024**3
MAX_PROJECT_EXPANDED_BYTES = 16 * 1024**3
CHUNK_SIZE = 1024 * 1024


def _job_output_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise HTTPException(status_code=400, detail="Invalid Auto Story job ID.")
    root = OUTPUTS_DIR.resolve()
    path = (root / job_id).resolve()
    if path.parent != root:
        raise HTTPException(status_code=400, detail="Invalid project output path.")
    return path


def _safe_filename(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or any(char in value for char in "/\\:\x00")
    ):
        raise HTTPException(status_code=400, detail="Project contains an invalid filename.")
    return value


def _source_file(job: dict) -> Path:
    filename = job.get("upload_filename")
    try:
        path = Path(config_adapter.resolve_upload_path(filename))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="The episode source is no longer available.") from exc
    if not path.is_file():
        raise HTTPException(status_code=409, detail="The episode source is no longer available.")
    return path


def export_project(job_id: str) -> tuple[Path, str]:
    job = store.get_job(job_id)
    if job is None or not job.get("config", {}).get("auto_story_mode"):
        raise HTTPException(status_code=404, detail="Auto Story project not found.")
    if job.get("status") == JobStatus.RENDERING.value:
        raise HTTPException(
            status_code=409,
            detail="Wait for rendering to finish before exporting a consistent project backup.",
        )

    source = _source_file(job)
    source_id = (job.get("auto_story") or {}).get("source_id")
    content_id = f"episode-{pipeline._file_signature(str(source))['content_sha256'][:24]}"
    if (
        isinstance(source_id, str)
        and re.fullmatch(r"episode-[a-f0-9]{24}", source_id)
        and source_id != content_id
    ):
        raise HTTPException(status_code=409, detail="The episode file changed since this project was analyzed.")
    source_id = content_id
    cache_dir = OUTPUTS_DIR / "story_cache" / "auto_story" / source_id
    output_dir = _job_output_dir(job_id)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fd, archive_path = tempfile.mkstemp(prefix="auto-story-project-", suffix=".zip", dir=OUTPUTS_DIR)
    os.close(fd)
    try:
        project_data = {
            "format": "opensource-clipping-auto-story",
            "schema_version": 1,
            "source_id": source_id,
            "upload_filename": job.get("upload_filename"),
            "config": {
                key: job.get("config", {}).get(key)
                for key in (
                    "target_duration_sec", "min_gap_sec", "diarization_speakers",
                    "diarization_enabled", "workflow_mode",
                    "ai_provider", "source_name",
                )
            },
            "status": job.get("status"),
            "auto_story": job.get("auto_story") or {},
        }
        project_data["config"]["whisper_model"] = (
            job.get("config", {}).get("auto_story_whisper_model")
            or job.get("config", {}).get("whisper_model")
            or "small"
        )
        if type(project_data["config"].get("diarization_enabled")) is not bool:
            project_data["config"]["diarization_enabled"] = True
        if project_data["config"].get("workflow_mode") not in {"auto", "choose"}:
            project_data["config"]["workflow_mode"] = "auto"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr("project.json", json.dumps(project_data, ensure_ascii=False, indent=2))
            archive.write(source, f"source/{_safe_filename(source.name)}", compress_type=zipfile.ZIP_STORED)
            if output_dir.is_dir():
                for path in sorted(output_dir.iterdir()):
                    if path.is_file() and not path.name.startswith("."):
                        archive.write(path, f"outputs/{_safe_filename(path.name)}", compress_type=zipfile.ZIP_STORED)
            if cache_dir.is_dir():
                for path in sorted(cache_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() == ".json":
                        archive.write(path, f"cache/{_safe_filename(path.name)}", compress_type=zipfile.ZIP_DEFLATED)
    except Exception:
        Path(archive_path).unlink(missing_ok=True)
        raise

    project_name = re.sub(r"[^A-Za-z0-9._-]+", "_", (job.get("auto_story") or {}).get("source_name") or "episode")[:80]
    return Path(archive_path), f"{project_name or 'episode'}_auto_story_project.zip"


def _valid_member(name: str) -> tuple[str, str] | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return None
    if name == "project.json":
        return "manifest", "project.json"
    if len(path.parts) != 2 or path.parts[0] not in {"source", "outputs", "cache"}:
        return None
    filename = path.parts[1]
    if (
        not filename or filename in {".", ".."}
        or any(ch in filename for ch in '<>:"|?*\x00')
        or any(ord(ch) < 32 for ch in filename)
    ):
        return None
    if path.parts[0] == "source" and Path(filename).suffix.lower() not in {
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts",
    }:
        return None
    if path.parts[0] == "cache" and Path(filename).suffix.lower() != ".json":
        return None
    return path.parts[0], filename


def _copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=CHUNK_SIZE)


async def import_project(file: UploadFile) -> str:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Choose an Auto Story project .zip archive.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUTS_DIR / f".auto-story-import-{uuid.uuid4().hex}.zip"
    total = 0
    try:
        with temp_path.open("xb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_PROJECT_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Project archive exceeds the 12 GB import limit.")
                target.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Project archive is empty.")
        return _restore_archive(temp_path)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="This is not a valid project archive.") from exc
    except HTTPException:
        raise
    except (KeyError, RuntimeError, EOFError, NotImplementedError, zipfile.LargeZipFile) as exc:
        raise HTTPException(status_code=400, detail="Project archive is damaged or could not be restored.") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()


def _restore_archive(archive_path: Path) -> str:
    with zipfile.ZipFile(archive_path, "r") as archive:
        selected: dict[str, tuple[zipfile.ZipInfo, str]] = {}
        expanded = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            # Reject Unix symlinks and any path outside the documented archive layout.
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and (mode & 0o170000) == 0o120000:
                raise HTTPException(status_code=400, detail="Project archives cannot contain symbolic links.")
            kind_name = _valid_member(info.filename)
            if kind_name is None:
                raise HTTPException(status_code=400, detail="Project archive contains an unexpected or unsafe path.")
            kind, name = kind_name
            key = f"{kind}/{name}"
            if key in selected:
                raise HTTPException(status_code=400, detail="Project archive contains duplicate files.")
            expanded += info.file_size
            if expanded > MAX_PROJECT_EXPANDED_BYTES:
                raise HTTPException(status_code=413, detail="Expanded project exceeds the 16 GB import limit.")
            if info.file_size > 4 * 1024**3:
                raise HTTPException(status_code=413, detail="A project file exceeds the 4 GB per-file limit.")
            selected[key] = (info, kind)

        manifest_entry = selected.get("manifest/project.json")
        if manifest_entry is None or manifest_entry[0].file_size > 64 * 1024**2:
            raise HTTPException(status_code=400, detail="Project manifest is missing or too large.")
        final_source = None
        output_dir = None
        try:
            manifest = json.loads(archive.read(manifest_entry[0]))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Project manifest is invalid JSON.") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != "opensource-clipping-auto-story"
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("auto_story"), dict)
            or not isinstance(manifest.get("config"), dict)
        ):
            raise HTTPException(status_code=400, detail="Unsupported Auto Story project format.")

        source_entries = [(key, item) for key, item in selected.items() if key.startswith("source/")]
        if len(source_entries) != 1:
            raise HTTPException(status_code=400, detail="Project must contain exactly one episode source video.")
        source_name = source_entries[0][0].split("/", 1)[1]
        source_info = source_entries[0][1][0]
        settings = manifest["config"]
        if (
            settings.get("ai_provider", "gemini") not in {"gemini", "nvidia", "freellmapi"}
            or settings.get("workflow_mode", "auto") not in {"auto", "choose"}
            or type(settings.get("diarization_enabled", True)) is not bool
            or type(settings.get("target_duration_sec", 60)) is not int
            or not 60 <= settings.get("target_duration_sec", 60) <= 120
            or type(settings.get("min_gap_sec", 30)) is not int
            or not 15 <= settings.get("min_gap_sec", 30) <= 60
            or (settings.get("whisper_model") or "small") not in {
                "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo",
            }
            or (
                settings.get("diarization_speakers") is not None
                and (
                    type(settings.get("diarization_speakers")) is not int
                    or not 1 <= settings.get("diarization_speakers") <= 16
                )
            )
            or not isinstance(settings.get("source_name"), (str, type(None)))
            or len(settings.get("source_name") or "") > 180
        ):
            raise HTTPException(status_code=400, detail="Project settings are invalid.")
        imported_state = manifest["auto_story"]
        if (
            not isinstance(imported_state.get("storylines", []), list)
            or len(imported_state.get("storylines", [])) > 8
            or not isinstance(imported_state.get("stories", []), list)
            or len(imported_state.get("stories", [])) > 8
            or any(not isinstance(item, dict) for item in imported_state.get("stories", []))
        ):
            raise HTTPException(status_code=400, detail="Project story data is invalid.")
        speaker_names = imported_state.get("speaker_names", {})
        speaker_corrections = imported_state.get("speaker_corrections", [])
        speaker_merges = imported_state.get("speaker_merges", {})
        speaker_ids = imported_state.get("speaker_ids", [])
        if (
            not isinstance(speaker_ids, list)
            or len(speaker_ids) > 16
            or any(
                not isinstance(speaker_id, str)
                or not re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]{1,32}", speaker_id)
                for speaker_id in speaker_ids
            )
            or not isinstance(speaker_names, dict)
            or len(speaker_names) > 16
            or any(
                not isinstance(speaker_id, str)
                or not re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]{1,32}", speaker_id)
                or not isinstance(name, str)
                or len(name) > 60
                for speaker_id, name in speaker_names.items()
            )
            or not isinstance(speaker_corrections, list)
            or len(speaker_corrections) > 1000
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("start_word_id"), str)
                or not re.fullmatch(r"W\d{6,}", item.get("start_word_id", ""))
                or not isinstance(item.get("end_word_id"), str)
                or not re.fullmatch(r"W\d{6,}", item.get("end_word_id", ""))
                or (
                    item.get("speaker_id") is not None
                    and (
                        not isinstance(item.get("speaker_id"), str)
                        or not re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]{1,32}", item["speaker_id"])
                    )
                )
                for item in speaker_corrections
            )
            or not isinstance(speaker_merges, dict)
            or len(speaker_merges) > 16
            or any(
                not isinstance(source, str)
                or not re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]{1,32}", source)
                or not isinstance(target, str)
                or not re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]{1,32}", target)
                or source == target
                for source, target in speaker_merges.items()
            )
            or not set(speaker_names).issubset(speaker_ids)
            or not set(speaker_merges).issubset(speaker_ids)
            or not set(speaker_merges.values()).issubset(speaker_ids)
        ):
            raise HTTPException(status_code=400, detail="Project speaker edits are invalid.")
        output_names = {
            key.split("/", 1)[1]
            for key in selected
            if key.startswith("outputs/")
        }
        story_ids = set()
        for story in imported_state.get("stories", []):
            story_id = story.get("story_id")
            recipe_filename = story.get("recipe_filename")
            if (
                not isinstance(story_id, str)
                or not re.fullmatch(r"story-[1-8]", story_id)
                or story_id in story_ids
                or not isinstance(recipe_filename, str)
                or _safe_filename(recipe_filename) != recipe_filename
                or recipe_filename not in output_names
                or not isinstance(story.get("storyline"), dict)
                or not isinstance(story["storyline"].get("title"), str)
            ):
                raise HTTPException(status_code=400, detail="Project contains an invalid story plan.")
            story_ids.add(story_id)
            if story.get("manifest_file") is not None:
                manifest_filename = story["manifest_file"]
                if (
                    not isinstance(manifest_filename, str)
                    or _safe_filename(manifest_filename) != manifest_filename
                    or manifest_filename not in output_names
                ):
                    raise HTTPException(status_code=400, detail="Project contains an invalid story manifest reference.")
        legacy_recipe = imported_state.get("recipe_filename")
        if legacy_recipe is not None and (
            not isinstance(legacy_recipe, str)
            or _safe_filename(legacy_recipe) != legacy_recipe
            or legacy_recipe not in output_names
        ):
            raise HTTPException(status_code=400, detail="Project contains an invalid recipe reference.")

        source_temp = UPLOAD_DIR / f".restore-{uuid.uuid4().hex}{Path(source_name).suffix.lower()}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        created_cache_files: list[Path] = []
        try:
            _copy_member(archive, source_info, source_temp)
            signature = pipeline._file_signature(str(source_temp))
            source_id = f"episode-{signature['content_sha256'][:24]}"
            if manifest.get("source_id") != source_id:
                raise HTTPException(status_code=400, detail="Episode source does not match the project cache identity.")

            job_id = uuid.uuid4().hex[:12]
            upload_filename = f"{uuid.uuid4().hex[:12]}_{source_name}"
            final_source = UPLOAD_DIR / upload_filename
            source_temp.replace(final_source)
            output_dir = _job_output_dir(job_id)
            output_dir.mkdir(parents=True, exist_ok=False)
            for key, (info, kind) in selected.items():
                if kind == "outputs":
                    _copy_member(archive, info, output_dir / key.split("/", 1)[1])
                elif kind == "cache":
                    cache_dir = OUTPUTS_DIR / "story_cache" / "auto_story" / source_id
                    cache_file = cache_dir / key.split("/", 1)[1]
                    if not cache_file.exists():
                        created_cache_files.append(cache_file)
                        _copy_member(archive, info, cache_file)

            if speaker_corrections or speaker_merges:
                transcript_key = imported_state.get("transcript_key")
                transcript_cache = pipeline._read_cache(
                    str(OUTPUTS_DIR / "story_cache" / "auto_story" / source_id / "combined_transcript.json"),
                    transcript_key,
                ) if isinstance(transcript_key, str) else None
                if transcript_cache is None or not isinstance(transcript_cache.get("words"), list):
                    raise HTTPException(status_code=400, detail="Project speaker edits require the saved transcript cache.")
                from . import auto_story_worker

                try:
                    auto_story_worker._apply_speaker_corrections(
                        transcript_cache["words"], imported_state
                    )
                except RuntimeError as exc:
                    raise HTTPException(status_code=400, detail="Project speaker edits do not match the saved transcript.") from exc

            # Archives written before workflow_mode existed used the multi-story
            # selection screen whenever discovery returned multiple storylines.
            # Preserve that intent on restore instead of silently treating the
            # old project as a one-story Auto workflow.
            workflow_mode = manifest["config"].get("workflow_mode")
            if workflow_mode is None:
                workflow_mode = (
                    "choose"
                    if len(imported_state.get("storylines", [])) > 1
                    else "auto"
                )
            config = {
                "auto_story_mode": True,
                "workflow_mode": workflow_mode,
                "target_duration_sec": manifest["config"].get("target_duration_sec", 60),
                "min_gap_sec": manifest["config"].get("min_gap_sec", 30),
                "diarization_speakers": manifest["config"].get("diarization_speakers"),
                "diarization_enabled": manifest["config"].get("diarization_enabled", True),
                "ai_provider": manifest["config"].get("ai_provider", "gemini"),
                "source_name": manifest["config"].get("source_name") or source_name,
                "auto_story_whisper_model": manifest["config"].get("whisper_model") or "small",
                "auto_story_whisper_device": "cpu",
                "auto_story_whisper_compute_type": "int8",
            }
            state = manifest["auto_story"]
            state["source_id"] = source_id
            state["source_name"] = config["source_name"]
            state["output_files"] = []
            for path in sorted(output_dir.iterdir()):
                if path.is_file():
                    state["output_files"].append({
                        "filename": path.name,
                        "download_url": f"/api/outputs/{job_id}/{path.name}",
                    })
            for story in state.get("stories", []):
                story["output_files"] = [
                    entry for entry in state["output_files"]
                    if entry["filename"].startswith(f"{story.get('story_id')}_")
                    or story.get("manifest_file") == entry["filename"]
                    or story.get("recipe_filename") == entry["filename"]
                ]
            status = manifest.get("status")
            if status not in {
                JobStatus.COMPLETED.value,
                JobStatus.AWAITING_STORYLINE.value,
                JobStatus.AWAITING_RECIPE.value,
            }:
                status = JobStatus.AWAITING_STORYLINE.value
                state.update(
                    phase="awaiting_storyline",
                    message="Project restored. Rerun discovery or choose from the saved storylines.",
                )
            if status == JobStatus.AWAITING_RECIPE.value and state.get("stories"):
                transcript_key = state.get("transcript_key")
                try:
                    transcript_cache = pipeline._read_cache(
                        str(OUTPUTS_DIR / "story_cache" / "auto_story" / source_id / "combined_transcript.json"),
                        transcript_key,
                    ) if isinstance(transcript_key, str) else None
                except (OSError, ValueError):
                    transcript_cache = None
                if transcript_cache is None or not isinstance(transcript_cache.get("words"), list):
                    status = JobStatus.AWAITING_STORYLINE.value
                    state.update(
                        phase="awaiting_storyline",
                        message="Project restored, but its editable transcript cache is missing. Rerun discovery to rebuild analysis.",
                    )
            job_id = store.create_job(
                upload_filename=upload_filename,
                source="local",
                config=config,
                job_id=job_id,
            )
            store.update_job(job_id, auto_story=state, status=status, error=None)
            return job_id
        except Exception:
            source_temp.unlink(missing_ok=True)
            if final_source is not None:
                final_source.unlink(missing_ok=True)
            if output_dir is not None:
                shutil.rmtree(output_dir, ignore_errors=True)
            for cache_file in created_cache_files:
                cache_file.unlink(missing_ok=True)
            raise
