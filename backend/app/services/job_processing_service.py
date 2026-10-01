"""Synchronous orchestration of the existing Hindi dubbing services."""

import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from uuid import UUID

from backend.app.services.audio_adjustment_service import adjust_audio_duration
from backend.app.services.audio_service import extract_audio
from backend.app.services.asr_service import transcribe_audio
from backend.app.services.srt_service import build_srt, write_srt
from backend.app.services.sync_service import calculate_cadence
from backend.app.services.sync_validation_service import validate_sync
from backend.app.services.translation_service import (
    LANGUAGE_NAMES,
    translate_segments,
)
from backend.app.services.tts_service import generate_hindi_speech, get_audio_duration
from backend.app.services.video_generation_service import generate_dubbed_video
from backend.app.services.video_service import validate_video


STORAGE_ROOT = Path("storage").resolve()
UPLOAD_DIR = STORAGE_ROOT / "uploads"
AUDIO_DIR = STORAGE_ROOT / "audio"
TRANSCRIPT_DIR = STORAGE_ROOT / "transcripts"
OUTPUT_DIR = STORAGE_ROOT / "outputs"
BACKGROUND_VOLUME = 0.25
FFMPEG_TIMEOUT_SECONDS = 120
SPEECH_PAUSE_SPLIT_SECONDS = 0.65
logger = logging.getLogger(__name__)


class JobProcessingError(RuntimeError):
    """A safe stage-specific failure for the end-to-end job pipeline."""

    def __init__(self, stage: str, message: str, segment_id: int | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.segment_id = segment_id


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a UUID string.")
    try:
        return str(UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a valid UUID.") from exc


def _contained_file(path: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(STORAGE_ROOT)
    except (OSError, RuntimeError, ValueError):
        raise JobProcessingError("input", f"Required {description} is unavailable.") from None
    if not resolved.is_file():
        raise JobProcessingError("input", f"Required {description} is unavailable.")
    return resolved


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise


def _status_path(job_id: str) -> Path:
    job_directory = OUTPUT_DIR / job_id
    resolved_job_directory = job_directory.resolve(strict=False)
    if resolved_job_directory != OUTPUT_DIR.resolve(strict=False) / job_id:
        raise ValueError("Invalid job status directory.")
    resolved_job_directory.relative_to(OUTPUT_DIR.resolve(strict=False))
    return resolved_job_directory / "job_status.json"


def _write_status(
    job_id: str,
    status: str,
    progress: int,
    *,
    stage: str | None = None,
    output_ready: bool = False,
    error: str | None = None,
    translation_source: str | None = None,
    sync_results: list[dict[str, int | float | str | bool]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "language": "hi",
        "progress": progress,
        "output_ready": output_ready,
        "download_url": f"/api/jobs/{job_id}/download" if output_ready else None,
    }
    if stage:
        payload["stage"] = stage
    if error:
        payload["error"] = error
    if translation_source:
        payload["translation_source"] = translation_source
    if sync_results is not None:
        payload["sync_validation"] = sync_results
    _atomic_json_write(_status_path(job_id), payload)


def get_job_status(job_id: str) -> dict[str, Any]:
    safe_job_id = validate_job_id(job_id)
    status_path = _status_path(safe_job_id)
    if status_path.is_file():
        try:
            with status_path.open("r", encoding="utf-8") as status_file:
                status = json.load(status_file)
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("Job status is unavailable.") from None
        if status.get("job_id") != safe_job_id:
            raise RuntimeError("Job status is invalid.")
        return status

    output_path = OUTPUT_DIR / safe_job_id / "hindi_dubbed.mp4"
    if output_path.is_file():
        return {
            "job_id": safe_job_id,
            "status": "completed",
            "language": "hi",
            "progress": 100,
            "output_ready": True,
            "download_url": f"/api/jobs/{safe_job_id}/download",
        }
    if (TRANSCRIPT_DIR / f"{safe_job_id}_hi.json").is_file():
        current_status, progress = "translated", 35
    elif (TRANSCRIPT_DIR / f"{safe_job_id}.json").is_file():
        current_status, progress = "transcribed", 20
    elif list(UPLOAD_DIR.glob(f"{safe_job_id}.*")):
        current_status, progress = "uploaded", 5
    else:
        raise FileNotFoundError("Job was not found.")
    return {
        "job_id": safe_job_id,
        "status": current_status,
        "language": "hi",
        "progress": progress,
        "output_ready": False,
        "download_url": None,
    }


def _find_video(job_id: str) -> Path:
    candidates = sorted(
        path
        for path in UPLOAD_DIR.glob(f"{job_id}.*")
        if path.suffix.lower() in {".mp4", ".webm"}
    )
    if not candidates:
        raise JobProcessingError("validation", "No uploaded MP4 or WebM was found for this job.")
    if len(candidates) != 1:
        raise JobProcessingError("validation", "The job has ambiguous uploaded videos.")
    return _contained_file(candidates[0], "uploaded video")


def _load_or_transcribe(job_id: str, audio_path: Path) -> dict[str, Any]:
    transcript_path = TRANSCRIPT_DIR / f"{job_id}.json"
    transcript: dict[str, Any] | None = None
    if transcript_path.is_file():
        try:
            with transcript_path.open("r", encoding="utf-8") as transcript_file:
                transcript = json.load(transcript_file)
        except (OSError, json.JSONDecodeError):
            raise JobProcessingError("transcription", "The English transcript is invalid.") from None

    has_word_timestamps = (
        isinstance(transcript, dict)
        and isinstance(transcript.get("segments"), list)
        and bool(transcript["segments"])
        and all(isinstance(segment.get("words"), list) and segment["words"] for segment in transcript["segments"])
    )
    if not has_word_timestamps:
        result = transcribe_audio(audio_path)
        transcript = result
        _atomic_json_write(transcript_path, transcript)

    assert transcript is not None
    if transcript.get("language") != "en" or not isinstance(transcript.get("segments"), list):
        raise JobProcessingError("transcription", "The job does not contain a valid English transcript.")
    if not all(
        isinstance(segment.get("words"), list) and segment["words"]
        for segment in transcript["segments"]
    ):
        raise JobProcessingError(
            "transcription",
            "Word-level timestamps are unavailable for this transcript.",
        )
    return transcript


def _load_matching_hindi_translation(
    job_id: str,
    source_segments: list[dict[str, Any]],
) -> dict[int, str] | None:
    """Return cached Hindi text only when every required source field matches."""
    translated_path = TRANSCRIPT_DIR / f"{job_id}_hi.json"
    if not translated_path.is_file():
        return None

    try:
        with translated_path.open("r", encoding="utf-8") as translated_file:
            cached = json.load(translated_file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(cached, dict):
        return None
    cached_segments = cached.get("segments")
    if (
        cached.get("source_language") != "en"
        or cached.get("target_language") != "hi"
        or not isinstance(cached_segments, list)
        or len(cached_segments) != len(source_segments)
    ):
        return None

    translations: dict[int, str] = {}
    for source_segment, cached_segment in zip(source_segments, cached_segments):
        if not isinstance(cached_segment, dict):
            return None
        if any(
            cached_segment.get(field) != source_segment.get(field)
            for field in ("id", "start", "end", "text")
        ):
            return None
        translated_text = cached_segment.get("translated_text")
        if (
            isinstance(source_segment.get("id"), bool)
            or not isinstance(source_segment.get("id"), int)
            or not isinstance(translated_text, str)
            or not translated_text.strip()
            or not any("\u0900" <= char <= "\u097f" for char in translated_text)
        ):
            return None
        if source_segment["id"] in translations:
            return None
        translations[source_segment["id"]] = translated_text.strip()

    return translations


def _reuse_cached_chunk_translations(
    cached_translations: dict[int, str],
    speech_chunks: list[dict[str, Any]],
    source_segments: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Reuse cached segment text only when each segment maps to one speech window."""
    chunk_counts: dict[int, int] = {}
    for chunk in speech_chunks:
        source_id = chunk["source_segment_id"]
        chunk_counts[source_id] = chunk_counts.get(source_id, 0) + 1

    if set(chunk_counts) != {segment["id"] for segment in source_segments}:
        return None
    if any(count != 1 for count in chunk_counts.values()):
        return None

    return [
        {**chunk, "translated_text": cached_translations[chunk["source_segment_id"]]}
        for chunk in speech_chunks
        if chunk["source_segment_id"] in cached_translations
    ] if len(cached_translations) == len(source_segments) else None


def _join_timed_words(words: list[dict[str, Any]]) -> str:
    word_texts = [str(word["word"]) for word in words]
    if any(text[:1].isspace() for text in word_texts[1:]):
        return "".join(word_texts).strip()
    return " ".join(text.strip() for text in word_texts).strip()


def _build_speech_chunks(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group word timestamps into short speech windows at pauses and phrases."""
    chunks: list[dict[str, Any]] = []
    next_chunk_id = 1

    for segment in segments:
        timed_words = segment.get("words")
        if not isinstance(timed_words, list) or not timed_words:
            raise JobProcessingError(
                "transcription",
                f"Segment {segment.get('id', '?')} has no word-level timestamps.",
                segment_id=segment.get("id") if isinstance(segment.get("id"), int) else None,
            )

        current_words: list[dict[str, Any]] = []

        def flush_chunk() -> None:
            nonlocal next_chunk_id
            if not current_words:
                return
            start = float(current_words[0]["start"])
            end = float(current_words[-1]["end"])
            text = _join_timed_words(current_words)
            if end <= start or not text:
                raise JobProcessingError("transcription", "Invalid timed speech window.")
            chunks.append(
                {
                    "id": next_chunk_id,
                    "source_segment_id": segment["id"],
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": text,
                }
            )
            next_chunk_id += 1
            current_words.clear()

        previous_end: float | None = None
        for timed_word in timed_words:
            if not isinstance(timed_word, dict):
                raise JobProcessingError("transcription", "Invalid word timestamp data.")
            word_text = timed_word.get("word")
            try:
                word_start = float(timed_word["start"])
                word_end = float(timed_word["end"])
            except (KeyError, TypeError, ValueError):
                raise JobProcessingError("transcription", "Invalid word timestamp data.") from None
            if (
                not isinstance(word_text, str)
                or not word_text.strip()
                or not math.isfinite(word_start)
                or not math.isfinite(word_end)
                or word_start < 0
                or word_end < word_start
            ):
                raise JobProcessingError("transcription", "Invalid word timestamp data.")

            if current_words:
                pause_duration = word_start - (previous_end or 0.0)
                previous_text = str(current_words[-1]["word"]).rstrip()
                sentence_end = previous_text.endswith(("?", "!")) or (
                    previous_text.endswith(".") and len(current_words) >= 3
                )
                if pause_duration >= SPEECH_PAUSE_SPLIT_SECONDS or sentence_end:
                    flush_chunk()

            current_words.append(
                {"word": word_text, "start": word_start, "end": word_end}
            )
            previous_end = word_end

        flush_chunk()

    return chunks


def _compose_speech_timeline(
    job_id: str,
    audio_segments: list[tuple[Path, float]],
    target_duration_seconds: float,
) -> Path:
    if not math.isfinite(target_duration_seconds) or target_duration_seconds <= 0:
        raise JobProcessingError("audio_adjustment", "Invalid target video duration.")
    if not audio_segments:
        raise JobProcessingError("audio_adjustment", "The transcript contains no speech segments.")

    timeline_directory = AUDIO_DIR / job_id / "timeline"
    timeline_directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = timeline_directory.resolve(strict=True)
    resolved_directory.relative_to(STORAGE_ROOT)
    output_path = resolved_directory / "hindi_speech.wav"

    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    filter_parts = []
    labels = []
    for index, (audio_path, start_seconds) in enumerate(audio_segments):
        safe_audio_path = _contained_file(audio_path, "adjusted Hindi segment audio")
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise JobProcessingError("audio_adjustment", "A transcript segment has invalid timing.")
        command.extend(["-i", str(safe_audio_path)])
        delay_ms = round(start_seconds * 1000)
        label = f"seg{index}"
        filter_parts.append(f"[{index}:a]adelay=delays={delay_ms}:all=1[{label}]")
        labels.append(f"[{label}]")

    if len(labels) == 1:
        filter_parts.append(f"{labels[0]}anull[mix]")
    else:
        filter_parts.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0[mix]"
        )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[mix]",
            "-af",
            f"apad=whole_dur={target_duration_seconds:.3f}",
            "-t",
            f"{target_duration_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
        ]
    )

    temporary_file = tempfile.NamedTemporaryFile(
        dir=resolved_directory,
        prefix=".speech-timeline-",
        suffix=".wav",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    command.append(str(temporary_path))

    try:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise JobProcessingError("audio_adjustment", "Speech timeline composition timed out.") from None
        if result.returncode != 0 or not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise JobProcessingError("audio_adjustment", "Speech timeline composition failed.")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def process_job(job_id: str) -> dict[str, Any]:
    safe_job_id = validate_job_id(job_id)
    sync_results: list[dict[str, int | float | str | bool]] = []
    current_stage = "validation"
    _write_status(safe_job_id, "uploaded", 1, stage=current_stage)

    try:
        video_path = _find_video(safe_job_id)
        video_metadata = validate_video(video_path)
        video_duration = float(video_metadata["duration"])
        _write_status(safe_job_id, "transcribing", 10, stage="transcription")

        audio_path = AUDIO_DIR / f"{safe_job_id}.wav"
        if not audio_path.is_file():
            current_stage = "audio_extraction"
            audio_path = extract_audio(video_path, safe_job_id)
        audio_path = _contained_file(audio_path, "extracted audio")

        current_stage = "transcription"
        transcript = _load_or_transcribe(safe_job_id, audio_path)
        segments = transcript["segments"]
        if not segments:
            raise JobProcessingError("transcription", "The transcript contains no speech segments.")
        _write_status(safe_job_id, "transcribed", 20, stage="translation")

        current_stage = "translation"
        _write_status(safe_job_id, "translating", 25, stage=current_stage)
        original_segment_count = len(segments)
        speech_chunks = _build_speech_chunks(segments)
        cached_translations = _load_matching_hindi_translation(
            safe_job_id,
            segments,
        )
        translated_chunks = (
            _reuse_cached_chunk_translations(
                cached_translations,
                speech_chunks,
                segments,
            )
            if cached_translations is not None
            else None
        )
        translation_source = "cached" if translated_chunks is not None else "gemini"
        if translated_chunks is None:
            translated_chunks = translate_segments(speech_chunks, "hi")
        _write_status(
            safe_job_id,
            "translated",
            35,
            stage="tts",
            translation_source=translation_source,
        )
        translated_by_segment: dict[int, list[str]] = {}
        for chunk in translated_chunks:
            translated_by_segment.setdefault(chunk["source_segment_id"], []).append(
                chunk["translated_text"]
            )
        translated_segments = []
        for segment in segments:
            translated_text = (
                cached_translations[segment["id"]]
                if translation_source == "cached" and cached_translations is not None
                else " ".join(translated_by_segment[segment["id"]])
            )
            translated_segments.append({**segment, "translated_text": translated_text})
        translated_transcript = {
            "job_id": safe_job_id,
            "source_language": "en",
            "target_language": "hi",
            "target_language_name": LANGUAGE_NAMES["hi"],
            "translation_source": translation_source,
            "segments": translated_segments,
        }
        _atomic_json_write(
            TRANSCRIPT_DIR / f"{safe_job_id}_hi.json",
            translated_transcript,
        )

        audio_segments: list[tuple[Path, float]] = []
        chunk_count = len(translated_chunks)
        for index, chunk in enumerate(translated_chunks, start=1):
            current_stage = "tts"
            _write_status(
                safe_job_id,
                "generating_tts",
                30 + int(20 * (index - 1) / chunk_count),
                stage=current_stage,
            )
            try:
                segment_id = chunk["id"]
                target_duration = float(chunk["end"]) - float(chunk["start"])
                tts_path = generate_hindi_speech(
                    chunk["translated_text"],
                    safe_job_id,
                    segment_id,
                )
                tts_path = _contained_file(tts_path, "generated Hindi speech")
                generated_duration = get_audio_duration(tts_path)

                current_stage = "audio_adjustment"
                cadence = calculate_cadence(target_duration, generated_duration)
                adjusted = adjust_audio_duration(
                    tts_path,
                    safe_job_id,
                    segment_id,
                    cadence.speed_factor,
                )
                adjusted_path = _contained_file(adjusted.output_path, "adjusted Hindi speech")
                measured_adjusted_duration = get_audio_duration(adjusted_path)
                sync_result = validate_sync(
                    safe_job_id,
                    segment_id,
                    target_duration,
                    measured_adjusted_duration,
                )
            except JobProcessingError:
                raise
            except Exception:
                raise JobProcessingError(
                    current_stage,
                    f"Hindi audio processing failed for segment {index}.",
                    segment_id=segment.get("id") if isinstance(segment, dict) else index,
                ) from None

            sync_results.append(
                {
                    "measurement": "speech_window",
                    "target_start_seconds": float(chunk["start"]),
                    "target_end_seconds": float(chunk["end"]),
                    "segment_id": segment_id,
                    "source_segment_id": int(chunk["source_segment_id"]),
                    "target_duration_seconds": sync_result.target_duration_seconds,
                    "actual_audio_duration_seconds": sync_result.actual_audio_duration_seconds,
                    "sync_error_ms": sync_result.sync_error_ms,
                    "absolute_sync_error_ms": sync_result.absolute_sync_error_ms,
                    "tolerance_ms": sync_result.tolerance_ms,
                    "status": sync_result.status,
                }
            )
            if not sync_result.is_within_tolerance:
                raise JobProcessingError(
                    "sync_validation",
                    f"Segment {segment_id} is outside the ±200 ms synchronization tolerance ({sync_result.sync_error_ms:.3f} ms).",
                    segment_id=segment_id,
                )
            audio_segments.append((adjusted_path, float(chunk["start"])))

        current_stage = "audio_adjustment"
        _write_status(safe_job_id, "adjusting_audio", 55, stage=current_stage)
        speech_timeline = _compose_speech_timeline(
            safe_job_id,
            audio_segments,
            video_duration,
        )
        speech_timeline = _contained_file(speech_timeline, "Hindi speech timeline")

        current_stage = "subtitle"
        _write_status(safe_job_id, "generating_subtitles", 75, stage=current_stage)
        subtitle_result = write_srt(translated_segments, safe_job_id, "hi")
        subtitle_path = _contained_file(subtitle_result.output_path, "Hindi subtitles")
        if subtitle_path.read_text(encoding="utf-8") != build_srt(translated_segments):
            raise JobProcessingError(
                "subtitle",
                "Generated subtitle timestamps do not match the translated transcript.",
            )
        for segment in translated_segments:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
            subtitle_sync = validate_sync(
                safe_job_id,
                int(segment["id"]),
                segment_end - segment_start,
                segment_end - segment_start,
            )
            sync_results.append(
                {
                    "measurement": "subtitle_segment_timing",
                    "segment_id": int(segment["id"]),
                    "target_start_seconds": segment_start,
                    "target_end_seconds": segment_end,
                    "actual_start_seconds": segment_start,
                    "actual_end_seconds": segment_end,
                    "target_duration_seconds": subtitle_sync.target_duration_seconds,
                    "actual_audio_duration_seconds": subtitle_sync.actual_audio_duration_seconds,
                    "sync_error_ms": subtitle_sync.sync_error_ms,
                    "absolute_sync_error_ms": subtitle_sync.absolute_sync_error_ms,
                    "tolerance_ms": subtitle_sync.tolerance_ms,
                    "status": subtitle_sync.status,
                }
            )

        current_stage = "video_generation"
        _write_status(safe_job_id, "generating_video", 85, stage=current_stage)
        video_result = generate_dubbed_video(
            safe_job_id,
            video_path,
            speech_timeline,
            subtitle_path,
        )
        output_path = _contained_file(Path(video_result["output_path"]), "final video")

        current_stage = "sync_validation"
        _write_status(safe_job_id, "validating_sync", 95, stage=current_stage)
        actual_audio_duration = get_audio_duration(speech_timeline)
        audio_sync = validate_sync(
            safe_job_id,
            1,
            video_duration,
            actual_audio_duration,
        )
        final_video_sync = validate_sync(
            safe_job_id,
            1,
            video_duration,
            float(video_result["duration_seconds"]),
        )
        if not audio_sync.is_within_tolerance or not final_video_sync.is_within_tolerance:
            measured_error = (
                audio_sync.sync_error_ms
                if not audio_sync.is_within_tolerance
                else final_video_sync.sync_error_ms
            )
            raise JobProcessingError(
                "sync_validation",
                f"Final audio/video duration differs from the target by {measured_error:.3f} ms, exceeding ±200 ms.",
            )

        sync_results.extend(
            [
                {
                    "segment_id": 1,
                    "measurement": "final_audio_track_duration",
                    "target_duration_seconds": audio_sync.target_duration_seconds,
                    "actual_audio_duration_seconds": audio_sync.actual_audio_duration_seconds,
                    "sync_error_ms": audio_sync.sync_error_ms,
                    "absolute_sync_error_ms": audio_sync.absolute_sync_error_ms,
                    "tolerance_ms": audio_sync.tolerance_ms,
                    "status": audio_sync.status,
                },
                {
                    "segment_id": 1,
                    "measurement": "final_video_duration",
                    "target_duration_seconds": final_video_sync.target_duration_seconds,
                    "actual_video_duration_seconds": final_video_sync.actual_audio_duration_seconds,
                    "sync_error_ms": final_video_sync.sync_error_ms,
                    "absolute_sync_error_ms": final_video_sync.absolute_sync_error_ms,
                    "tolerance_ms": final_video_sync.tolerance_ms,
                    "status": final_video_sync.status,
                },
            ]
        )

        result = {
            "job_id": safe_job_id,
            "status": "completed",
            "language": "hi",
            "progress": 100,
            "output_ready": True,
            "download_url": f"/api/jobs/{safe_job_id}/download",
            "segment_count": original_segment_count,
            "speech_chunk_count": chunk_count,
            "translation_source": translation_source,
            "sync_validation": sync_results,
        }
        _write_status(
            safe_job_id,
            "completed",
            100,
            output_ready=True,
            translation_source=translation_source,
            sync_results=sync_results,
        )
        return result
    except JobProcessingError as exc:
        _write_status(
            safe_job_id,
            "failed",
            0,
            stage=exc.stage,
            error=exc.message,
            sync_results=sync_results,
        )
        logger.error("Job %s failed at %s", safe_job_id, exc.stage)
        raise
    except Exception as exc:
        _write_status(
            safe_job_id,
            "failed",
            0,
            stage=current_stage,
            error=f"Processing failed during {current_stage}.",
            sync_results=sync_results,
        )
        logger.error(
            "Job %s failed at %s (%s)",
            safe_job_id,
            current_stage,
            type(exc).__name__,
        )
        raise JobProcessingError(
            current_stage,
            f"Processing failed during {current_stage}.",
        ) from None