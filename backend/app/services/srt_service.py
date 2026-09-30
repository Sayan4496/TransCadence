"""Validate Hindi transcript segments and write UTF-8 SRT subtitles."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import UUID


SUBTITLE_ROOT = Path("storage/subtitles")
SUPPORTED_LANGUAGE = "hi"
REQUIRED_SEGMENT_FIELDS = {"id", "start", "end", "text"}


@dataclass(frozen=True)
class SrtResult:
    """Details about a generated SRT subtitle file."""

    job_id: str
    language: str
    segment_count: int
    output_path: Path


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a UUID string.")

    try:
        return str(UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a valid UUID.") from exc


def _validate_timestamp_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Segment {field_name} must be numeric and finite.")

    timestamp = float(value)
    if timestamp != timestamp or timestamp in (float("inf"), float("-inf")):
        raise ValueError(f"Segment {field_name} must be finite.")

    return timestamp


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(
        (Decimal(str(seconds)) * 1000).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )
    hours, remaining_milliseconds = divmod(total_milliseconds, 3_600_000)
    minutes, remaining_milliseconds = divmod(remaining_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remaining_milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def _get_subtitle_text(segment: Mapping[str, Any], index: int) -> str:
    value = segment.get("translated_text", segment.get("text"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Segment {index} subtitle text must not be empty.")
    return value


def _validate_segments(segments: Any) -> list[tuple[float, float, str]]:
    if not isinstance(segments, list):
        raise ValueError("segments must be provided as a list.")
    if not segments:
        raise ValueError("segments must contain at least one segment.")

    validated: list[tuple[float, float, str]] = []
    previous_start = -1.0
    previous_end = -1.0

    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping):
            raise ValueError(f"Segment {index} must be a mapping.")

        missing_fields = REQUIRED_SEGMENT_FIELDS - segment.keys()
        if missing_fields:
            raise ValueError(f"Segment {index} is missing required fields.")

        segment_id = segment["id"]
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise ValueError(f"Segment {index} id must be a positive integer.")
        if segment_id <= 0:
            raise ValueError(f"Segment {index} id must be a positive integer.")

        start = _validate_timestamp_value(segment["start"], "start")
        end = _validate_timestamp_value(segment["end"], "end")
        if start < 0:
            raise ValueError(f"Segment {index} start must be non-negative.")
        if end <= start:
            raise ValueError(f"Segment {index} end must be greater than start.")

        if start < previous_start:
            raise ValueError("Segments must be ordered by start time.")
        if index > 1 and start < previous_end:
            raise ValueError("Segments must not overlap.")

        subtitle_text = _get_subtitle_text(segment, index)
        validated.append((start, end, subtitle_text))
        previous_start = start
        previous_end = end

    return validated


def build_srt(segments: list[Mapping[str, Any]]) -> str:
    """Return validated segments as standard SRT text."""
    validated_segments = _validate_segments(segments)
    cues = []

    for cue_number, (start, end, subtitle_text) in enumerate(
        validated_segments,
        start=1,
    ):
        cues.append(
            f"{cue_number}\n"
            f"{_format_timestamp(start)} --> {_format_timestamp(end)}\n"
            f"{subtitle_text}"
        )

    return "\n\n".join(cues) + "\n"


def _get_output_path(job_id: str, language: str) -> tuple[Path, Path]:
    subtitle_root = SUBTITLE_ROOT.resolve(strict=False)
    expected_job_directory = subtitle_root / job_id
    job_directory = (SUBTITLE_ROOT / job_id).resolve(strict=False)
    if job_directory != expected_job_directory:
        raise ValueError("Subtitle job directory must not redirect to another path.")
    if not job_directory.is_relative_to(subtitle_root):
        raise ValueError("Subtitle job directory is outside subtitle storage.")

    job_directory.mkdir(parents=True, exist_ok=True)
    resolved_job_directory = job_directory.resolve(strict=True)
    if resolved_job_directory != expected_job_directory:
        raise ValueError("Subtitle job directory must not redirect to another path.")
    if not resolved_job_directory.is_relative_to(subtitle_root):
        raise ValueError("Subtitle directory escapes subtitle storage.")

    output_path = resolved_job_directory / f"{language}.srt"
    if not output_path.resolve(strict=False).is_relative_to(resolved_job_directory):
        raise ValueError("Subtitle output path escapes its job directory.")

    return resolved_job_directory, output_path


def write_srt(
    segments: list[Mapping[str, Any]],
    job_id: str,
    language: str = SUPPORTED_LANGUAGE,
) -> SrtResult:
    """Validate Hindi segments and atomically write a job-scoped SRT file."""
    safe_job_id = _validate_job_id(job_id)
    if language != SUPPORTED_LANGUAGE:
        raise ValueError("Only Hindi (language code 'hi') is supported.")

    srt_text = build_srt(segments)
    output_directory, output_path = _get_output_path(safe_job_id, language)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_directory,
            prefix=".hi-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(srt_text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, output_path)
    except OSError as exc:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Failed to write subtitle file.") from None

    return SrtResult(
        job_id=safe_job_id,
        language=language,
        segment_count=len(segments),
        output_path=output_path,
    )