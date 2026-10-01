"""Adjust generated audio duration with local FFmpeg processing."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from uuid import UUID

from backend.app.services.sync_service import (
    MAX_DURATION_SECONDS,
    TOLERANCE_MS,
)
from backend.app.services.tts_service import get_audio_duration


AUDIO_ROOT = Path("storage/audio")
MIN_SPEED_FACTOR = 0.25
MAX_SPEED_FACTOR = 4.0
MIN_ATEMPO_FACTOR = 0.5
MAX_ATEMPO_FACTOR = 2.0
FFMPEG_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class AudioAdjustmentResult:
    """Measured result of adjusting one generated audio segment."""

    input_duration: float
    target_duration: float
    speed_factor: float
    output_duration: float
    sync_error_ms: float
    within_tolerance: bool
    output_path: Path


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a UUID string.")

    try:
        parsed_job_id = UUID(job_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a valid UUID.") from exc

    return str(parsed_job_id)


def _validate_segment_id(segment_id: int) -> int:
    if isinstance(segment_id, bool) or not isinstance(segment_id, int):
        raise ValueError("segment_id must be a positive integer.")
    if segment_id <= 0:
        raise ValueError("segment_id must be a positive integer.")

    return segment_id


def _validate_speed_factor(speed_factor: float) -> float:
    if isinstance(speed_factor, bool) or not isinstance(speed_factor, (int, float)):
        raise ValueError("speed_factor must be a finite number.")

    factor = float(speed_factor)
    if not math.isfinite(factor):
        raise ValueError("speed_factor must be finite.")
    if not MIN_SPEED_FACTOR <= factor <= MAX_SPEED_FACTOR:
        raise ValueError(
            f"speed_factor must be between {MIN_SPEED_FACTOR} and {MAX_SPEED_FACTOR}."
        )

    return factor


def _build_atempo_filter(speed_factor: float) -> str:
    """Build a filter chain whose individual atempo values stay in range."""
    remaining_factor = speed_factor
    factors: list[float] = []

    while remaining_factor > MAX_ATEMPO_FACTOR:
        factors.append(MAX_ATEMPO_FACTOR)
        remaining_factor /= MAX_ATEMPO_FACTOR

    while remaining_factor < MIN_ATEMPO_FACTOR:
        factors.append(MIN_ATEMPO_FACTOR)
        remaining_factor /= MIN_ATEMPO_FACTOR

    factors.append(remaining_factor)
    return ",".join(f"atempo={factor:.10g}" for factor in factors)


def adjust_audio_duration(
    input_path: Path,
    job_id: str,
    segment_id: int,
    speed_factor: float,
) -> AudioAdjustmentResult:
    """Adjust one WAV's playback speed and return its measured duration.

    The target duration is derived as input duration divided by speed factor.
    Output is written to storage/audio/{job_id}/adjusted/{segment_id}.wav.
    """
    if not isinstance(input_path, Path):
        raise ValueError("input_path must be a local Path.")

    safe_job_id = _validate_job_id(job_id)
    safe_segment_id = _validate_segment_id(segment_id)
    factor = _validate_speed_factor(speed_factor)

    try:
        resolved_input_path = input_path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"Input audio file not found: {input_path}") from exc

    if not resolved_input_path.is_file():
        raise ValueError("input_path must reference a file.")

    input_duration = get_audio_duration(resolved_input_path)
    if input_duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Input duration must not exceed {MAX_DURATION_SECONDS} seconds."
        )

    target_duration = input_duration / factor
    if target_duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Target duration must not exceed {MAX_DURATION_SECONDS} seconds."
        )

    output_directory = AUDIO_ROOT / safe_job_id / "adjusted"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{safe_segment_id}.wav"

    if resolved_input_path == output_path.resolve(strict=False):
        raise ValueError("input_path and output_path must be different files.")

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise FileNotFoundError("FFmpeg executable was not found in PATH.")

    temporary_file = tempfile.NamedTemporaryFile(
        dir=output_directory,
        prefix=f".{safe_segment_id}-",
        suffix=".wav",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    command = [
        ffmpeg_executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,crypto",
        "-y",
        "-i",
        str(resolved_input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        _build_atempo_filter(factor),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary_path),
    ]

    try:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"FFmpeg audio adjustment exceeded {FFMPEG_TIMEOUT_SECONDS} seconds."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio adjustment failed: {result.stderr.strip()}"
            )
        if not temporary_path.is_file():
            raise RuntimeError("FFmpeg did not create the adjusted WAV file.")

        os.replace(temporary_path, output_path)
        output_duration = get_audio_duration(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    sync_error_ms = (output_duration - target_duration) * 1000
    return AudioAdjustmentResult(
        input_duration=input_duration,
        target_duration=target_duration,
        speed_factor=factor,
        output_duration=output_duration,
        sync_error_ms=sync_error_ms,
        within_tolerance=abs(sync_error_ms) <= TOLERANCE_MS,
        output_path=output_path,
    )