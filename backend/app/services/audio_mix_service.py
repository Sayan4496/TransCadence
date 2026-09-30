"""Mix original background audio with generated Hindi speech."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from uuid import UUID

from backend.app.services.tts_service import get_audio_duration


AUDIO_ROOT = Path("storage/audio")
FFMPEG_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class AudioMixResult:
    """Measured details for a mixed original-audio and speech WAV."""

    original_duration: float
    speech_duration: float
    output_duration: float
    background_volume: float
    output_path: Path


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a UUID string.")

    try:
        return str(UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a valid UUID.") from exc


def _validate_segment_id(segment_id: int) -> int:
    if isinstance(segment_id, bool) or not isinstance(segment_id, int):
        raise ValueError("segment_id must be a positive integer.")
    if segment_id <= 0:
        raise ValueError("segment_id must be a positive integer.")

    return segment_id


def _validate_background_volume(background_volume: float) -> float:
    if isinstance(background_volume, bool) or not isinstance(
        background_volume,
        (int, float),
    ):
        raise ValueError("background_volume must be a finite number.")

    volume = float(background_volume)
    if not math.isfinite(volume) or not 0 < volume <= 1.0:
        raise ValueError("background_volume must be greater than 0 and at most 1.0.")

    return volume


def _resolve_audio_file(path: Path, argument_name: str) -> Path:
    if not isinstance(path, Path):
        raise ValueError(f"{argument_name} must be a local Path.")

    try:
        resolved_path = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"{argument_name} does not exist.") from exc

    if not resolved_path.is_file():
        raise ValueError(f"{argument_name} must reference a regular file.")

    return resolved_path


def _build_filter_graph(background_volume: float) -> str:
    volume = f"{background_volume:.10g}"
    return (
        f"[0:a]volume={volume}[bg];"
        "[1:a]volume=1.0[speech];"
        "[bg][speech]amix=inputs=2:duration=longest:dropout_transition=0[mix]"
    )


def _get_output_path(job_id: str, segment_id: int) -> tuple[Path, Path]:
    audio_root = AUDIO_ROOT.resolve(strict=False)
    job_directory = (AUDIO_ROOT / job_id).resolve(strict=False)
    output_directory = job_directory / "mixed"

    if not job_directory.is_relative_to(audio_root):
        raise ValueError("Output job directory is outside the audio storage root.")

    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_output_directory = output_directory.resolve(strict=True)
    if not resolved_output_directory.is_relative_to(job_directory):
        raise ValueError("Output directory escapes the expected job directory.")

    output_path = resolved_output_directory / f"{segment_id}.wav"
    if not output_path.resolve(strict=False).is_relative_to(job_directory):
        raise ValueError("Output path escapes the expected job directory.")

    return resolved_output_directory, output_path


def mix_audio(
    original_audio_path: Path,
    speech_audio_path: Path,
    job_id: str,
    segment_id: int,
    background_volume: float = 0.25,
) -> AudioMixResult:
    """Mix reduced original audio under normal-volume Hindi speech.

    The result is a PCM WAV saved only under
    ``storage/audio/{job_id}/mixed/{segment_id}.wav``.
    """
    safe_job_id = _validate_job_id(job_id)
    safe_segment_id = _validate_segment_id(segment_id)
    volume = _validate_background_volume(background_volume)
    original_path = _resolve_audio_file(original_audio_path, "original_audio_path")
    speech_path = _resolve_audio_file(speech_audio_path, "speech_audio_path")

    if original_path == speech_path:
        raise ValueError("Original audio and speech audio must be different files.")

    original_duration = get_audio_duration(original_path)
    speech_duration = get_audio_duration(speech_path)

    output_directory, output_path = _get_output_path(safe_job_id, safe_segment_id)
    if output_path.resolve(strict=False) in {original_path, speech_path}:
        raise ValueError("Output path must not overwrite an input audio file.")

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
        "-y",
        "-i",
        str(original_path),
        "-i",
        str(speech_path),
        "-filter_complex",
        _build_filter_graph(volume),
        "-map",
        "[mix]",
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
                f"FFmpeg audio mixing exceeded {FFMPEG_TIMEOUT_SECONDS} seconds."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError("FFmpeg audio mixing failed.")
        if not temporary_path.is_file():
            raise RuntimeError("FFmpeg did not create the mixed WAV file.")

        os.replace(temporary_path, output_path)
        output_duration = get_audio_duration(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return AudioMixResult(
        original_duration=original_duration,
        speech_duration=speech_duration,
        output_duration=output_duration,
        background_volume=volume,
        output_path=output_path,
    )