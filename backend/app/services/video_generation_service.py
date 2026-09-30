from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from backend.app.services.video_service import get_video_metadata


STORAGE_ROOT = Path("storage").resolve()
OUTPUT_ROOT = (STORAGE_ROOT / "outputs").resolve()


def _validate_job_id(job_id: str) -> str:
    try:
        return str(uuid.UUID(job_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid job_id") from None


def _resolve_input_path(path: str | Path, name: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError(f"{name} must be a filesystem path")

    try:
        resolved_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise FileNotFoundError(f"{name} was not found") from None

    try:
        resolved_path.relative_to(STORAGE_ROOT)
    except ValueError:
        raise ValueError(f"{name} must remain inside storage") from None

    if not resolved_path.is_file():
        raise ValueError(f"{name} must be a file")

    return resolved_path


def _validate_paths(
    video_path: str | Path,
    audio_path: str | Path,
    srt_path: str | Path,
) -> tuple[Path, Path, Path]:
    video = _resolve_input_path(video_path, "video_path")
    audio = _resolve_input_path(audio_path, "audio_path")
    srt = _resolve_input_path(srt_path, "srt_path")

    if video.suffix.lower() not in {".mp4", ".webm"}:
        raise ValueError("Unsupported video format")
    if audio.suffix.lower() != ".wav":
        raise ValueError("Audio input must be WAV")
    if srt.suffix.lower() != ".srt":
        raise ValueError("Subtitle input must be SRT")

    return video, audio, srt


def _subtitle_filter_expression(srt_path: Path) -> str:
    """Escape a validated path for FFmpeg's filtergraph and Windows drive colon."""
    filter_path = srt_path.as_posix()
    filter_path = filter_path.replace("\\", "\\\\")
    if "'" in filter_path or "\n" in filter_path or "\r" in filter_path:
        raise ValueError("Subtitle path contains unsupported filter characters")
    filter_path = filter_path.replace(":", "\\:")
    return f"subtitles='{filter_path}'"


def _get_output_path(job_id: str) -> tuple[Path, Path]:
    output_root = OUTPUT_ROOT.resolve(strict=False)
    expected_job_directory = output_root / job_id
    job_output_dir = expected_job_directory.resolve(strict=False)

    if job_output_dir != expected_job_directory:
        raise ValueError("Output directory must not redirect to another path")
    try:
        job_output_dir.relative_to(output_root)
    except ValueError:
        raise ValueError("Invalid output directory") from None

    job_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_job_output_dir = job_output_dir.resolve(strict=True)
    if resolved_job_output_dir != expected_job_directory:
        raise ValueError("Output directory must not redirect to another path")

    output_path = resolved_job_output_dir / "hindi_dubbed.mp4"
    if not output_path.resolve(strict=False).is_relative_to(resolved_job_output_dir):
        raise ValueError("Output path escapes the job directory")

    return resolved_job_output_dir, output_path


def _measure_video_duration(video_path: Path) -> float:
    try:
        metadata = get_video_metadata(video_path)
        duration = float(metadata.get("format", {}).get("duration", 0))
    except Exception:
        raise RuntimeError("Could not measure generated video duration") from None

    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Could not measure generated video duration")

    return duration


def generate_dubbed_video(
    job_id: str,
    video_path: str | Path,
    audio_path: str | Path,
    srt_path: str | Path,
) -> dict:
    """
    Generate a final MP4 containing the original video, Hindi dubbed audio,
    and Hindi subtitles.

    The original video is never modified.
    """

    safe_job_id = _validate_job_id(job_id)
    video, audio, srt = _validate_paths(
        video_path,
        audio_path,
        srt_path,
    )

    job_output_dir, output_path = _get_output_path(safe_job_id)
    if output_path.resolve(strict=False) in {video, audio, srt}:
        raise ValueError("Output path must not overwrite an input file")

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise FileNotFoundError("FFmpeg executable was not found in PATH")

    subtitle_filter = _subtitle_filter_expression(srt)

    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
    ]

    temporary_file = tempfile.NamedTemporaryFile(
        dir=job_output_dir,
        prefix=".hindi-dubbed-",
        suffix=".mp4",
        delete=False,
    )
    temporary_output = Path(temporary_file.name)
    temporary_file.close()
    command.append(str(temporary_output))

    try:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Final video generation timed out") from None

        if result.returncode != 0:
            raise RuntimeError("Final video generation failed")
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an invalid output file")

        duration = _measure_video_duration(temporary_output)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    return {
        "job_id": safe_job_id,
        "output_path": str(output_path),
        "duration_seconds": duration,
    }
