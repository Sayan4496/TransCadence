import json
import subprocess
from pathlib import Path


ALLOWED_EXTENSIONS = {".mp4", ".webm"}
MAX_DURATION_SECONDS = 120


def get_video_metadata(video_path: Path) -> dict:
    """
    Extract video metadata using FFprobe.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed: {result.stderr}"
        )

    return json.loads(result.stdout)


def validate_video(video_path: Path) -> dict:
    """
    Validate video extension and duration.
    """

    extension = video_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported format. Please upload MP4 or WebM."
        )

    metadata = get_video_metadata(video_path)

    duration = float(
        metadata.get("format", {}).get("duration", 0)
    )

    if duration <= 0:
        raise ValueError(
            "Could not determine video duration."
        )

    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            "Video exceeds the 2-minute hackathon limit."
        )

    return {
        "filename": video_path.name,
        "extension": extension,
        "duration": round(duration, 2),
        "metadata": metadata,
    }