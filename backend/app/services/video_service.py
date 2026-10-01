import json
import subprocess
from pathlib import Path


ALLOWED_EXTENSIONS = {".mp4", ".webm"}
MAX_DURATION_SECONDS = 120
MAX_VIDEO_WIDTH = 7680
MAX_VIDEO_HEIGHT = 4320
MAX_FRAME_RATE = 120
MAX_AUDIO_CHANNELS = 8
MAX_AUDIO_SAMPLE_RATE = 96000
MAX_AUDIO_SAMPLE_RATE_MIN = 8000


def _looks_like_media_signature(video_path: Path, extension: str) -> bool:
    try:
        header = video_path.read_bytes()[:256]
    except OSError:
        return False

    if extension == ".mp4":
        return b"ftyp" in header[:64] or header.startswith(b"\x00\x00\x00")
    if extension == ".webm":
        return header.startswith(b"\x1aE\xdf\xa3") or b"EBML" in header[:32]
    return False


def _fraction_to_float(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def get_video_metadata(video_path: Path) -> dict:
    """Extract video metadata using FFprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,channels,sample_rate",
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
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    return json.loads(result.stdout)


def validate_video(video_path: Path) -> dict:
    """Validate the actual container and stream properties before expensive processing."""

    if not isinstance(video_path, Path):
        raise ValueError("Video path must be a Path object.")

    if not video_path.is_file():
        raise ValueError("Uploaded file is missing.")

    extension = video_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported format. Please upload MP4 or WebM.")

    if not _looks_like_media_signature(video_path, extension):
        raise ValueError("Uploaded file does not contain a valid MP4 or WebM signature.")

    metadata = get_video_metadata(video_path)

    duration = float(metadata.get("format", {}).get("duration", 0))
    if not duration or duration <= 0:
        raise ValueError("Could not determine video duration.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Video exceeds the 2-minute hackathon limit.")

    streams = metadata.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError("Uploaded video does not contain a valid video stream.")

    codec_name = (video_stream.get("codec_name") or "").lower()
    if extension == ".mp4" and codec_name not in {"h264", "avc1", "mpeg4", "mp4v"}:
        raise ValueError("Uploaded MP4 codec is not supported for local processing.")
    if extension == ".webm" and codec_name not in {"vp8", "vp9", "av1"}:
        raise ValueError("Uploaded WebM codec is not supported for local processing.")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0 or width > MAX_VIDEO_WIDTH or height > MAX_VIDEO_HEIGHT:
        raise ValueError("Uploaded video resolution is unsupported.")

    frame_rate = _fraction_to_float(video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate"))
    if frame_rate <= 0 or frame_rate > MAX_FRAME_RATE:
        raise ValueError("Uploaded video frame rate is unsupported.")

    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audio_streams:
        raise ValueError("Uploaded video does not contain an audio track required for dubbing.")

    audio_stream = audio_streams[0]
    audio_channels = int(audio_stream.get("channels") or 0)
    sample_rate = int(audio_stream.get("sample_rate") or 0)
    if audio_channels <= 0 or audio_channels > MAX_AUDIO_CHANNELS:
        raise ValueError("Uploaded audio channel count is unsupported.")
    if sample_rate < MAX_AUDIO_SAMPLE_RATE_MIN or sample_rate > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError("Uploaded audio sample rate is unsupported.")

    return {
        "filename": video_path.name,
        "extension": extension,
        "duration": round(duration, 2),
        "metadata": metadata,
    }