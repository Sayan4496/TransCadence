import subprocess
from pathlib import Path


AUDIO_DIR = Path("storage/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path: Path, job_id: str) -> Path:
    output_path = AUDIO_DIR / f"{job_id}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError("Audio file was not created.")

    return output_path