"""Generate and measure local Hindi speech audio with Piper TTS."""

import math
import os
from pathlib import Path
import tempfile
from uuid import UUID
import wave

from huggingface_hub import hf_hub_download
from piper import PiperVoice


VOICE_REPOSITORY = "rhasspy/piper-voices"
VOICE_MODEL = "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"
VOICE_CONFIG = f"{VOICE_MODEL}.json"
AUDIO_ROOT = Path("storage/audio")

_hindi_voice: PiperVoice | None = None


def get_hindi_voice() -> PiperVoice:
    """Load the public Hindi Piper voice once, caching model files locally."""
    global _hindi_voice

    if _hindi_voice is None:
        model_path = hf_hub_download(
            repo_id=VOICE_REPOSITORY,
            filename=VOICE_MODEL,
        )
        config_path = hf_hub_download(
            repo_id=VOICE_REPOSITORY,
            filename=VOICE_CONFIG,
        )
        _hindi_voice = PiperVoice.load(
            model_path,
            config_path=config_path,
            use_cuda=False,
        )

    return _hindi_voice


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


def generate_hindi_speech(
    text: str,
    job_id: str,
    segment_id: int,
    language: str = "hi",
) -> Path:
    """Generate one Hindi transcript segment as a PCM WAV file.

    The Piper voice weights are downloaded from the public voice repository on
    first use and then reused from the Hugging Face local cache.
    """
    if language != "hi":
        raise ValueError("Only Hindi (language code 'hi') is supported.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must contain non-whitespace characters.")

    safe_job_id = _validate_job_id(job_id)
    safe_segment_id = _validate_segment_id(segment_id)

    output_directory = AUDIO_ROOT / safe_job_id / "tts"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{safe_segment_id}.wav"

    temporary_file = tempfile.NamedTemporaryFile(
        dir=output_directory,
        prefix=f".{safe_segment_id}-",
        suffix=".wav",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        with wave.open(str(temporary_path), "wb") as wav_file:
            get_hindi_voice().synthesize_wav(text.strip(), wav_file)

        get_audio_duration(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Return WAV duration in seconds, rejecting missing or invalid audio."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Audio file is not a valid WAV: {audio_path}") from exc

    if frame_count <= 0 or frame_rate <= 0:
        raise ValueError(f"Audio file has no measurable duration: {audio_path}")

    duration = frame_count / frame_rate
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Audio file has an invalid duration: {audio_path}")

    return duration