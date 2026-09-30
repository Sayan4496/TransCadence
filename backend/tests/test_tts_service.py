r"""Unit tests for local Hindi TTS and an opt-in real-engine integration test.

Run real Piper generation with:
    $env:TRANSCADENCE_RUN_REAL_TTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_tts_service.py::test_real_hindi_tts_manual -q -s

That test downloads the public Hindi voice on first use and writes a real WAV
under storage/audio/{job_id}/tts/; subsequent runs use the local model cache.
"""

import os
from pathlib import Path
from uuid import uuid4
import wave

import pytest

from backend.app.services import tts_service


SAMPLE_HINDI_TEXT = "नमस्ते, यह TransCadence का हिंदी डबिंग परीक्षण है।"
RUN_REAL_TTS = os.getenv("TRANSCADENCE_RUN_REAL_TTS") == "1"


class FakeHindiVoice:
    """Write a short valid PCM WAV without initializing a TTS model."""

    def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
        assert text
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)


@pytest.fixture
def fake_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tts_service, "AUDIO_ROOT", tmp_path / "audio")
    monkeypatch.setattr(tts_service, "get_hindi_voice", lambda: FakeHindiVoice())


def test_hindi_text_is_accepted_and_wav_is_created(
    fake_voice: None,
) -> None:
    job_id = str(uuid4())

    audio_path = tts_service.generate_hindi_speech(
        SAMPLE_HINDI_TEXT,
        job_id,
        1,
    )

    assert audio_path.is_file()
    assert audio_path.name == "1.wav"
    assert audio_path.parent.name == "tts"
    assert job_id in audio_path.parts
    assert tts_service.get_audio_duration(audio_path) > 0


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(text: str, fake_voice: None) -> None:
    with pytest.raises(ValueError, match="text"):
        tts_service.generate_hindi_speech(text, str(uuid4()), 1)


def test_unsupported_language_is_rejected(fake_voice: None) -> None:
    with pytest.raises(ValueError, match="Only Hindi"):
        tts_service.generate_hindi_speech(
            SAMPLE_HINDI_TEXT,
            str(uuid4()),
            1,
            language="bn",
        )


@pytest.mark.parametrize("job_id", ["not-a-uuid", "../outside", ""])
def test_invalid_job_id_is_rejected(job_id: str, fake_voice: None) -> None:
    with pytest.raises(ValueError, match="job_id"):
        tts_service.generate_hindi_speech(SAMPLE_HINDI_TEXT, job_id, 1)


@pytest.mark.parametrize("segment_id", [0, -1, True, 1.5])
def test_invalid_segment_id_is_rejected(
    segment_id: int,
    fake_voice: None,
) -> None:
    with pytest.raises(ValueError, match="segment_id"):
        tts_service.generate_hindi_speech(
            SAMPLE_HINDI_TEXT,
            str(uuid4()),
            segment_id,
        )


def test_output_path_is_isolated_by_job_id(fake_voice: None) -> None:
    first_job_id = str(uuid4())
    second_job_id = str(uuid4())

    first_path = tts_service.generate_hindi_speech(
        SAMPLE_HINDI_TEXT,
        first_job_id,
        1,
    )
    second_path = tts_service.generate_hindi_speech(
        SAMPLE_HINDI_TEXT,
        second_job_id,
        1,
    )

    assert first_path != second_path
    assert first_job_id in first_path.parts
    assert second_job_id in second_path.parts
    assert first_path.is_file() and second_path.is_file()


def test_audio_duration_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tts_service.get_audio_duration(tmp_path / "missing.wav")


def test_audio_duration_measures_valid_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "valid.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 8000)

    assert tts_service.get_audio_duration(audio_path) == pytest.approx(0.5)


@pytest.mark.skipif(
    not RUN_REAL_TTS,
    reason="Set TRANSCADENCE_RUN_REAL_TTS=1 to run real Hindi synthesis.",
)
def test_real_hindi_tts_manual() -> None:
    """Generate and measure real Hindi speech with the locally cached Piper voice."""
    job_id = str(uuid4())

    audio_path = tts_service.generate_hindi_speech(
        SAMPLE_HINDI_TEXT,
        job_id,
        1,
    )
    duration = tts_service.get_audio_duration(audio_path)

    print(f"job_id={job_id}")
    print(f"audio_path={audio_path}")
    print(f"duration_seconds={duration:.3f}")
    assert audio_path.is_file()
    assert duration > 0