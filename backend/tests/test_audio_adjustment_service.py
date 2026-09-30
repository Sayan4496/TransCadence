import math
from pathlib import Path
import struct
from uuid import uuid4
import wave

import pytest

from backend.app.services import audio_adjustment_service
from backend.app.services.audio_adjustment_service import adjust_audio_duration
from backend.app.services.tts_service import get_audio_duration


@pytest.fixture
def audio_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "audio"
    monkeypatch.setattr(audio_adjustment_service, "AUDIO_ROOT", root)
    return root


@pytest.fixture
def input_wav(tmp_path: Path) -> Path:
    path = tmp_path / "input.wav"
    sample_rate = 16000
    frame_count = sample_rate * 2
    frames = bytearray()
    for frame in range(frame_count):
        sample = int(10000 * math.sin(2 * math.pi * 440 * frame / sample_rate))
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    return path


def test_missing_input_file_is_rejected(audio_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        adjust_audio_duration(
            audio_root / "missing.wav",
            str(uuid4()),
            1,
            1.2,
        )


def test_non_file_input_is_rejected(
    tmp_path: Path,
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="file"):
        adjust_audio_duration(tmp_path, str(uuid4()), 1, 1.2)


@pytest.mark.parametrize(
    "speed_factor",
    [0.0, -1.0, math.nan, math.inf, -math.inf, 0.24, 4.01],
)
def test_invalid_speed_factor_is_rejected(
    speed_factor: float,
    input_wav: Path,
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="speed_factor"):
        adjust_audio_duration(input_wav, str(uuid4()), 1, speed_factor)


def test_same_input_and_output_path_is_rejected(
    audio_root: Path,
) -> None:
    job_id = str(uuid4())
    input_path = audio_root / job_id / "adjusted" / "1.wav"
    input_path.parent.mkdir(parents=True)
    with wave.open(str(input_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)

    with pytest.raises(ValueError, match="different files"):
        adjust_audio_duration(input_path, job_id, 1, 1.2)


def test_invalid_job_id_is_rejected(input_wav: Path, audio_root: Path) -> None:
    with pytest.raises(ValueError, match="job_id"):
        adjust_audio_duration(input_wav, "../outside", 1, 1.2)


@pytest.mark.parametrize("segment_id", [0, -1, True, 1.5])
def test_invalid_segment_id_is_rejected(
    segment_id: int,
    input_wav: Path,
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="segment_id"):
        adjust_audio_duration(input_wav, str(uuid4()), segment_id, 1.2)


@pytest.mark.parametrize(
    ("speed_factor", "expected_filter"),
    [
        (0.25, "atempo=0.5,atempo=0.5"),
        (0.5, "atempo=0.5"),
        (1.2, "atempo=1.2"),
        (2.0, "atempo=2"),
        (4.0, "atempo=2,atempo=2"),
    ],
)
def test_atempo_chain_stays_within_filter_limits(
    speed_factor: float,
    expected_filter: str,
) -> None:
    assert audio_adjustment_service._build_atempo_filter(speed_factor) == expected_filter


def test_successful_adjustment_measures_actual_output(
    input_wav: Path,
    audio_root: Path,
) -> None:
    job_id = str(uuid4())
    input_duration = get_audio_duration(input_wav)

    result = adjust_audio_duration(input_wav, job_id, 1, 1.2)

    measured_output_duration = get_audio_duration(result.output_path)
    assert result.output_path.is_file()
    assert result.output_path != input_wav
    assert result.output_path == audio_root / job_id / "adjusted" / "1.wav"
    assert result.output_duration == measured_output_duration
    assert result.input_duration == input_duration
    assert result.target_duration == pytest.approx(input_duration / 1.2)
    assert result.sync_error_ms == pytest.approx(
        (measured_output_duration - result.target_duration) * 1000
    )