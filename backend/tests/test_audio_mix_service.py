r"""Tests for background-audio mixing; opt in to real local FFmpeg with:

    $env:TRANSCADENCE_RUN_REAL_AUDIO_MIX = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_audio_mix_service.py::test_real_ffmpeg_mix -q -s
"""

import math
import os
from pathlib import Path
import struct
from uuid import uuid4
import wave

import pytest

from backend.app.services import audio_mix_service
from backend.app.services.audio_mix_service import AudioMixResult, mix_audio
from backend.app.services.tts_service import get_audio_duration


RUN_REAL_FFMPEG = os.getenv("TRANSCADENCE_RUN_REAL_AUDIO_MIX") == "1"


def _write_wav(path: Path, duration: float, frequency: float = 440.0) -> Path:
    sample_rate = 16000
    frame_count = int(sample_rate * duration)
    frames = bytearray()
    for frame in range(frame_count):
        sample = int(8000 * math.sin(2 * math.pi * frequency * frame / sample_rate))
        frames.extend(struct.pack("<h", sample))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    return path


@pytest.fixture
def audio_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "audio"
    monkeypatch.setattr(audio_mix_service, "AUDIO_ROOT", root)
    monkeypatch.setattr(audio_mix_service.shutil, "which", lambda _: "ffmpeg-test")
    return root


@pytest.fixture
def input_wavs(tmp_path: Path) -> tuple[Path, Path]:
    original = _write_wav(tmp_path / "original.wav", 1.0, 220.0)
    speech = _write_wav(tmp_path / "speech.wav", 0.75, 440.0)
    return original, speech


def _write_mock_mix(path: Path, duration: float = 1.0) -> None:
    _write_wav(path, duration, 330.0)


def _mock_successful_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    duration: float = 1.0,
) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == audio_mix_service.FFMPEG_TIMEOUT_SECONDS
        _write_mock_mix(Path(command[-1]), duration)
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(audio_mix_service.subprocess, "run", fake_run)
    return commands


def test_valid_input_arguments_are_accepted(
    input_wavs: tuple[Path, Path],
    audio_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, speech = input_wavs
    _mock_successful_ffmpeg(monkeypatch)

    result = mix_audio(original, speech, str(uuid4()), 1)

    assert isinstance(result, AudioMixResult)
    assert result.background_volume == 0.25


def test_invalid_job_uuid_is_rejected(
    input_wavs: tuple[Path, Path],
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="job_id"):
        mix_audio(*input_wavs, "../outside", 1)


@pytest.mark.parametrize("segment_id", [0, -1, True, 1.5])
def test_invalid_segment_id_is_rejected(
    segment_id: int,
    input_wavs: tuple[Path, Path],
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="segment_id"):
        mix_audio(*input_wavs, str(uuid4()), segment_id)


def test_missing_original_audio_is_rejected(
    tmp_path: Path,
    audio_root: Path,
) -> None:
    speech = _write_wav(tmp_path / "speech.wav", 1.0)
    with pytest.raises(FileNotFoundError, match="original_audio_path"):
        mix_audio(tmp_path / "missing.wav", speech, str(uuid4()), 1)


def test_missing_speech_audio_is_rejected(
    tmp_path: Path,
    audio_root: Path,
) -> None:
    original = _write_wav(tmp_path / "original.wav", 1.0)
    with pytest.raises(FileNotFoundError, match="speech_audio_path"):
        mix_audio(original, tmp_path / "missing.wav", str(uuid4()), 1)


@pytest.mark.parametrize("background_volume", [0.0, -0.1, 1.01, math.nan, math.inf])
def test_invalid_background_volume_is_rejected(
    background_volume: float,
    input_wavs: tuple[Path, Path],
    audio_root: Path,
) -> None:
    with pytest.raises(ValueError, match="background_volume"):
        mix_audio(*input_wavs, str(uuid4()), 1, background_volume)


def test_output_path_is_isolated_by_job_id(
    input_wavs: tuple[Path, Path],
    audio_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _mock_successful_ffmpeg(monkeypatch)
    first_job_id = str(uuid4())
    second_job_id = str(uuid4())

    first = mix_audio(*input_wavs, first_job_id, 1)
    second = mix_audio(*input_wavs, second_job_id, 1)

    assert first.output_path == audio_root / first_job_id / "mixed" / "1.wav"
    assert second.output_path == audio_root / second_job_id / "mixed" / "1.wav"
    assert first.output_path != second.output_path
    assert len(commands) == 2


def test_ffmpeg_command_uses_ordered_inputs_filter_and_pcm(
    input_wavs: tuple[Path, Path],
    audio_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, speech = input_wavs
    commands = _mock_successful_ffmpeg(monkeypatch)

    result = mix_audio(original, speech, str(uuid4()), 7, 0.4)
    command = commands[0]
    filter_graph = command[command.index("-filter_complex") + 1]

    assert command[command.index("-i") + 1] == str(original.resolve())
    second_input_index = command.index("-i", command.index("-i") + 1)
    assert command[second_input_index + 1] == str(speech.resolve())
    assert filter_graph == (
        "[0:a]volume=0.4[bg];[1:a]volume=1.0[speech];"
        "[bg][speech]amix=inputs=2:duration=longest:dropout_transition=0[mix]"
    )
    assert command[command.index("-map") + 1] == "[mix]"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[command.index("-f") + 1] == "wav"
    assert result.output_path.suffix == ".wav"


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_ffmpeg_failures_are_clear_and_clean_up_temporary_files(
    failure: str,
    input_wavs: tuple[Path, Path],
    audio_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> object:
        if failure == "timeout":
            raise audio_mix_service.subprocess.TimeoutExpired(command, 120)
        return type(
            "Completed",
            (),
            {"returncode": 1, "stderr": "sensitive arbitrary diagnostic"},
        )()

    monkeypatch.setattr(audio_mix_service.subprocess, "run", fake_run)
    job_id = str(uuid4())

    expected_exception = TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(expected_exception) as error:
        mix_audio(*input_wavs, job_id, 1)

    assert "sensitive arbitrary diagnostic" not in str(error.value)
    output_directory = audio_root / job_id / "mixed"
    assert list(output_directory.iterdir()) == []


@pytest.mark.skipif(
    not RUN_REAL_FFMPEG,
    reason="Set TRANSCADENCE_RUN_REAL_AUDIO_MIX=1 to run real FFmpeg integration.",
)
def test_real_ffmpeg_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_mix_service, "AUDIO_ROOT", tmp_path / "audio")
    original, speech = (
        _write_wav(tmp_path / "original.wav", 1.0, 220.0),
        _write_wav(tmp_path / "speech.wav", 0.6, 440.0),
    )
    job_id = str(uuid4())

    result = mix_audio(original, speech, job_id, 1)

    assert result.output_path.is_file()
    assert result.output_duration == get_audio_duration(result.output_path)
    assert result.output_duration > 0
    assert original.is_file()
    assert speech.is_file()