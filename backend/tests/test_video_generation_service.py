from pathlib import Path, PureWindowsPath
import os
import subprocess
from uuid import uuid4

import pytest

from backend.app.services import video_generation_service as service
from backend.app.services.video_generation_service import generate_dubbed_video
from backend.app.services.video_service import get_video_metadata


RUN_REAL_VIDEO = os.getenv("TRANSCADENCE_RUN_REAL_VIDEO") == "1"


@pytest.fixture
def storage_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    storage_root = tmp_path / "storage"
    output_root = storage_root / "outputs"
    video_path = storage_root / "uploads" / "input.webm"
    audio_path = storage_root / "audio" / "dub.wav"
    srt_path = storage_root / "subtitles" / "hindi.srt"

    for path, content in (
        (video_path, b"original-video"),
        (audio_path, b"hindi-audio"),
        (srt_path, "नमस्ते".encode("utf-8")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    monkeypatch.setattr(service, "STORAGE_ROOT", storage_root.resolve())
    monkeypatch.setattr(service, "OUTPUT_ROOT", output_root.resolve())
    monkeypatch.setattr(service.shutil, "which", lambda _: "ffmpeg-test.exe")
    return storage_root, video_path, audio_path, srt_path


def _mock_successful_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    metadata_duration: str = "4.25",
) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 120
        temporary_output = Path(command[-1])
        assert temporary_output.suffix == ".mp4"
        assert temporary_output.name != "hindi_dubbed.mp4"
        temporary_output.write_bytes(b"generated-mp4")
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    def fake_metadata(path: Path) -> dict[str, object]:
        assert path.is_file()
        assert path.suffix == ".mp4"
        return {"format": {"duration": metadata_duration}}

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service, "get_video_metadata", fake_metadata)
    return commands


def test_generate_video_uses_existing_video_metadata_duration(
    storage_tree: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root, video_path, audio_path, srt_path = storage_tree
    commands = _mock_successful_ffmpeg(monkeypatch)
    job_id = str(uuid4())
    original_video = video_path.read_bytes()
    replacements: list[tuple[Path, Path]] = []
    real_replace = service.os.replace

    def track_replace(source: Path, destination: Path) -> None:
        assert source.is_file()
        assert source.parent == destination.parent
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(service.os, "replace", track_replace)

    result = generate_dubbed_video(
        job_id,
        video_path,
        audio_path,
        srt_path,
    )

    expected_output = storage_root / "outputs" / job_id / "hindi_dubbed.mp4"
    assert result == {
        "job_id": job_id,
        "output_path": str(expected_output),
        "duration_seconds": 4.25,
    }
    assert expected_output.read_bytes() == b"generated-mp4"
    assert video_path.read_bytes() == original_video
    assert len(replacements) == 1
    assert replacements[0][1] == expected_output
    assert replacements[0][0] != expected_output
    assert not replacements[0][0].exists()
    assert len(commands) == 1


def test_ffmpeg_command_maps_inputs_and_burns_subtitles(
    storage_tree: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, video_path, audio_path, srt_path = storage_tree
    commands = _mock_successful_ffmpeg(monkeypatch)

    generate_dubbed_video(str(uuid4()), video_path, audio_path, srt_path)

    command = commands[0]
    assert command[0] == "ffmpeg-test.exe"
    assert command[command.index("-i") + 1] == str(video_path.resolve())
    second_input = command.index("-i", command.index("-i") + 1)
    assert command[second_input + 1] == str(audio_path.resolve())
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "1:a:0"
    assert command[command.index("-vf") + 1] == service._subtitle_filter_expression(
        srt_path.resolve()
    )
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-shortest" in command
    assert command[command.index("-movflags") + 1] == "+faststart"


def test_subtitle_filter_escapes_windows_path_characters() -> None:
    windows_path = PureWindowsPath(r"C:\Users\A Name\subs.srt")

    assert service._subtitle_filter_expression(windows_path) == (
        r"subtitles='C\:/Users/A Name/subs.srt'"
    )


def test_subtitle_filter_rejects_unparseable_apostrophes() -> None:
    with pytest.raises(ValueError, match="unsupported filter characters"):
        service._subtitle_filter_expression(PureWindowsPath(r"C:\storage\man's.srt"))


def test_invalid_job_id_is_rejected(
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    _, video_path, audio_path, srt_path = storage_tree

    with pytest.raises(ValueError, match="Invalid job_id"):
        generate_dubbed_video("../outside", video_path, audio_path, srt_path)


@pytest.mark.parametrize("suffix", [".mov", ".mkv"])
def test_unsupported_video_extension_is_rejected(
    suffix: str,
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    storage_root, _, audio_path, srt_path = storage_tree
    video_path = storage_root / "uploads" / f"input{suffix}"
    video_path.write_bytes(b"video")

    with pytest.raises(ValueError, match="Unsupported video format"):
        generate_dubbed_video(str(uuid4()), video_path, audio_path, srt_path)


def test_input_outside_storage_is_rejected(
    tmp_path: Path,
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    _, _, audio_path, srt_path = storage_tree
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"video")

    with pytest.raises(ValueError, match="inside storage"):
        generate_dubbed_video(str(uuid4()), outside_video, audio_path, srt_path)


def test_missing_input_is_rejected(
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    _, video_path, _, srt_path = storage_tree
    missing_audio = video_path.parent / "missing.wav"

    with pytest.raises(FileNotFoundError, match="audio_path was not found"):
        generate_dubbed_video(str(uuid4()), video_path, missing_audio, srt_path)


@pytest.mark.parametrize(
    ("filename", "expected_message"),
    [("audio.mp3", "Audio input must be WAV"), ("subs.txt", "Subtitle input must be SRT")],
)
def test_invalid_input_extensions_are_rejected(
    filename: str,
    expected_message: str,
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    storage_root, video_path, _, _ = storage_tree
    invalid_path = storage_root / "other" / filename
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_bytes(b"data")

    audio_path = invalid_path if filename == "audio.mp3" else storage_root / "audio" / "valid.wav"
    srt_path = invalid_path if filename == "subs.txt" else storage_root / "subtitles" / "valid.srt"
    if not audio_path.exists():
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")
    if not srt_path.exists():
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_bytes(b"srt")

    with pytest.raises(ValueError, match=expected_message):
        generate_dubbed_video(str(uuid4()), video_path, audio_path, srt_path)


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_ffmpeg_failure_is_safe_and_removes_temporary_output(
    failure: str,
    storage_tree: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root, video_path, audio_path, srt_path = storage_tree

    def fake_run(command: list[str], **kwargs: object) -> object:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        Path(command[-1]).write_bytes(b"partial")
        return type(
            "Completed",
            (),
            {"returncode": 1, "stderr": "arbitrary ffmpeg details"},
        )()

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(
        service,
        "get_video_metadata",
        lambda _: {"format": {"duration": "1.0"}},
    )
    job_id = str(uuid4())

    with pytest.raises(RuntimeError) as error:
        generate_dubbed_video(job_id, video_path, audio_path, srt_path)

    assert "arbitrary ffmpeg details" not in str(error.value)
    output_directory = storage_root / "outputs" / job_id
    assert list(output_directory.iterdir()) == []


def test_output_path_cannot_overwrite_an_input(
    storage_tree: tuple[Path, Path, Path, Path],
) -> None:
    storage_root, _, audio_path, srt_path = storage_tree
    job_id = str(uuid4())
    video_path = storage_root / "outputs" / job_id / "hindi_dubbed.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"original")

    with pytest.raises(ValueError, match="must not overwrite"):
        generate_dubbed_video(job_id, video_path, audio_path, srt_path)


@pytest.mark.skipif(
    not RUN_REAL_VIDEO,
    reason="Set TRANSCADENCE_RUN_REAL_VIDEO=1 to run real FFmpeg video generation.",
)
def test_real_ffmpeg_video_generation() -> None:
    storage_root = service.STORAGE_ROOT
    uploads_directory = storage_root / "uploads"
    audio_directory = storage_root / "audio"
    subtitles_directory = storage_root / "subtitles"

    video_candidates = sorted(
        path
        for path in uploads_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm"}
    ) if uploads_directory.is_dir() else []
    audio_candidates = sorted(audio_directory.glob("*/tts/*.wav"))
    subtitle_candidates = sorted(subtitles_directory.glob("*/hi.srt"))

    if not video_candidates:
        pytest.skip("No MP4/WebM input exists under storage/uploads.")
    if not audio_candidates:
        pytest.skip("No Hindi TTS WAV exists under storage/audio.")
    if not subtitle_candidates:
        pytest.skip("No Hindi hi.srt exists under storage/subtitles.")

    video_path = video_candidates[0]
    audio_path = audio_candidates[0]
    subtitle_path = subtitle_candidates[0]
    job_id = str(uuid4())

    result = generate_dubbed_video(
        job_id,
        video_path,
        audio_path,
        subtitle_path,
    )

    output_path = Path(result["output_path"])
    output_metadata = get_video_metadata(output_path)
    format_names = output_metadata.get("format", {}).get("format_name", "")

    assert output_path == service.OUTPUT_ROOT / job_id / "hindi_dubbed.mp4"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert "mp4" in format_names.split(",")
    assert video_path.is_file()
    assert audio_path.is_file()
    assert subtitle_path.is_file()
    assert result["duration_seconds"] > 0

    print(f"input_video_path={video_path}")
    print(f"input_audio_path={audio_path}")
    print(f"subtitle_path={subtitle_path}")
    print(f"output_path={output_path}")
    print(f"output_size_bytes={output_path.stat().st_size}")
    print(f"output_duration_seconds={result['duration_seconds']:.3f}")