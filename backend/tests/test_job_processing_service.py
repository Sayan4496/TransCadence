import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from backend.app.api import jobs as jobs_api
from backend.app.main import app
from backend.app.services import job_processing_service as processing
from backend.app.services import srt_service


@pytest.fixture
def job_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    storage_root = tmp_path / "storage"
    upload_dir = storage_root / "uploads"
    audio_dir = storage_root / "audio"
    transcript_dir = storage_root / "transcripts"
    output_dir = storage_root / "outputs"
    subtitle_dir = storage_root / "subtitles"
    job_id = str(uuid4())
    video_path = upload_dir / f"{job_id}.mp4"
    audio_path = audio_dir / f"{job_id}.wav"
    transcript_path = transcript_dir / f"{job_id}.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"original-video-with-english-audio")
    audio_path.write_bytes(b"extracted-original-audio")

    source_segments = [
        {
            "id": 1,
            "start": 0.0,
            "end": 3.0,
            "duration": 3.0,
            "text": "Hello there. How are you?",
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.4},
                {"word": " there.", "start": 0.45, "end": 0.9},
                {"word": " How", "start": 1.7, "end": 1.85},
                {"word": " are", "start": 1.9, "end": 2.05},
                {"word": " you?", "start": 2.1, "end": 2.4},
            ],
        },
        {
            "id": 2,
            "start": 4.0,
            "end": 5.0,
            "duration": 1.0,
            "text": "Welcome back.",
            "words": [
                {"word": " Welcome", "start": 4.0, "end": 4.4},
                {"word": " back.", "start": 4.45, "end": 4.9},
            ],
        },
    ]
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps({"language": "en", "segments": source_segments}),
        encoding="utf-8",
    )

    monkeypatch.setattr(processing, "STORAGE_ROOT", storage_root.resolve())
    monkeypatch.setattr(processing, "UPLOAD_DIR", upload_dir.resolve())
    monkeypatch.setattr(processing, "AUDIO_DIR", audio_dir.resolve())
    monkeypatch.setattr(processing, "TRANSCRIPT_DIR", transcript_dir.resolve())
    monkeypatch.setattr(processing, "OUTPUT_DIR", output_dir.resolve())
    monkeypatch.setattr(srt_service, "SUBTITLE_ROOT", subtitle_dir.resolve())
    monkeypatch.setattr(
        processing,
        "validate_video",
        lambda _: {"duration": 5.0, "metadata": {}},
    )
    monkeypatch.setattr(
        processing,
        "translate_segments",
        lambda chunks, language: [
            {**chunk, "translated_text": f"हिंदी {chunk['id']}"}
            for chunk in chunks
        ],
    )

    durations: dict[Path, float] = {audio_path.resolve(): 5.0}
    ffmpeg_commands: list[list[str]] = []
    final_video_audio_inputs: list[Path] = []

    def measure_duration(path: Path) -> float:
        return durations.get(path.resolve(), 1.2)

    monkeypatch.setattr(processing, "get_audio_duration", measure_duration)

    def fake_tts(text: str, requested_job_id: str, chunk_id: int) -> Path:
        path = audio_dir / requested_job_id / "tts" / f"{chunk_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"hindi-tts")
        durations[path.resolve()] = 1.2
        return path

    monkeypatch.setattr(processing, "generate_hindi_speech", fake_tts)

    def fake_adjust(
        input_path: Path,
        requested_job_id: str,
        chunk_id: int,
        speed_factor: float,
    ) -> SimpleNamespace:
        path = audio_dir / requested_job_id / "adjusted" / f"{chunk_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"adjusted-hindi")
        durations[path.resolve()] = 1.2 / speed_factor
        return SimpleNamespace(output_path=path)

    monkeypatch.setattr(processing, "adjust_audio_duration", fake_adjust)

    def fake_ffmpeg(command: list[str], **kwargs: object) -> SimpleNamespace:
        ffmpeg_commands.append(command)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == processing.FFMPEG_TIMEOUT_SECONDS
        output_path = Path(command[-1])
        output_path.write_bytes(b"hindi-only-timeline")
        pad_argument = command[command.index("-af") + 1]
        final_timeline_path = output_path.parent / "hindi_speech.wav"
        durations[final_timeline_path.resolve()] = float(pad_argument.split("=")[-1])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(processing.subprocess, "run", fake_ffmpeg)

    def fake_video(
        requested_job_id: str,
        source_video: Path,
        speech_audio: Path,
        subtitle_path: Path,
    ) -> dict[str, object]:
        final_video_audio_inputs.append(speech_audio.resolve())
        path = output_dir / requested_job_id / "hindi_dubbed.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"final-video")
        return {
            "job_id": requested_job_id,
            "output_path": str(path),
            "duration_seconds": 5.0,
        }

    monkeypatch.setattr(processing, "generate_dubbed_video", fake_video)

    return {
        "job_id": job_id,
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "source_segments": source_segments,
        "storage_root": storage_root,
        "durations": durations,
        "ffmpeg_commands": ffmpeg_commands,
        "final_video_audio_inputs": final_video_audio_inputs,
    }


def test_word_timing_splits_speech_around_pauses() -> None:
    segments = [
        {
            "id": 1,
            "start": 10.0,
            "end": 17.0,
            "words": [
                {"word": " First", "start": 10.0, "end": 11.0},
                {"word": " phrase.", "start": 11.1, "end": 13.0},
                {"word": " Second", "start": 14.5, "end": 15.0},
                {"word": " phrase.", "start": 15.1, "end": 17.0},
            ],
        }
    ]

    chunks = processing._build_speech_chunks(segments)

    assert [(chunk["start"], chunk["end"]) for chunk in chunks] == [
        (10.0, 13.0),
        (14.5, 17.0),
    ]
    assert chunks[0]["duration"] == 3.0
    assert chunks[1]["duration"] == 2.5
    assert chunks[1]["start"] - chunks[0]["end"] == 1.5


def test_zero_width_word_timestamp_is_kept_inside_measured_speech_window() -> None:
    chunks = processing._build_speech_chunks(
        [
            {
                "id": 5,
                "start": 46.54,
                "end": 47.26,
                "words": [
                    {"word": " So", "start": 46.54, "end": 46.54},
                    {"word": " never", "start": 46.54, "end": 46.78},
                    {"word": " give", "start": 46.78, "end": 46.98},
                    {"word": " up", "start": 46.98, "end": 47.26},
                ],
            }
        ]
    )

    assert len(chunks) == 1
    assert chunks[0]["start"] == 46.54
    assert chunks[0]["end"] == 47.26
    assert chunks[0]["text"] == "So never give up"


def test_cached_translation_is_reused_only_for_exact_one_window_source(
    job_setup: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_segment = {
        "id": 1,
        "start": 0.0,
        "end": 1.0,
        "duration": 1.0,
        "text": "Hello there.",
        "words": [
            {"word": " Hello", "start": 0.0, "end": 0.4},
            {"word": " there.", "start": 0.45, "end": 1.0},
        ],
    }
    source_segments = [source_segment]
    translated_segment = {**source_segment, "translated_text": "नमस्ते।"}
    cache_path = Path(job_setup["transcript_path"]).with_name(
        f"{job_setup['job_id']}_hi.json"
    )
    cache_path.write_text(
        json.dumps(
            {
                "job_id": job_setup["job_id"],
                "source_language": "en",
                "target_language": "hi",
                "segments": [translated_segment],
            }
        ),
        encoding="utf-8",
    )
    Path(job_setup["transcript_path"]).write_text(
        json.dumps({"language": "en", "segments": source_segments}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        processing,
        "translate_segments",
        lambda *_: (_ for _ in ()).throw(AssertionError("Gemini should not be called")),
    )

    result = processing.process_job(str(job_setup["job_id"]))

    assert result["translation_source"] == "cached"
    assert processing.get_job_status(str(job_setup["job_id"]))["translation_source"] == "cached"


def test_cached_translation_is_rejected_when_source_changes(
    job_setup: dict[str, object],
) -> None:
    source_segment = {
        "id": 1,
        "start": 0.0,
        "end": 1.0,
        "duration": 1.0,
        "text": "Current English.",
    }
    cache_path = Path(job_setup["transcript_path"]).with_name(
        f"{job_setup['job_id']}_hi.json"
    )
    cache_path.write_text(
        json.dumps(
            {
                "source_language": "en",
                "target_language": "hi",
                "segments": [
                    {
                        **source_segment,
                        "text": "Stale English.",
                        "translated_text": "पुराना अनुवाद।",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert processing._load_matching_hindi_translation(
        str(job_setup["job_id"]),
        [source_segment],
    ) is None


def test_cached_segment_translation_is_not_used_for_multiple_windows() -> None:
    source_segments = [
        {
            "id": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "First phrase. Second phrase.",
            "words": [
                {"word": " First", "start": 0.0, "end": 1.0},
                {"word": " phrase.", "start": 1.1, "end": 1.4},
                {"word": " Second", "start": 2.1, "end": 2.4},
                {"word": " phrase.", "start": 2.5, "end": 3.0},
            ],
        }
    ]
    chunks = processing._build_speech_chunks(source_segments)

    assert len(chunks) == 2
    assert processing._reuse_cached_chunk_translations(
        {1: "पहला वाक्य। दूसरा वाक्य।"},
        chunks,
        source_segments,
    ) is None


def test_full_pipeline_uses_hindi_only_audio_and_preserves_source_gaps(
    job_setup: dict[str, object],
) -> None:
    result = processing.process_job(str(job_setup["job_id"]))

    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["output_ready"] is True
    assert result["segment_count"] == 2
    assert result["speech_chunk_count"] == 3
    assert len(result["sync_validation"]) == 7
    assert {
        item["measurement"] for item in result["sync_validation"]
    } == {
        "speech_window",
        "subtitle_segment_timing",
        "final_audio_track_duration",
        "final_video_duration",
    }

    timeline_command = job_setup["ffmpeg_commands"][0]
    timeline_filter = timeline_command[
        timeline_command.index("-filter_complex") + 1
    ]
    assert "adelay=delays=1700:all=1" in timeline_filter
    assert "apad=whole_dur=5.000" in timeline_command
    assert str(job_setup["audio_path"].resolve()) not in timeline_command

    final_audio = job_setup["final_video_audio_inputs"]
    assert len(final_audio) == 1
    assert final_audio[0].name == "hindi_speech.wav"
    assert final_audio[0] != Path(job_setup["audio_path"]).resolve()
    assert not hasattr(processing, "mix_audio")

    original_segments = job_setup["source_segments"]
    translated_path = (
        Path(job_setup["storage_root"])
        / "transcripts"
        / f"{job_setup['job_id']}_hi.json"
    )
    translated = json.loads(translated_path.read_text(encoding="utf-8"))
    for original, output in zip(original_segments, translated["segments"]):
        assert output["start"] == original["start"]
        assert output["end"] == original["end"]
        assert output["text"] == original["text"]
        assert output["translated_text"]

    status = processing.get_job_status(str(job_setup["job_id"]))
    assert status["status"] == "completed"
    assert "storage" not in json.dumps(status)


def test_out_of_tolerance_speech_chunk_stops_before_video_generation(
    job_setup: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durations = job_setup["durations"]
    original_adjust = processing.adjust_audio_duration
    stale_output = (
        Path(job_setup["storage_root"])
        / "outputs"
        / str(job_setup["job_id"])
        / "hindi_dubbed.mp4"
    )
    stale_output.parent.mkdir(parents=True, exist_ok=True)
    stale_output.write_bytes(b"old-final-video")

    def out_of_sync_adjust(
        input_path: Path,
        requested_job_id: str,
        chunk_id: int,
        speed_factor: float,
    ) -> SimpleNamespace:
        result = original_adjust(input_path, requested_job_id, chunk_id, speed_factor)
        if chunk_id == 2:
            durations[result.output_path.resolve()] += 0.201
        return result

    monkeypatch.setattr(processing, "adjust_audio_duration", out_of_sync_adjust)
    with pytest.raises(processing.JobProcessingError) as raised:
        processing.process_job(str(job_setup["job_id"]))

    assert raised.value.stage == "sync_validation"
    assert raised.value.segment_id == 2
    assert len(job_setup["final_video_audio_inputs"]) == 0
    status = processing.get_job_status(str(job_setup["job_id"]))
    assert status["status"] == "failed"
    assert status["stage"] == "sync_validation"
    assert TestClient(app).get(
        f"/api/jobs/{job_setup['job_id']}/download"
    ).status_code == 404


def test_process_download_and_status_routes_are_job_scoped(
    job_setup: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs_api, "OUTPUT_ROOT", processing.OUTPUT_DIR)
    client = TestClient(app)

    process_response = client.post(f"/api/jobs/{job_setup['job_id']}/process")
    assert process_response.status_code == 200
    assert process_response.json()["status"] == "completed"

    status_response = client.get(f"/api/jobs/{job_setup['job_id']}/status")
    download_response = client.get(f"/api/jobs/{job_setup['job_id']}/download")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert download_response.status_code == 200
    assert download_response.content == b"final-video"
    assert "storage" not in status_response.text


def test_invalid_or_unknown_jobs_return_not_found() -> None:
    client = TestClient(app)
    assert client.get("/api/jobs/not-a-uuid/status").status_code == 404
    assert client.get(f"/api/jobs/{uuid4()}/download").status_code == 404