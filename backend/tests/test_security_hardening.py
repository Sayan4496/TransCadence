import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


def _generate_valid_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for media validation tests.")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_upload_rejects_fake_mp4_file_before_processing() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("fake.mp4", b"this is not a real mp4 file", "video/mp4")},
    )

    assert response.status_code == 400
    detail = response.json().get("detail")
    assert isinstance(detail, str)
    assert "video" in detail.lower()


def test_upload_ignores_misleading_browser_mime_type(tmp_path: Path) -> None:
    client = TestClient(app)
    video_path = tmp_path / "transcadence_wrong_mime_test.mp4"
    _generate_valid_video(video_path)

    with video_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("../../safe_video.mp4", handle.read(), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["job_id"]
    assert payload["stored_filename"].endswith(".mp4")
    assert ".." not in payload["stored_filename"]


def test_path_traversal_filename_is_sanitized_to_job_scoped_storage(tmp_path: Path) -> None:
    client = TestClient(app)
    video_path = tmp_path / "transcadence_traversal_test.mp4"
    _generate_valid_video(video_path)

    with video_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("../../evil.mp4", handle.read(), "video/mp4")},
        )

    assert response.status_code == 200, response.text
    stored_name = response.json()["stored_filename"]
    assert stored_name != "../../evil.mp4"
    assert ".." not in stored_name
    assert stored_name.endswith(".mp4")


def test_malicious_job_ids_fail_safely() -> None:
    client = TestClient(app)
    for path in [
        "/api/jobs/../../status",
        "/api/jobs/C:/evil/status",
        "/api/jobs/not-a-uuid/status",
        "/api/jobs/01234567-89ab-cdef-0123-456789abcdef/download",
    ]:
        response = client.get(path)
        assert response.status_code == 404
        assert "traceback" not in response.text.lower()
        assert "storage" not in response.text.lower()


def test_download_route_rejects_non_job_files() -> None:
    client = TestClient(app)
    response = client.get("/api/jobs/00000000-0000-0000-0000-000000000000/download")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    assert "storage" not in response.text.lower()


def test_upload_rejects_oversized_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("too_big.mp4", b"x" * (250 * 1024 * 1024), "video/mp4")},
    )

    assert 400 <= response.status_code < 500
    assert "limit" in response.json()["detail"].lower() or "video" in response.json()["detail"].lower()
