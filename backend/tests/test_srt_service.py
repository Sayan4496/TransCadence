r"""Tests for Hindi SRT generation.

Opt into a real job-scoped file write with:
    $env:TRANSCADENCE_RUN_REAL_SRT = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_srt_service.py::test_real_srt_write -q -s
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.services import srt_service
from backend.app.services.srt_service import build_srt, write_srt


RUN_REAL_SRT = os.getenv("TRANSCADENCE_RUN_REAL_SRT") == "1"


@pytest.fixture
def subtitle_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "subtitles"
    monkeypatch.setattr(srt_service, "SUBTITLE_ROOT", root)
    return root


def segment(
    segment_id: int = 1,
    start: float = 10.45,
    end: float = 21.45,
    text: str = "नमस्ते, सभी को।",
) -> dict[str, object]:
    return {
        "id": segment_id,
        "start": start,
        "end": end,
        "text": "Hello everyone.",
        "translated_text": text,
    }


def test_valid_single_segment() -> None:
    assert build_srt([segment()]) == (
        "1\n"
        "00:00:10,450 --> 00:00:21,450\n"
        "नमस्ते, सभी को।\n"
    )


def test_multiple_segments_use_sequential_srt_indices() -> None:
    content = build_srt(
        [
            segment(segment_id=8, start=0, end=1.25, text="पहला वाक्य।"),
            segment(segment_id=9, start=1.25, end=2.5, text="दूसरा वाक्य।"),
        ]
    )

    assert content == (
        "1\n00:00:00,000 --> 00:00:01,250\nपहला वाक्य।\n\n"
        "2\n00:00:01,250 --> 00:00:02,500\nदूसरा वाक्य।\n"
    )


def test_translated_text_is_used_and_unicode_is_preserved() -> None:
    content = build_srt([segment(text="नमस्ते\nदुनिया 🌍")])

    assert "नमस्ते\nदुनिया 🌍" in content
    assert "Hello everyone." not in content


def test_millisecond_formatting() -> None:
    assert build_srt([segment(start=10.45, end=21.45)]).splitlines()[1] == (
        "00:00:10,450 --> 00:00:21,450"
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (60.0, "00:01:00,000"),
        (3600.0, "01:00:00,000"),
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
        (1.2345, "00:00:01,235"),
    ],
)
def test_timestamp_boundaries_and_rounding(seconds: float, expected: str) -> None:
    assert srt_service._format_timestamp(seconds) == expected


def test_zero_start_time_is_valid() -> None:
    assert build_srt([segment(start=0, end=1)]).splitlines()[1].startswith(
        "00:00:00,000"
    )


def test_non_list_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="list"):
        build_srt((segment(),))  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_segment", [None, "segment", 3])
def test_invalid_segment_type_is_rejected(invalid_segment: object) -> None:
    with pytest.raises(ValueError, match="mapping"):
        build_srt([invalid_segment])  # type: ignore[list-item]


def test_missing_required_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        build_srt([{"id": 1, "start": 0, "end": 1}])  # type: ignore[list-item]


@pytest.mark.parametrize("segment_id", [0, -1, True, 1.5])
def test_invalid_segment_id_is_rejected(segment_id: object) -> None:
    with pytest.raises(ValueError, match="id must be a positive integer"):
        build_srt([segment(segment_id=segment_id)])  # type: ignore[arg-type]


def test_negative_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="start must be non-negative"):
        build_srt([segment(start=-0.01, end=1)])


@pytest.mark.parametrize(("start", "end"), [(1, 1), (2, 1)])
def test_end_must_be_greater_than_start(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="end must be greater than start"):
        build_srt([segment(start=start, end=end)])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("start", float("nan")),
        ("end", float("nan")),
        ("start", float("inf")),
        ("end", float("-inf")),
    ],
)
def test_nan_and_infinity_are_rejected(field: str, invalid_value: float) -> None:
    values = {"start": 0.0, "end": 1.0}
    values[field] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        build_srt([segment(**values)])  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_subtitle_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_srt([segment(text=text)])


def test_text_fallback_preserves_internal_line_breaks() -> None:
    value = {"id": 1, "start": 0, "end": 1, "text": "पहली पंक्ति\nदूसरी पंक्ति"}

    assert build_srt([value]) == (
        "1\n00:00:00,000 --> 00:00:01,000\nपहली पंक्ति\nदूसरी पंक्ति\n"
    )


def test_empty_translated_text_does_not_fall_back_to_english() -> None:
    value = segment(text="")

    with pytest.raises(ValueError, match="must not be empty"):
        build_srt([value])


def test_overlapping_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        build_srt(
            [
                segment(start=0, end=2),
                segment(start=1.5, end=3),
            ]
        )


def test_incorrectly_ordered_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordered by start time"):
        build_srt(
            [
                segment(start=2, end=3),
                segment(start=0, end=1),
            ]
        )


def test_invalid_job_uuid_is_rejected(subtitle_root: Path) -> None:
    with pytest.raises(ValueError, match="job_id"):
        write_srt([segment()], "../outside")


def test_unsupported_language_is_rejected(subtitle_root: Path) -> None:
    with pytest.raises(ValueError, match="Only Hindi"):
        write_srt([segment()], str(uuid4()), language="bn")


def test_output_path_is_isolated_by_job_id(subtitle_root: Path) -> None:
    first_job_id = str(uuid4())
    second_job_id = str(uuid4())

    first = write_srt([segment()], first_job_id)
    second = write_srt([segment()], second_job_id)

    assert first.output_path == subtitle_root / first_job_id / "hi.srt"
    assert second.output_path == subtitle_root / second_job_id / "hi.srt"
    assert first.output_path != second.output_path


def test_job_directory_symlink_cannot_redirect_output(
    subtitle_root: Path,
) -> None:
    job_id = str(uuid4())
    other_job_directory = subtitle_root / str(uuid4())
    other_job_directory.mkdir(parents=True)
    job_directory_link = subtitle_root / job_id

    try:
        job_directory_link.symlink_to(other_job_directory, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="must not redirect"):
        write_srt([segment()], job_id)


def test_srt_write_is_atomic(
    subtitle_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = srt_service.os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def observe_replace(source: Path, destination: Path) -> None:
        assert source.is_file()
        assert source.parent == destination.parent
        replace_calls.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(srt_service.os, "replace", observe_replace)
    result = write_srt([segment()], str(uuid4()))

    assert len(replace_calls) == 1
    assert result.output_path.is_file()
    assert not replace_calls[0][0].exists()


def test_generated_srt_content_and_result(
    subtitle_root: Path,
) -> None:
    job_id = str(uuid4())
    result = write_srt([segment()], job_id)

    assert result.job_id == job_id
    assert result.language == "hi"
    assert result.segment_count == 1
    assert result.output_path.read_text(encoding="utf-8") == (
        "1\n00:00:10,450 --> 00:00:21,450\nनमस्ते, सभी को।\n"
    )


@pytest.mark.skipif(
    not RUN_REAL_SRT,
    reason="Set TRANSCADENCE_RUN_REAL_SRT=1 to write a real job-scoped SRT file.",
)
def test_real_srt_write() -> None:
    """Write and read a Hindi SRT under the actual subtitle storage directory."""
    job_id = str(uuid4())
    result = write_srt([segment()], job_id)
    content = result.output_path.read_text(encoding="utf-8")

    assert result.output_path.is_file()
    assert "नमस्ते, सभी को।" in content
    assert "1\n00:00:10,450 --> 00:00:21,450\n" in content