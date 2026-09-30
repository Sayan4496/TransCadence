import math
import os
from pathlib import Path
from uuid import uuid4
import wave

import pytest

from backend.app.services.sync_validation_service import validate_sync


JOB_ID = str(uuid4())
RUN_REAL_SYNC = os.getenv("TRANSCADENCE_RUN_REAL_SYNC") == "1"


def test_exact_match_passes() -> None:
    result = validate_sync(JOB_ID, 1, 10.0, 10.0)

    assert result.sync_error_ms == 0.0
    assert result.absolute_sync_error_ms == 0.0
    assert result.is_within_tolerance is True
    assert result.status == "PASS"


def test_100_ms_difference_passes() -> None:
    result = validate_sync(JOB_ID, 1, 10.0, 10.1)

    assert result.sync_error_ms == 100.0
    assert result.status == "PASS"


@pytest.mark.parametrize("actual_duration", [10.2, 9.8])
def test_exactly_200_ms_in_either_direction_passes(
    actual_duration: float,
) -> None:
    result = validate_sync(JOB_ID, 1, 10.0, actual_duration)

    assert result.absolute_sync_error_ms == 200.0
    assert result.is_within_tolerance is True
    assert result.status == "PASS"


@pytest.mark.parametrize("actual_duration", [10.200001, 9.799999])
def test_slightly_more_than_200_ms_fails(actual_duration: float) -> None:
    result = validate_sync(JOB_ID, 1, 10.0, actual_duration)

    assert result.absolute_sync_error_ms == pytest.approx(200.001)
    assert result.is_within_tolerance is False
    assert result.status == "FAIL"


@pytest.mark.parametrize(
    ("target_duration", "actual_duration", "expected_error"),
    [
        (10.0, 12.0, 2000.0),
        (10.0, 8.0, -2000.0),
    ],
)
def test_large_difference_fails(
    target_duration: float,
    actual_duration: float,
    expected_error: float,
) -> None:
    result = validate_sync(JOB_ID, 2, target_duration, actual_duration)

    assert result.sync_error_ms == expected_error
    assert result.is_within_tolerance is False
    assert result.status == "FAIL"


@pytest.mark.parametrize(
    ("target_duration", "actual_duration"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (math.nan, 1.0),
        (1.0, math.nan),
        (math.inf, 1.0),
        (1.0, math.inf),
        (-math.inf, 1.0),
        (1.0, -math.inf),
    ],
)
def test_invalid_durations_are_rejected(
    target_duration: float,
    actual_duration: float,
) -> None:
    with pytest.raises(ValueError):
        validate_sync(JOB_ID, 1, target_duration, actual_duration)


def test_invalid_uuid_is_rejected() -> None:
    with pytest.raises(ValueError, match="job_id"):
        validate_sync("not-a-uuid", 1, 1.0, 1.0)


@pytest.mark.parametrize("segment_id", [0, -1, True, 1.5])
def test_invalid_segment_id_is_rejected(segment_id: int) -> None:
    with pytest.raises(ValueError, match="segment_id"):
        validate_sync(JOB_ID, segment_id, 1.0, 1.0)


@pytest.mark.parametrize("tolerance_ms", [-0.001, math.nan, math.inf, -math.inf])
def test_invalid_tolerance_is_rejected(tolerance_ms: float) -> None:
    with pytest.raises(ValueError, match="tolerance_ms"):
        validate_sync(JOB_ID, 1, 1.0, 1.0, tolerance_ms)


def test_zero_tolerance_is_valid_for_exact_match() -> None:
    result = validate_sync(JOB_ID, 1, 2.0, 2.0, tolerance_ms=0.0)

    assert result.tolerance_ms == 0.0
    assert result.status == "PASS"


def test_result_is_json_compatible() -> None:
    result = validate_sync(JOB_ID, 3, 10.0, 10.1)
    serialized = result.to_dict()

    assert serialized["job_id"] == JOB_ID
    assert serialized["segment_id"] == 3
    assert serialized["tolerance_ms"] == 200.0
    assert serialized["status"] == "PASS"


@pytest.mark.skipif(
    not RUN_REAL_SYNC,
    reason="Set TRANSCADENCE_RUN_REAL_SYNC=1 to run measured-WAV validation.",
)
def test_real_sync_validation_with_measured_audio(tmp_path: Path) -> None:
    target_duration_seconds = 4.0
    sample_rate = 16000
    wav_path = tmp_path / "measured-target.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(sample_rate * target_duration_seconds))

    with wave.open(str(wav_path), "rb") as wav_file:
        measured_duration_seconds = wav_file.getnframes() / wav_file.getframerate()

    job_id = str(uuid4())
    result = validate_sync(
        job_id,
        1,
        target_duration_seconds,
        measured_duration_seconds,
    )

    assert measured_duration_seconds > 0
    assert result.sync_error_ms == pytest.approx(
        (measured_duration_seconds - target_duration_seconds) * 1000
    )
    assert result.absolute_sync_error_ms <= 200
    assert result.status == "PASS"
    assert result.is_within_tolerance is True

    print(f"target_duration_seconds={target_duration_seconds:.3f}")
    print(f"measured_wav_duration_seconds={measured_duration_seconds:.3f}")
    print(f"sync_error_ms={result.sync_error_ms:.3f}")
    print(f"absolute_sync_error_ms={result.absolute_sync_error_ms:.3f}")
    print(f"tolerance_ms={result.tolerance_ms:.3f}")
    print(f"status={result.status}")

    failing_result = validate_sync(
        job_id,
        1,
        target_duration_seconds,
        measured_duration_seconds + 0.201,
    )
    assert failing_result.status == "FAIL"
    assert failing_result.is_within_tolerance is False
    assert failing_result.absolute_sync_error_ms == pytest.approx(201.0)
    print(f"modified_absolute_sync_error_ms={failing_result.absolute_sync_error_ms:.3f}")
    print(f"modified_status={failing_result.status}")