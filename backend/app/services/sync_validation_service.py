"""Validate measured audio synchronization against a target duration."""

from dataclasses import asdict, dataclass
from decimal import Decimal
import math
from uuid import UUID


DEFAULT_TOLERANCE_MS = 200.0


@dataclass(frozen=True)
class SyncValidationResult:
    """Measured synchronization result for one generated audio segment."""

    job_id: str
    segment_id: int
    target_duration_seconds: float
    actual_audio_duration_seconds: float
    duration_difference_seconds: float
    sync_error_ms: float
    absolute_sync_error_ms: float
    tolerance_ms: float
    is_within_tolerance: bool
    status: str

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return the result as a JSON-compatible dictionary."""
        return asdict(self)


def _validate_duration(value: float, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number of seconds.")

    duration = float(value)
    if not math.isfinite(duration):
        raise ValueError(f"{name} must be finite.")
    if duration <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return Decimal(str(value))


def _validate_tolerance(tolerance_ms: float) -> Decimal:
    if isinstance(tolerance_ms, bool) or not isinstance(tolerance_ms, (int, float)):
        raise ValueError("tolerance_ms must be a finite number.")

    tolerance = float(tolerance_ms)
    if not math.isfinite(tolerance):
        raise ValueError("tolerance_ms must be finite.")
    if tolerance < 0:
        raise ValueError("tolerance_ms must be non-negative.")

    return Decimal(str(tolerance_ms))


def validate_sync(
    job_id: str,
    segment_id: int,
    target_duration_seconds: float,
    actual_audio_duration_seconds: float,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
) -> SyncValidationResult:
    """Compare measured audio duration to a target using a precise tolerance."""
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a UUID string.")
    try:
        safe_job_id = str(UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a valid UUID.") from exc

    if isinstance(segment_id, bool) or not isinstance(segment_id, int):
        raise ValueError("segment_id must be a positive integer.")
    if segment_id <= 0:
        raise ValueError("segment_id must be a positive integer.")

    target = _validate_duration(target_duration_seconds, "target_duration_seconds")
    actual = _validate_duration(
        actual_audio_duration_seconds,
        "actual_audio_duration_seconds",
    )
    tolerance = _validate_tolerance(tolerance_ms)

    difference = actual - target
    sync_error = difference * Decimal(1000)
    absolute_sync_error = abs(sync_error)
    is_within_tolerance = absolute_sync_error <= tolerance

    return SyncValidationResult(
        job_id=safe_job_id,
        segment_id=segment_id,
        target_duration_seconds=float(target),
        actual_audio_duration_seconds=float(actual),
        duration_difference_seconds=float(difference),
        sync_error_ms=float(sync_error),
        absolute_sync_error_ms=float(absolute_sync_error),
        tolerance_ms=float(tolerance),
        is_within_tolerance=is_within_tolerance,
        status="PASS" if is_within_tolerance else "FAIL",
    )