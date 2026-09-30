"""Calculate how generated audio fits an original transcript segment."""

from dataclasses import asdict, dataclass
import math
from typing import Literal


TOLERANCE_MS = 200.0
MIN_DURATION_SECONDS = 0.001
MAX_DURATION_SECONDS = 24 * 60 * 60

CadenceStatus = Literal[
    "within_tolerance",
    "requires_adjustment",
    "invalid",
]


@dataclass(frozen=True)
class CadenceResult:
    """Serializable cadence measurements for one transcript segment."""

    target_duration: float
    generated_duration: float
    duration_difference: float
    speed_factor: float
    adjusted_duration: float
    sync_error_ms: float
    tolerance_ms: float
    is_within_tolerance: bool
    status: CadenceStatus

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return the result as a JSON-compatible dictionary."""
        return asdict(self)


def _validate_duration(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number of seconds.")

    duration = float(value)
    if not math.isfinite(duration):
        raise ValueError(f"{name} must be finite.")
    if duration <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            f"{name} must be at least {MIN_DURATION_SECONDS} seconds."
        )
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"{name} must not exceed {MAX_DURATION_SECONDS} seconds."
        )

    return duration


def calculate_cadence(
    target_duration: float,
    generated_duration: float,
) -> CadenceResult:
    """Measure segment fit and recommend a future playback speed factor.

    The speed factor is generated duration divided by target duration. No
    audio adjustment is performed, so adjusted_duration records the current
    generated duration and sync_error_ms measures its current timing error.
    """
    target = _validate_duration(target_duration, "target_duration")
    generated = _validate_duration(generated_duration, "generated_duration")

    duration_difference = generated - target
    speed_factor = generated / target
    adjusted_duration = generated
    sync_error_ms = (adjusted_duration - target) * 1000
    is_within_tolerance = abs(sync_error_ms) <= TOLERANCE_MS

    return CadenceResult(
        target_duration=target,
        generated_duration=generated,
        duration_difference=duration_difference,
        speed_factor=speed_factor,
        adjusted_duration=adjusted_duration,
        sync_error_ms=sync_error_ms,
        tolerance_ms=TOLERANCE_MS,
        is_within_tolerance=is_within_tolerance,
        status=(
            "within_tolerance"
            if is_within_tolerance
            else "requires_adjustment"
        ),
    )