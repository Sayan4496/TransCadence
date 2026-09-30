import math

import pytest

from backend.app.services.sync_service import (
    MAX_DURATION_SECONDS,
    TOLERANCE_MS,
    calculate_cadence,
)


def test_matching_durations_are_within_tolerance() -> None:
    result = calculate_cadence(10.0, 10.0)

    assert result.speed_factor == 1.0
    assert result.adjusted_duration == 10.0
    assert result.sync_error_ms == 0.0
    assert result.is_within_tolerance is True
    assert result.status == "within_tolerance"


def test_generated_audio_longer_than_target() -> None:
    result = calculate_cadence(10.0, 12.0)

    assert result.duration_difference == 2.0
    assert result.speed_factor == 1.2
    assert result.status == "requires_adjustment"


def test_generated_audio_shorter_than_target() -> None:
    result = calculate_cadence(10.0, 8.0)

    assert result.duration_difference == -2.0
    assert result.speed_factor == 0.8
    assert result.status == "requires_adjustment"


def test_exactly_200_ms_difference_is_within_tolerance() -> None:
    result = calculate_cadence(10.0, 10.2)

    assert result.sync_error_ms == pytest.approx(TOLERANCE_MS)
    assert result.is_within_tolerance is True
    assert result.status == "within_tolerance"


def test_difference_over_200_ms_requires_adjustment() -> None:
    result = calculate_cadence(10.0, 10.201)

    assert result.sync_error_ms > TOLERANCE_MS
    assert result.is_within_tolerance is False
    assert result.status == "requires_adjustment"


@pytest.mark.parametrize(
    ("target_duration", "generated_duration"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (math.nan, 1.0),
        (1.0, math.nan),
        (math.inf, 1.0),
        (1.0, math.inf),
        (0.0001, 1.0),
        (1.0, 0.0001),
        (MAX_DURATION_SECONDS + 1, 1.0),
        (1.0, MAX_DURATION_SECONDS + 1),
    ],
)
def test_invalid_durations_are_rejected(
    target_duration: float,
    generated_duration: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_cadence(target_duration, generated_duration)


def test_speed_factor_calculation_for_both_directions() -> None:
    assert calculate_cadence(10.0, 12.0).speed_factor == pytest.approx(1.2)
    assert calculate_cadence(10.0, 8.0).speed_factor == pytest.approx(0.8)