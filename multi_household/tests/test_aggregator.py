"""Unit tests for the aggregator broadcast logic.

Run:  pytest multi_household/tests/test_aggregator.py -v
"""
from __future__ import annotations
import pytest

from multi_household.aggregator.price_broadcast import (
    aggregate_and_price, base_tou_price, is_off_peak,
)
from multi_household.config import (
    PEAK_PRICE_GBP, MID_PRICE_GBP, OFFPEAK_PRICE_GBP,
    GRID_THRESHOLD_W,
)


def test_below_threshold_no_peak_event():
    """Aggregate forecast below threshold → peak_event = False."""
    forecasts = [1000.0] * 5             # 5 kW total, far below threshold
    bc = aggregate_and_price(forecasts, timestep=0, hour=18)
    assert bc.peak_event is False
    assert bc.overage_ratio == 0.0
    # Price should be the standard ToU for hour 18 (peak hours = 17-22)
    assert bc.p_now_gbp_kwh == PEAK_PRICE_GBP


def test_above_threshold_triggers_peak_event():
    """Aggregate forecast above threshold → peak_event = True, price up."""
    forecasts = [GRID_THRESHOLD_W + 5000.0]    # well above threshold
    bc = aggregate_and_price(forecasts, timestep=0, hour=18)
    assert bc.peak_event is True
    assert bc.overage_ratio > 0
    # Price should be ABOVE the base ToU
    assert bc.p_now_gbp_kwh > PEAK_PRICE_GBP


def test_hold_release_triggered_by_prev_served():
    """Prev served near limit → hold_release = True even if Y < threshold."""
    forecasts = [1000.0] * 5             # low Y
    bc = aggregate_and_price(forecasts, timestep=0, hour=2,
                              prev_served_w=0.90 * GRID_THRESHOLD_W)
    assert bc.hold_release is True       # prev_served exceeds 85% threshold


def test_hold_release_not_triggered_when_low():
    """Prev served far below limit → no hold_release."""
    forecasts = [1000.0] * 5
    bc = aggregate_and_price(forecasts, timestep=0, hour=2,
                              prev_served_w=2000.0)
    assert bc.hold_release is False


# --- ToU pricing ----------------------------------------------------------

@pytest.mark.parametrize("hour, expected", [
    (0, OFFPEAK_PRICE_GBP),  (3, OFFPEAK_PRICE_GBP),  (5, OFFPEAK_PRICE_GBP),
    (6, MID_PRICE_GBP),      (12, MID_PRICE_GBP),     (16, MID_PRICE_GBP),
    (17, PEAK_PRICE_GBP),    (20, PEAK_PRICE_GBP),    (21, PEAK_PRICE_GBP),
    (22, MID_PRICE_GBP),     (23, MID_PRICE_GBP),
])
def test_base_tou_price(hour, expected):
    assert base_tou_price(hour) == expected


@pytest.mark.parametrize("hour, expected", [
    (0, True), (3, True), (5, True),
    (6, False), (12, False), (17, False), (22, False),
])
def test_is_off_peak(hour, expected):
    assert is_off_peak(hour) == expected
