"""Shared pytest fixtures — small, fast, self-contained.
NO real REFIT loading, NO model training. Everything synthetic."""
from __future__ import annotations
import pytest

from multi_household.agent.appliance_controller import (
    HouseAgentState, build_state,
)
from multi_household.aggregator.price_broadcast import Broadcast


# --- minimal broadcasts ---------------------------------------------------

def make_broadcast(*,
                   hour: int = 18,
                   timestep: int = 0,
                   peak_event: bool = True,
                   hold_release: bool = False,
                   p_now: float = 0.30,
                   p_off: float = 0.08,
                   aggregate_w: float = 10000.0,
                   overage: float = 0.0) -> Broadcast:
    return Broadcast(
        timestep=timestep, hour=hour,
        p_now_gbp_kwh=p_now, p_off_gbp_kwh=p_off,
        peak_event=peak_event,
        aggregate_forecast_w=aggregate_w,
        threshold_w=12500.0,
        overage_ratio=overage,
        hold_release=hold_release,
    )


# --- minimal house states -------------------------------------------------

@pytest.fixture
def state_washer():
    """A house with one deferable appliance (washing machine)."""
    return build_state(house_id=99,
                       deferable_cols=["appliance_washing_machine_w"])


@pytest.fixture
def state_multi():
    """A house with washer + dishwasher + dryer."""
    return build_state(
        house_id=99,
        deferable_cols=["appliance_washing_machine_w",
                        "appliance_dishwasher_w",
                        "appliance_tumble_dryer_w"],
    )


# --- broadcast fixtures ---------------------------------------------------

@pytest.fixture
def bc_peak_evening():
    """Peak event at 18:00 (evening peak)."""
    return make_broadcast(hour=18, peak_event=True)


@pytest.fixture
def bc_offpeak_night():
    """Off-peak at 02:00 (no peak event)."""
    return make_broadcast(hour=2, peak_event=False,
                          p_now=0.08, aggregate_w=5000.0)


@pytest.fixture
def bc_offpeak_with_hold():
    """Off-peak but aggregator says hold release (natural spike)."""
    return make_broadcast(hour=2, peak_event=False, hold_release=True)


@pytest.fixture
def bc_mid_no_event():
    """Mid-price hour, no peak event."""
    return make_broadcast(hour=12, peak_event=False, p_now=0.15)
