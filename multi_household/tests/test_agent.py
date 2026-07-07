"""Unit tests for the appliance-aware rule controller.

Run:  pytest multi_household/tests/test_agent.py -v
"""
from __future__ import annotations
import pytest

from multi_household.agent.appliance_controller import (
    decide_step, _hour_bucket, _short_appliance,
    USER_REJECT_SUPPRESS_THRESHOLD,
)
from multi_household.tests.conftest import make_broadcast


# =========================================================================
# 1. Baseline: nothing fires when there's no peak event AND no forecast spike
# =========================================================================

def test_no_event_no_fire(state_washer, bc_mid_no_event):
    """Mid-price hour with no peak_event → agent does nothing."""
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_mid_no_event, step=0)
    assert d.action == "no_op"
    assert state_washer.release_pool_wh == 0.0


def test_no_appliance_running_no_fire(state_washer, bc_peak_evening):
    """Peak event but appliance is idle (< on_thr) → no fire."""
    appliance_loads = {"appliance_washing_machine_w": 10.0}    # < 50W
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)
    assert d.action == "no_op"


# =========================================================================
# 2. Rising-edge detection: only fires on the transition idle → running
# =========================================================================

def test_rising_edge_fires_defer(state_washer, bc_peak_evening):
    """0 → 500W (above 50W threshold) is a rising edge → fires defer."""
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    # state.last_appliance_w is empty → treated as 0 → 500 is a rising edge
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)
    assert d.action == "defer"
    assert d.target_appliance == "appliance_washing_machine_w"
    assert d.deferred_wh > 0.0


def test_no_edge_no_fire(state_washer, bc_peak_evening):
    """Already-running appliance (no rising edge) → no NEW defer
    (continuing defer is a different branch, not tested here because
     deferring_until hasn't been set yet)."""
    state_washer.last_appliance_w["appliance_washing_machine_w"] = 500.0
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)
    assert d.action == "no_op"          # last_w == w, not a rising edge


# =========================================================================
# 3. Cooldown: once a defer fires, can't refire same appliance immediately
# =========================================================================

def test_cooldown_blocks_immediate_refire(state_washer, bc_peak_evening):
    """Two consecutive rising edges with cooldown in between → only first
    one fires the new-defer branch."""
    appliance_loads = {"appliance_washing_machine_w": 500.0}

    # Step 0 — rising edge → defer
    d0 = decide_step(state_washer, appliance_loads,
                     forecast_w=400, broadcast=bc_peak_evening, step=0)
    assert d0.action == "defer"
    cd = state_washer.cooldown_until["appliance_washing_machine_w"]
    assert cd > 0

    # Reset last_w to simulate appliance turning off and on again quickly
    state_washer.last_appliance_w["appliance_washing_machine_w"] = 0.0
    bc1 = make_broadcast(hour=18, timestep=1, peak_event=True)
    d1 = decide_step(state_washer, appliance_loads,
                     forecast_w=400, broadcast=bc1, step=1)
    # cooldown is set to step + cycle_steps (washing_machine = 9), so step 1
    # is within cooldown → no new defer. But the appliance is still running
    # under deferring_until from step 0, so it should CONTINUE deferring.
    assert d1.action == "defer"
    assert d1.rationale.get("reason") == "continuing cycle defer"


# =========================================================================
# 4. Off-peak auto drain
# =========================================================================

def test_offpeak_drains_pool(state_washer, bc_offpeak_night):
    """Off-peak with pool > 0 → release happens, pool decrements."""
    state_washer.release_pool_wh = 300.0
    d = decide_step(state_washer, {}, forecast_w=100,
                    broadcast=bc_offpeak_night, step=100)
    assert d.action == "release"
    assert d.released_wh > 0
    assert state_washer.release_pool_wh < 300.0


def test_empty_pool_no_release(state_washer, bc_offpeak_night):
    """Off-peak but pool == 0 → nothing happens."""
    state_washer.release_pool_wh = 0.0
    d = decide_step(state_washer, {}, forecast_w=100,
                    broadcast=bc_offpeak_night, step=100)
    assert d.action == "no_op"


def test_hold_release_blocks_drain(state_washer, bc_offpeak_with_hold):
    """Aggregator says hold_release → pool stays full even in off-peak."""
    state_washer.release_pool_wh = 300.0
    d = decide_step(state_washer, {}, forecast_w=100,
                    broadcast=bc_offpeak_with_hold, step=100)
    assert d.action == "no_op"
    assert state_washer.release_pool_wh == 300.0


# =========================================================================
# 5. Energy conservation: defer N steps + release should sum to 0 (within ε)
# =========================================================================

def test_defer_and_release_conserve_energy(state_washer, bc_peak_evening):
    """Simulate: defer for cycle, then drain at off-peak. Total energy
    deferred == total energy released (within rounding)."""
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    total_deferred = 0.0
    total_released = 0.0

    # Step 0: rising edge → defer starts, deferring_until = 9
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)
    if d.action == "defer":
        total_deferred += d.deferred_wh

    # Steps 1-8: continuing defer (appliance still running)
    state_washer.last_appliance_w["appliance_washing_machine_w"] = 500.0
    for step in range(1, 9):
        bc = make_broadcast(hour=18, timestep=step, peak_event=True)
        d = decide_step(state_washer, appliance_loads,
                        forecast_w=400, broadcast=bc, step=step)
        if d.action == "defer":
            total_deferred += d.deferred_wh

    # Now we have ~9 steps of 500W * 1/6 h = ~750 Wh in pool
    pool_after_defer = state_washer.release_pool_wh
    assert pool_after_defer > 0
    assert abs(pool_after_defer - total_deferred) < 1e-6   # pool == total deferred

    # Now drain at off-peak. Pool draws pool/60 each step. Run enough to drain
    # almost everything (200 steps is enough).
    appliance_loads_idle = {"appliance_washing_machine_w": 0.0}
    for step in range(20, 220):
        # alternate off-peak hours
        bc = make_broadcast(hour=2, timestep=step, peak_event=False)
        d = decide_step(state_washer, appliance_loads_idle,
                        forecast_w=100, broadcast=bc, step=step)
        if d.action == "release":
            total_released += d.released_wh

    # After 200 drain steps, pool should be near 0
    assert state_washer.release_pool_wh < 1.0
    # Total released should equal total deferred (energy conservation)
    assert abs(total_released - total_deferred) < 1.0


# =========================================================================
# 6. Hour bucket logic (used for user_choices pattern matching)
# =========================================================================

@pytest.mark.parametrize("h, bucket", [
    (0, "off-peak"), (3, "off-peak"), (5, "off-peak"),
    (6, "mid"),     (12, "mid"),     (16, "mid"),
    (17, "peak"),   (20, "peak"),    (21, "peak"),
    (22, "mid"),    (23, "mid"),
])
def test_hour_bucket(h, bucket):
    assert _hour_bucket(h) == bucket


def test_short_appliance():
    assert _short_appliance("appliance_washing_machine_w") == "washing_machine"
    assert _short_appliance("appliance_dishwasher_w") == "dishwasher"


# =========================================================================
# 7. Closed loop: user rejection rate suppresses pattern
# =========================================================================

def test_user_rejection_suppresses_pattern(state_washer, bc_peak_evening):
    """If user has rejected washing_machine@peak > 50%, agent suppresses
    new defer recommendations for that pattern."""
    # Set rejection rate to 80% for the matching pattern
    state_washer.pattern_rejection_rate = {
        "washing_machine@peak": 0.80,
    }
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    n_suppressed_before = state_washer.n_suppressed_by_user_history

    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)

    # Should NOT have fired defer
    assert d.action == "no_op"
    # Counter should have incremented
    assert state_washer.n_suppressed_by_user_history > n_suppressed_before
    # Pool stays empty
    assert state_washer.release_pool_wh == 0.0


def test_user_rejection_below_threshold_does_not_suppress(state_washer, bc_peak_evening):
    """If rejection rate is BELOW threshold (e.g. 30%), pattern is NOT
    suppressed."""
    state_washer.pattern_rejection_rate = {
        "washing_machine@peak": 0.30,    # below 50% threshold
    }
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc_peak_evening, step=0)
    assert d.action == "defer"           # should still defer
    assert state_washer.n_suppressed_by_user_history == 0


def test_user_rejection_wrong_bucket_does_not_suppress(state_washer):
    """Rejection at PEAK doesn't suppress at OFF-PEAK."""
    state_washer.pattern_rejection_rate = {
        "washing_machine@peak": 1.00,    # 100% reject at peak
    }
    appliance_loads = {"appliance_washing_machine_w": 500.0}
    # Fire at hour 12 (mid bucket) — should NOT be suppressed
    bc = make_broadcast(hour=12, peak_event=True, p_now=0.15)
    d = decide_step(state_washer, appliance_loads,
                    forecast_w=400, broadcast=bc, step=0)
    assert d.action == "defer"


# --- fairness budget ---------------------------------------------------------

def test_fairness_budget_caps_daily_recommendations():
    from multi_household.agent.appliance_controller import (
        build_state, decide_step)
    from multi_household.aggregator.price_broadcast import Broadcast

    col = "appliance_washing_machine_w"
    state = build_state(1, [col])
    state.rec_budget_per_day = 1

    def bc(step):
        return Broadcast(timestep=step, hour=12, p_now_gbp_kwh=0.30,
                         p_off_gbp_kwh=0.08, peak_event=True,
                         aggregate_forecast_w=20000.0, threshold_w=18000.0,
                         overage_ratio=0.1)

    def edge(step):
        # off then on -> rising edge at the "on" step
        decide_step(state, {col: 0.0}, 500.0, bc(step), step,
                    accept_rate=1.0, current_demand_w=1000.0)
        return decide_step(state, {col: 400.0}, 500.0, bc(step + 1), step + 1,
                           accept_rate=1.0, current_demand_w=1000.0)

    d1 = edge(10)                       # first rec of the day → allowed
    assert d1.action == "defer"
    # let the deferring cycle + cooldown expire, then try again SAME day
    state.deferring_until.clear(); state.cooldown_until.clear()
    d2 = edge(60)                       # second rec same day → budget-blocked
    assert d2.action == "no_op"
    assert state.n_skipped_by_fairness >= 1
    # next day the budget resets
    state.deferring_until.clear(); state.cooldown_until.clear()
    state.ledger.clear(); state.release_pool_wh = 0.0
    d3 = edge(150)                      # step 150 // 144 = day 1 → allowed
    assert d3.action == "defer"
