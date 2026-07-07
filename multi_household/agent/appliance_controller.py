"""Appliance-aware rule controller — v2.

Improvements vs v1:
  • Per-appliance cooldown matching the physical cycle length
    (washer 90 min, dishwasher 60 min, dryer 60 min).
  • Cycle EDGE detection — only defer when an appliance just turned on
    (rising edge from <on_thr to >on_thr). Mid-cycle high-power readings
    no longer fire.
  • Defer DURATION tracking — each defer is logged as (step, energy),
    so we can report mean / P95 wait time and force-release if a cycle
    has been held too long.
  • Forecast-aware bias — agent slightly more aggressive when ŷ(t+1)
    indicates the next step will also be high.
  • Hold-release flag — when the aggregator says the grid is currently
    near limit (server feedback), agent skips release to avoid hitting
    a natural demand spike from the other side.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import random

from multi_household.aggregator.price_broadcast import Broadcast, is_off_peak


# --- per-appliance cycle parameters -----------------------------------------
# (on_threshold_w, cycle_steps, max_defer_steps)
# cycle_steps == cooldown == how long this cycle is treated as ONE decision.
# max_defer_steps == comfort cap: cycle force-released if held longer.
CYCLE_PARAMS: dict[str, tuple[float, int, int]] = {
    "washing_machine":  (50.0,  9, 36),    # 90 min cycle, 6h max wait
    "dishwasher":       (50.0,  6, 36),    # 60 min cycle, 6h max wait
    "tumble_dryer":     (80.0,  6, 24),    # 60 min cycle, 4h max wait
    "dryer":            (80.0,  6, 24),
    "washer_dryer":     (80.0, 12, 24),    # 120 min combined cycle
    "electric_heater":  (200.0, 6, 18),    # 60 min throttle, 3h max wait
    "water_heater":     (200.0, 6, 18),
    "synthetic_ev":     (1500.0, 24, 48),  # 4h charge, 8h max defer
}

POOL_DRAIN_STEPS = 60          # pool drains across 10 hours
SMALL_CHUNK_WH   = 1.0         # below this we dump remaining pool at once


def _appliance_params(col_name: str) -> Optional[tuple[float, int, int]]:
    n = col_name.lower()
    # Match the LONGEST key first so e.g. "washer_dryer" wins over "dryer"
    # (substring matching is order-sensitive otherwise).
    for key in sorted(CYCLE_PARAMS, key=len, reverse=True):
        if key in n:
            return CYCLE_PARAMS[key]
    return None


# --- data classes -----------------------------------------------------------

@dataclass
class DeferRecord:
    """One entry in the defer ledger — for tracking wait time per cycle."""
    appliance_col: str
    energy_wh: float
    deferred_at_step: int


@dataclass
class HouseAgentState:
    house_id: int
    deferable_cols: list[str]
    # ledger: pending defers in chronological order (FIFO)
    ledger: deque[DeferRecord] = field(default_factory=deque)
    # cooldown: next allowed step at which we can defer this appliance again
    cooldown_until: dict[str, int] = field(default_factory=dict)
    # last_appliance_w: previous step's appliance power, for edge detection
    last_appliance_w: dict[str, float] = field(default_factory=dict)
    # deferring_until[col] = step BEFORE which the appliance is still actively
    # being deferred (we keep subtracting its energy from served each step).
    deferring_until: dict[str, int] = field(default_factory=dict)
    # release_pool_wh: total deferred energy waiting to drain at off-peak
    release_pool_wh: float = 0.0
    # counters
    n_recommendations: int = 0
    n_accepted:        int = 0
    n_force_released:  int = 0
    n_suppressed_by_user_history: int = 0     # ← closed loop counter
    # release wait records: list of (wait_steps, energy_wh) for completed defers
    wait_log: list[tuple[int, float]] = field(default_factory=list)
    # ★ CLOSED LOOP — pattern_rejection_rate[(appliance_col, hour_bucket)] = float
    # Loaded from user_choices.json at start of rollout. If a pattern has
    # rejection rate > USER_REJECT_SUPPRESS_THRESHOLD, agent suppresses that
    # pattern in future recommendations.
    pattern_rejection_rate: dict[str, float] = field(default_factory=dict)
    # ★ FAIRNESS — per-house daily recommendation budget (None = unlimited).
    # Once a house has received `rec_budget_per_day` NEW recommendations in one
    # day, further rising-edge recommendations are skipped for the rest of that
    # day, spreading the burden across houses (Jain ↑ at some peak cost).
    rec_budget_per_day: Optional[int] = None
    budget_day: int = -1
    recs_today: int = 0
    n_skipped_by_fairness: int = 0


def _hour_bucket(hour: int) -> str:
    """Coarse hour bucket for pattern matching against user history."""
    if 0 <= hour < 6:   return "off-peak"
    if 17 <= hour < 22: return "peak"
    return "mid"


def _short_appliance(col: str) -> str:
    """Convert column name to short name used in user_choices.json."""
    return col.replace("appliance_", "").replace("_w", "")


USER_REJECT_SUPPRESS_THRESHOLD = 0.50    # > 50% reject rate → suppress
USER_REJECT_MIN_SAMPLES        = 2       # need at least 2 entries to count


@dataclass
class AgentDecision:
    action: str                                # no_op | defer | release
    target_appliance: Optional[str] = None
    deferred_wh: float = 0.0
    released_wh: float = 0.0
    expected_saving_gbp: float = 0.0
    rationale: dict = field(default_factory=dict)


# --- helpers ----------------------------------------------------------------

def build_state(house_id: int, deferable_cols: list[str]) -> HouseAgentState:
    return HouseAgentState(house_id=house_id, deferable_cols=deferable_cols)


def _accept(p: float) -> bool:
    if p >= 1.0:  return True
    if p <= 0.0:  return False
    return random.random() < p


def _pop_oldest_for_amount(state: HouseAgentState,
                            energy_wh: float,
                            current_step: int) -> None:
    """Take `energy_wh` out of the head of the ledger and log wait time(s)."""
    remaining = energy_wh
    while remaining > 1e-6 and state.ledger:
        head = state.ledger[0]
        take = min(head.energy_wh, remaining)
        wait = current_step - head.deferred_at_step
        state.wait_log.append((wait, take))
        head.energy_wh -= take
        remaining -= take
        if head.energy_wh <= 1e-6:
            state.ledger.popleft()


# --- decision rule ----------------------------------------------------------

def decide_step(state: HouseAgentState,
                appliance_loads_w: dict[str, float],
                forecast_w: float,
                broadcast: Broadcast,
                step: int,
                accept_rate: float = 1.0,
                current_demand_w: float = 0.0) -> AgentDecision:
    """One agent step.

    Order of operations:
      1. Force release any cycle that has been held too long (comfort cap).
      2. Auto release pool during off-peak — UNLESS the aggregator broadcast
         says hold_release (grid currently near limit, releasing now would
         add to a natural spike).
      3. Defer only at a RISING EDGE of a deferable appliance, under peak
         event, with cooldown clear.
    """
    # ---- update last_appliance_w trackers up-front ------------------------
    # We do this here so all later branches see consistent edge state.
    rising_edges: dict[str, float] = {}
    for col in state.deferable_cols:
        w = appliance_loads_w.get(col, 0.0)
        last_w = state.last_appliance_w.get(col, 0.0)
        params = _appliance_params(col)
        if params is not None:
            on_thr, _, _ = params
            if last_w <= on_thr and w > on_thr:
                rising_edges[col] = w
        state.last_appliance_w[col] = w

    # ---- 1. CONTINUE deferring any cycle already in progress --------------
    # If a previously-deferred cycle is still running (deferring_until > step),
    # we subtract THIS step's appliance energy from served and bank it.
    # This is what makes energy conservation work — we defer one step at a
    # time over the whole cycle, not the whole cycle in one shot.
    for col, end_step in list(state.deferring_until.items()):
        if end_step <= step:
            del state.deferring_until[col]
            continue
        w = appliance_loads_w.get(col, 0.0)
        if w <= 0.0:
            continue
        energy_wh = w * (10.0 / 60.0)
        state.release_pool_wh += energy_wh
        state.ledger.append(DeferRecord(
            appliance_col=col, energy_wh=energy_wh, deferred_at_step=step,
        ))
        return AgentDecision(
            action="defer",
            target_appliance=col,
            deferred_wh=energy_wh,
            rationale={"reason": "continuing cycle defer",
                       "ends_at_step": end_step,
                       "user_accepted": True,
                       "appliance_w_now": round(w, 1)},
        )

    # ---- 2. Comfort force-release: any cycle older than max_defer ---------
    for rec in list(state.ledger):
        params = _appliance_params(rec.appliance_col)
        if params is None: continue
        _, _, max_defer_steps = params
        if step - rec.deferred_at_step >= max_defer_steps:
            state.ledger.remove(rec)
            state.release_pool_wh = max(0.0, state.release_pool_wh - rec.energy_wh)
            state.n_force_released += 1
            wait = step - rec.deferred_at_step
            state.wait_log.append((wait, rec.energy_wh))
            return AgentDecision(
                action="release",
                target_appliance=rec.appliance_col,
                released_wh=rec.energy_wh,
                rationale={"reason": "comfort force release",
                           "waited_steps": wait},
            )

    # ---- 3. Auto release at off-peak, respecting hold_release -------------
    hold_release = getattr(broadcast, "hold_release", False)
    if is_off_peak(broadcast.hour) and state.release_pool_wh > 1e-6 and not hold_release:
        chunk = state.release_pool_wh / POOL_DRAIN_STEPS
        if chunk < SMALL_CHUNK_WH:
            chunk = state.release_pool_wh
        chunk = min(chunk, state.release_pool_wh)
        state.release_pool_wh -= chunk
        _pop_oldest_for_amount(state, chunk, step)
        return AgentDecision(
            action="release",
            released_wh=chunk,
            rationale={"reason": "auto off-peak drain",
                       "pool_remaining_wh": round(state.release_pool_wh, 2)},
        )

    # ---- 4. NEW defer on cycle edge under peak event ----------------------
    # Forecast-aware augmentation: fire locally if THIS house's own next-step
    # forecast is notably higher than its current whole-house demand. We compare
    # ŷ(t+1) against current_demand_w (the whole house), NOT against the sum of
    # submetered appliances — that earlier comparison was almost always true and
    # silently bypassed the peak_event gate.
    forecast_high = (current_demand_w > 0.0
                     and forecast_w > current_demand_w * 1.15)

    if (broadcast.peak_event or forecast_high) and rising_edges:
        # candidates: rising edges with cleared cooldown
        candidates = [(col, w) for col, w in rising_edges.items()
                      if state.cooldown_until.get(col, -1) <= step]

        # ★ CLOSED LOOP: filter out patterns user has consistently rejected
        bucket = _hour_bucket(broadcast.hour)
        filtered = []
        for col, w in candidates:
            short = _short_appliance(col)
            key   = f"{short}@{bucket}"
            rate  = state.pattern_rejection_rate.get(key, 0.0)
            if rate > USER_REJECT_SUPPRESS_THRESHOLD:
                state.n_suppressed_by_user_history += 1
                continue                              # skip this candidate
            filtered.append((col, w))
        candidates = filtered

        # ★ FAIRNESS budget: cap NEW recommendations per house per day
        if candidates and state.rec_budget_per_day is not None:
            day = step // 144
            if day != state.budget_day:
                state.budget_day = day
                state.recs_today = 0
            if state.recs_today >= state.rec_budget_per_day:
                state.n_skipped_by_fairness += len(candidates)
                candidates = []

        if candidates:
            col, w = max(candidates, key=lambda kv: kv[1])
            on_thr, cycle_steps, _ = _appliance_params(col)
            state.cooldown_until[col] = step + cycle_steps
            # Estimated full-cycle energy for saving estimate (display only)
            full_cycle_wh = w * (10.0 / 60.0) * cycle_steps
            saving = (broadcast.p_now_gbp_kwh - broadcast.p_off_gbp_kwh) \
                     * (full_cycle_wh / 1000.0)
            state.n_recommendations += 1
            state.recs_today += 1
            if _accept(accept_rate):
                state.n_accepted += 1
                # Mark this appliance as "being deferred for the next K steps".
                # The actual energy subtraction happens step-by-step in branch 1
                # above. This step contributes its own one-step chunk now.
                state.deferring_until[col] = step + cycle_steps
                this_step_wh = w * (10.0 / 60.0)
                state.release_pool_wh += this_step_wh
                state.ledger.append(DeferRecord(
                    appliance_col=col,
                    energy_wh=this_step_wh,
                    deferred_at_step=step,
                ))
                return AgentDecision(
                    action="defer",
                    target_appliance=col,
                    deferred_wh=this_step_wh,
                    expected_saving_gbp=round(saving, 3),
                    rationale={
                        "reason": ("peak event" if broadcast.peak_event
                                   else "forecast spike incoming"),
                        "appliance_w_now":  round(w, 1),
                        "cycle_steps":      cycle_steps,
                        "expected_cycle_wh": round(full_cycle_wh, 1),
                        "p_now":            broadcast.p_now_gbp_kwh,
                        "p_off":            broadcast.p_off_gbp_kwh,
                        "user_accepted":    True,
                        "pool_after_wh":    round(state.release_pool_wh, 2),
                    },
                )
            return AgentDecision(
                action="no_op",
                target_appliance=col,
                expected_saving_gbp=round(saving, 3),
                rationale={"reason": "peak event but user rejected",
                           "user_accepted": False},
            )

    return AgentDecision(action="no_op")
