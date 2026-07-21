"""EV smart-charging advisory coordinator (aggregator side).

Problem this solves
-------------------
The synthetic EVs all plug in between 21:00 and 23:50 and charge for 4 h at
7 kW. With 5 EV households that is up to 35 kW piling onto the same overnight
window — and naive per-house deferral just shuffles it within that same
window, creating a rebound peak.

What this does (honest scope)
-----------------------------
A fixed-interval STAGGER heuristic: for each night, EV blocks are offset by
STAGGER_STEPS (2 h) from the earliest plug-in. Because the stagger interval
(2 h) is SHORTER than the charge duration (4 h), adjacent accepted blocks
still overlap by up to 2 h (≤14 kW EV concurrency by design) — the heuristic
reduces the pile-up, it does NOT eliminate overlap. It is not a
constraint-aware scheduler: no SoC, arrival/departure, deadline, charger
efficiency or transformer limit is modelled. Total energy per EV is preserved
exactly — the block is moved in time, not resized.

The non-EV appliances are still handled by the per-house rule agent.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


STAGGER_STEPS = 12           # 2 h between successive EV starts (10-min steps)


def _detect_blocks(ev: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start_idx, length), ...] for each contiguous >0 run."""
    blocks = []
    i, n = 0, len(ev)
    while i < n:
        if ev[i] > 0:
            j = i
            while j < n and ev[j] > 0:
                j += 1
            blocks.append((i, j - i))
            i = j
        else:
            i += 1
    return blocks


# Fixed seed so the per-night EV accept decisions are reproducible AND monotone
# in accept_rate (a block accepted at 0.5 is still accepted at 0.85).
EV_ACCEPT_SEED = 20260710


TROUGH_STEPS = 60            # 10 h overnight placement span for baselines


def advisory_ev_schedule(ev_orig_by_house: dict[int, np.ndarray],
                         timestamps: pd.DatetimeIndex,
                         accept_rate: float = 1.0,
                         stagger_steps: int = STAGGER_STEPS,
                         seed: int = EV_ACCEPT_SEED,
                         strategy: str = "stagger"):
    """Advisory (human-in-the-loop) EV stagger. accept_rate=1.0 is the fully
    automatic schedule; below that, each night's reschedule is user-gated.

    Each night's EV reschedule is a RECOMMENDATION the user accepts with
    probability `accept_rate`:
      • accepted → the EV block is moved to its staggered slot,
      • rejected → the EV stays at its natural time (untouched).

    So user acceptance genuinely drives peak shaving: accept_rate 0 leaves the
    midnight pile-up intact, accept_rate 1 fully staggers it.

    Returns (ev_orig_applied, ev_shift_applied, decisions):
      • ev_orig_applied / ev_shift_applied: {house: (T,) watts}, populated ONLY
        for accepted blocks — the caller computes
        served = demand − ev_orig_applied + ev_shift_applied, so rejected
        blocks (both zero) leave the natural EV in `demand`; energy conserved.
      • decisions: one dict PER nightly recommendation (accepted or not) —
        {house, night, start_idx, length_steps, accepted, new_start_idx}.
        These are real user decision events and MUST be logged by the caller
        (they were previously invisible to all comfort/fairness metrics).

    Baseline strategies (`strategy` param) share the IDENTICAL accept-draw
    stream (one uniform per block from `rng`, drawn unconditionally), so
    stagger / random / edf comparisons are PAIRED — the same nightly
    recommendations get accepted in every arm; only placement differs:
      • "stagger" (ours): fixed 2 h offsets from the night's earliest plug-in
      • "random":  uniform placement in the 10 h overnight span (separate rng)
      • "edf":     earliest-deadline-first greedy — deadline = natural start
                   + 8 h comfort cap; pick the earliest slot minimising
                   overlap with already-placed blocks that still finishes by
                   the deadline
    """
    if strategy not in ("stagger", "random", "edf"):
        raise ValueError(f"unknown strategy {strategy}")
    rng = np.random.default_rng(seed)
    place_rng = np.random.default_rng(seed + 1)   # placement only — keeps the
    #                                               accept stream untouched
    houses = sorted(ev_orig_by_house)
    T = len(timestamps)
    ev_orig_app  = {h: np.zeros(T, dtype=np.float32) for h in houses}
    ev_shift_app = {h: np.zeros(T, dtype=np.float32) for h in houses}

    night_blocks: dict[object, list] = defaultdict(list)
    for h in houses:
        for (start, length) in _detect_blocks(ev_orig_by_house[h]):
            power = float(ev_orig_by_house[h][start])
            night_blocks[timestamps[start].date()].append((h, start, length, power))

    MAX_DEFER = 48                                # 8 h comfort cap (steps)
    decisions: list[dict] = []
    for night, blocks in night_blocks.items():
        blocks.sort(key=lambda b: b[1])
        anchor = blocks[0][1]
        occupancy = np.zeros(T + TROUGH_STEPS + MAX_DEFER, dtype=np.int32)
        for rank, (h, start, length, power) in enumerate(blocks):
            accepted = rng.random() < accept_rate
            if strategy == "stagger":
                new_start = anchor + rank * stagger_steps
            elif strategy == "random":
                span = max(TROUGH_STEPS - length, 1)
                new_start = anchor + int(place_rng.integers(0, span))
            else:                                 # edf greedy
                deadline_start = start + MAX_DEFER            # latest allowed start
                best, best_ov = anchor, None
                for cand in range(anchor, min(deadline_start, anchor + TROUGH_STEPS) + 1):
                    ov = int(occupancy[cand:cand + length].max())
                    if best_ov is None or ov < best_ov:
                        best, best_ov = cand, ov
                    if ov == 0:
                        break
                new_start = best
            new_start = max(0, min(new_start, T - length)) if length <= T else 0
            decisions.append({
                "house": int(h),
                "night": str(night),
                "start_idx": int(start),
                "length_steps": int(length),
                "accepted": bool(accepted),
                "new_start_idx": int(new_start) if accepted else None,
            })
            if not accepted:
                continue                        # rejected → EV stays natural
            occupancy[new_start:new_start + length] += 1
            end = min(new_start + length, T)
            ev_orig_app[h][start:min(start + length, T)] = power   # remove natural
            ev_shift_app[h][new_start:end] = power                 # add staggered
    return ev_orig_app, ev_shift_app, decisions
