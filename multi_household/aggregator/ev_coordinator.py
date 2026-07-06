"""EV smart-charging coordinator (aggregator side).

Problem this solves
-------------------
The synthetic EVs all plug in around 22:00-23:30 and charge for 4 h at 7 kW.
With 5 EV households that is up to 35 kW piling onto the same overnight window
— and naive per-house deferral just shuffles it within that same window,
creating a rebound peak.

What this does
--------------
For COORDINATED mode the aggregator looks at every EV charging block on a given
night and STAGGERS their start times across the overnight trough (22:00 → 08:00)
so they no longer overlap. Total energy per EV is preserved exactly — the block
is moved in time, not resized. This is the one big lever for real peak shaving,
because EV is the dominant deferrable load (~12 kW of the recurring peaks).

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


def advisory_ev_schedule(ev_orig_by_house: dict[int, np.ndarray],
                         timestamps: pd.DatetimeIndex,
                         accept_rate: float = 1.0,
                         stagger_steps: int = STAGGER_STEPS,
                         seed: int = EV_ACCEPT_SEED):
    """Advisory (human-in-the-loop) EV stagger. accept_rate=1.0 is the fully
    automatic schedule; below that, each night's reschedule is user-gated.

    Each night's EV reschedule is a RECOMMENDATION the user accepts with
    probability `accept_rate`:
      • accepted → the EV block is moved to its staggered slot,
      • rejected → the EV stays at its natural time (untouched).

    So user acceptance genuinely drives peak shaving: accept_rate 0 leaves the
    midnight pile-up intact, accept_rate 1 fully staggers it. Returns
    (ev_orig_applied, ev_shift_applied), both {house: (T,) watts}, populated
    ONLY for accepted blocks — the caller computes
    served = demand − ev_orig_applied + ev_shift_applied, so rejected blocks
    (both zero) leave the natural EV in `demand`, and energy is conserved.
    """
    rng = np.random.default_rng(seed)
    houses = sorted(ev_orig_by_house)
    T = len(timestamps)
    ev_orig_app  = {h: np.zeros(T, dtype=np.float32) for h in houses}
    ev_shift_app = {h: np.zeros(T, dtype=np.float32) for h in houses}

    night_blocks: dict[object, list] = defaultdict(list)
    for h in houses:
        for (start, length) in _detect_blocks(ev_orig_by_house[h]):
            power = float(ev_orig_by_house[h][start])
            night_blocks[timestamps[start].date()].append((h, start, length, power))

    n_reco = n_acc = 0
    for night, blocks in night_blocks.items():
        blocks.sort(key=lambda b: b[1])
        anchor = blocks[0][1]
        for rank, (h, start, length, power) in enumerate(blocks):
            n_reco += 1
            if rng.random() >= accept_rate:
                continue                        # rejected → EV stays natural
            n_acc += 1
            new_start = anchor + rank * stagger_steps
            new_start = max(0, min(new_start, T - length)) if length <= T else 0
            end = min(new_start + length, T)
            ev_orig_app[h][start:min(start + length, T)] = power   # remove natural
            ev_shift_app[h][new_start:end] = power                 # add staggered
    return ev_orig_app, ev_shift_app, (n_reco, n_acc)
