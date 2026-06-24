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


def stagger_ev_schedule(ev_orig_by_house: dict[int, np.ndarray],
                        timestamps: pd.DatetimeIndex,
                        stagger_steps: int = STAGGER_STEPS) -> dict[int, np.ndarray]:
    """Reschedule each night's EV blocks across houses so they fan out.

    Args:
        ev_orig_by_house: {house_id: EV watts array (T,)}
        timestamps:       (T,) DatetimeIndex aligned with the arrays
    Returns:
        {house_id: rescheduled EV watts array (T,)} — same energy, spread out.
    """
    houses = sorted(ev_orig_by_house)
    T = len(timestamps)
    ev_shift = {h: np.zeros(T, dtype=np.float32) for h in houses}

    # Group every EV block by the calendar date of its (evening) start. The
    # synthetic injection always plugs in at 21:00-23:00, so the start date is
    # an unambiguous "night" key.
    night_blocks: dict[object, list] = defaultdict(list)
    for h in houses:
        for (start, length) in _detect_blocks(ev_orig_by_house[h]):
            power = float(ev_orig_by_house[h][start])
            night = timestamps[start].date()
            night_blocks[night].append((h, start, length, power))

    for night, blocks in night_blocks.items():
        blocks.sort(key=lambda b: b[1])         # by original start step
        anchor = blocks[0][1]                   # earliest plug-in that night
        for rank, (h, start, length, power) in enumerate(blocks):
            new_start = anchor + rank * stagger_steps
            # keep the block inside the array; never resize it
            new_start = max(0, min(new_start, T - length)) if length <= T else 0
            end = min(new_start + length, T)
            ev_shift[h][new_start:end] = power
    return ev_shift
