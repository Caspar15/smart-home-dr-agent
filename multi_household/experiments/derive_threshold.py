"""Derive (and verify) the frozen grid-congestion threshold.

PROTOCOL: the trigger is the p85 of the cohort aggregate DEMAND over the
reference TRAINING window (train side of the fixed split) under the primary
semi-synthetic EV scenario. It is calibrated ONCE, frozen in config.py
(GRID_THRESHOLD_W), and reused unchanged across all test modes, seeds,
ablations, seasonal windows and EV factorial scenarios. It is a
training-derived congestion trigger, NOT a physical feeder capacity.

The legacy 18 kW value (calibrated on the TEST window — an evaluation-leakage
provenance) is kept only as a sensitivity point.

Run:  python -m multi_household.experiments.derive_threshold
Exits non-zero if the frozen config value differs from the derived value by
more than TOLERANCE_W (guards against silent drift after pipeline changes).
"""
from __future__ import annotations
import sys
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np

from multi_household.config import CLEAN_HOUSES, GRID_THRESHOLD_W, SPLIT_AT, CLEAN_WINDOW
from multi_household.data.preprocess import prepare_house

TOLERANCE_W = 1.0


def derive() -> float:
    agg = None
    for h in CLEAN_HOUSES:
        a = prepare_house(h)["train_df"]["aggregate_w"].values
        agg = a if agg is None else agg + a
    return float(np.percentile(agg, 85))


def main() -> int:
    p85 = derive()
    print(f"cohort           : {CLEAN_HOUSES}")
    print(f"window / split   : {CLEAN_WINDOW} / {SPLIT_AT}")
    print(f"train p85        : {p85:.1f} W  ({p85/1000:.3f} kW)")
    print(f"frozen in config : {GRID_THRESHOLD_W:.1f} W")
    if abs(p85 - GRID_THRESHOLD_W) > TOLERANCE_W:
        print(f"✗ MISMATCH > {TOLERANCE_W} W — update GRID_THRESHOLD_W to the "
              f"derived value (and note the change in the run manifest).")
        return 1
    print("✓ frozen threshold matches the derived protocol value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
