"""Paths and shared hyperparams for the multi-household pipeline."""
from __future__ import annotations
from pathlib import Path

# --- paths ----------------------------------------------------------
# __file__ = reproduction/multi_household/config.py
REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]      # reproduction/
PROJECT_ROOT      = REPRODUCTION_ROOT.parent                 # "AI Agent smart grid"/
REFIT_DIR         = PROJECT_ROOT / "archive"                 # raw REFIT CSVs
CACHE_DIR         = REPRODUCTION_ROOT / "multi_household" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- data --------------------------------------------------------------
RESAMPLE_FREQ = "10min"                                 # 10-min resolution
CLEAN_HOUSES  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 13, 15, 16, 17, 18, 20]                # 16 clean houses

# --- data cleaning -----------------------------------------------------
# REFIT only deglitches the per-appliance (IAM) channels (capped at 4000 W);
# the Aggregate stream is NOT cleaned and contains meter glitches (a single
# House-18 reading of 24.9 kW defined the old "peak"). No UK home draws this.
# Legitimate aggregate (incl. 7 kW EV) stays under ~14 kW, so cap at 15 kW.
AGG_DEGLITCH_W = 15000.0
# REFIT has multi-week outages (notably Feb 2014). A window scan over 2014
# found 2014-04-30 .. 2014-07-14 (75 days) where ALL 16 houses have ≥98.8%
# real coverage with a max single gap of only 5.7 h. Every house is reindexed
# onto this common 10-min grid (see preprocess) so the aggregate sums the SAME
# timestamps across houses — not misaligned positions.
CLEAN_WINDOW = ("2014-04-30", "2014-07-14")
MAX_INTERP_GAP_STEPS = 36                               # fill only ≤6 h gaps
# House 11, 21 have solar PV (aggregate is net load, not raw demand)
# House 12 has no deferable appliances
# House 14 is skipped in REFIT itself
# House 19 has only 1 deferable (washing machine) — almost zero DR contribution

# --- DR mechanism -----------------------------------------------------
PEAK_HOURS_LOCAL = (17, 22)                             # UK evening peak
OFFPEAK_HOURS    = (0, 6)                               # cheap overnight
PEAK_PRICE_GBP   = 0.30                                 # £/kWh, peak
MID_PRICE_GBP    = 0.15
OFFPEAK_PRICE_GBP = 0.08

# Aggregator broadcasts an extra "Peak Clipping" surcharge when aggregate
# forecast Y(t+1) exceeds GRID_THRESHOLD_W. Multiplier scales linearly
# with the overage ratio.
# Threshold ≈ the p85 of the 16-house aggregate on the test window, so
# coordinated mode triggers on roughly the busiest ~15% of timesteps —
# frequent enough to materially shave peaks, selective enough that the rebound
# stays small. (For reference, the current clean-window aggregate runs ~12.5 kW
# mean with a No-DR peak of ~40.5 kW, driven by the overnight EV pile-up.)
GRID_THRESHOLD_W = 18000.0
PEAK_CLIP_ALPHA  = 2.0                                  # 1 + α·(Y/G − 1)
