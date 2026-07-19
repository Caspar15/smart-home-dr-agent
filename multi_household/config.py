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
# Cache schema version — bump whenever cleaning / EV-injection logic changes so
# stale parquet caches are never silently reused (v2: min_count class totals,
# i.e. all-NaN appliance rows stay NaN instead of fake 0).
CACHE_VERSION = "v2"
# 15-house non-solar cohort. House 3 is EXCLUDED: the REFIT paper (Murray et
# al., Sci. Data 2017) states Houses 3, 11 & 21 aggregates are affected by
# solar PV (the non-directional clamp shows generation as additional positive
# consumption — a daytime bell-shaped artifact). Houses 1/6/7 also had PV but
# were re-wired by the REFIT team, so their aggregates are unaffected.
# The with-H3 cohort is kept only as an appendix sensitivity run.
CLEAN_HOUSES  = [1, 2, 4, 5, 6, 7, 8, 9, 10,
                 13, 15, 16, 17, 18, 20]                # 15 non-solar houses

# --- data cleaning -----------------------------------------------------
# REFIT only deglitches the per-appliance (IAM) channels (capped at 4000 W);
# the Aggregate stream is NOT cleaned and contains meter glitches (a single
# House-18 reading of 24.9 kW defined the old "peak"). No UK home draws this.
# Legitimate aggregate (incl. 7 kW EV) stays under ~14 kW, so cap at 15 kW.
AGG_DEGLITCH_W = 15000.0
# REFIT has multi-week outages (notably Feb 2014). A window scan over 2014
# found 2014-04-30 .. 2014-07-14 (75 days) where ALL houses have ≥98.8%
# real coverage with a max single gap of only 5.7 h. Every house is reindexed
# onto this common 10-min grid (see preprocess) so the aggregate sums the SAME
# timestamps across houses — not misaligned positions.
# NOON-ANCHORED: the window (and the fixed test split below) end at 12:00 so
# the final night's EV blocks always have their full overnight trough inside
# the horizon — a midnight-ending window clamps them onto the tail (finite-
# window artifact). Same convention as the seasonal windows.
CLEAN_WINDOW = ("2014-04-30", "2014-07-14 12:00")
# --- seasonal-window overrides (experiments/season_windows.py) ----------
# MH_CLEAN_WINDOW="YYYY-MM-DD,YYYY-MM-DD" swaps the analysis window;
# MH_SPLIT_AT="YYYY-MM-DD[ HH:MM]" overrides the train/test split timestamp.
# The headline DEFAULT is now itself a fixed-timestamp split (noon-to-noon,
# exactly 14.0 days of test = 2016 steps) — the same methodology as the
# seasonal windows, so no silent CLI truncation and no 13.6-day mismatch.
import os as _os
if _os.environ.get("MH_CLEAN_WINDOW"):
    CLEAN_WINDOW = tuple(_os.environ["MH_CLEAN_WINDOW"].split(","))
SPLIT_AT = _os.environ.get("MH_SPLIT_AT") or "2014-06-30 12:00"
MAX_INTERP_GAP_STEPS = 36    # causal forward-fill ONLY, gaps ≤6 h (36 steps)
# House 3, 11, 21 have solar PV interfering with the aggregate (non-directional
#   clamp → generation appears as ADDITIONAL positive consumption, daytime
#   bell-shaped artifact; per the REFIT Sci. Data paper). All three excluded.
# House 12 has no deferable appliances
# House 14 is skipped in REFIT itself
# House 19 has only 1 deferable (washing machine) — almost zero DR contribution
# NOTE: excluding H12/H19 (low flexibility) is a DR-potential selection choice;
# the paper reports an all-non-solar-households sensitivity for this.

# --- DR mechanism -----------------------------------------------------
PEAK_HOURS_LOCAL = (17, 22)                             # UK evening peak
OFFPEAK_HOURS    = (0, 6)                               # cheap overnight
PEAK_PRICE_GBP   = 0.30                                 # £/kWh, peak
MID_PRICE_GBP    = 0.15
OFFPEAK_PRICE_GBP = 0.08

# Aggregator broadcasts an extra "Peak Clipping" surcharge when the aggregate
# nowcast exceeds GRID_THRESHOLD_W. Multiplier scales linearly with overage.
# THRESHOLD PROTOCOL (training-derived congestion trigger, NOT a physical
# feeder capacity): calibrated ONCE as the p85 of the 15-house aggregate
# demand over the reference TRAINING window (2014-04-30 .. 2014-06-30 12:00)
# under the primary semi-synthetic EV scenario, then FROZEN across all test
# modes, seeds, ablations, seasonal windows and EV factorial scenarios.
# The legacy 18 kW (test-calibrated) value is retained only as a sensitivity
# point. Value derived & guarded by experiments/derive_threshold.py
# (train p85 = 17712.6 W on the v2 pipeline; reported as 17.7 kW).
GRID_THRESHOLD_W = 17712.6
PEAK_CLIP_ALPHA  = 2.0                                  # 1 + α·(Y/G − 1)
