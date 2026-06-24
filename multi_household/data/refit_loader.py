"""Load REFIT house CSVs at 10-min resolution, split into deferable /
semi-deferable / non-controllable channels, and cache.

Raw schema (per CSV row, ~6-8s):
    Time, Unix, Aggregate, Appliance1, ..., Appliance9     (all watts)

Loaded DataFrame (per row, 10 min):
    time                  pd.Timestamp     (10-min bucket start)
    aggregate_w           float            mean watts over the bucket
    aggregate_wh          float            aggregate_w * 10/60 → energy in Wh
    deferable_w           float            sum of deferable appliances
    semi_deferable_w      float            sum of semi-deferable appliances
    non_controllable_w    float            sum of non-controllable appliances
    appliance_<name>_w    float            per-appliance (one column each)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from multi_household.config import (
    REFIT_DIR, CACHE_DIR, RESAMPLE_FREQ, AGG_DEGLITCH_W,
)
from multi_household.data.appliance_map import (
    HOUSE_APPLIANCES, classify, appliance_summary,
)

# Households we equip with a synthetic EV charger.
# Chosen for higher base consumption (mid-size families likely to own EVs).
EV_HOUSES = {5, 7, 9, 13, 18}
EV_CHARGE_W   = 7000.0     # 7 kW typical UK home charger
EV_CHARGE_DURATION_STEPS = 24   # 4 hours @ 10-min step
EV_CHARGE_HOUR_RANGE = (22, 4)  # plug-in 22:00, finish before 04:00


def _inject_synthetic_ev(df: pd.DataFrame, house_id: int,
                          seed: int = None) -> pd.DataFrame:
    """Add a synthetic EV charger column for selected houses.

    Behaviour: every weekday evening between EV_CHARGE_HOUR_RANGE the EV
    charges at EV_CHARGE_W for EV_CHARGE_DURATION_STEPS steps. Weekend
    starts are skipped randomly (~30% of nights). The column is named
    `appliance_synthetic_ev_w` and is added to BOTH the raw aggregate
    (so demand reflects it) and as a separate channel (so agent can
    target it for deferral).
    """
    if house_id not in EV_HOUSES:
        return df
    rng = np.random.default_rng(seed if seed is not None else house_id * 17)
    out = df.copy()
    ts = pd.to_datetime(out["time"])
    n = len(out)
    ev_w = np.zeros(n, dtype=np.float32)

    # Walk day by day
    day_starts = []
    last_date = None
    for i, t in enumerate(ts):
        d = t.date()
        if d != last_date:
            day_starts.append((i, t))
            last_date = d

    for day_i, (i_start, t_start) in enumerate(day_starts):
        # 70% chance of charging that night
        if rng.random() > 0.70:
            continue
        # find index of 22:00 (or first available step on that day)
        # 22:00 is hour 22, but plug-in might happen between 21:00 and 23:30
        plug_in_hour   = 21 + int(rng.random() * 3)   # 21, 22, or 23
        plug_in_minute = int(rng.random() * 6) * 10   # 0, 10, ..., 50
        plug_in_ts = pd.Timestamp(t_start.date()) + pd.Timedelta(
            hours=plug_in_hour, minutes=plug_in_minute)
        # find nearest index >= plug_in_ts
        idx = ts.searchsorted(plug_in_ts)
        end = min(idx + EV_CHARGE_DURATION_STEPS, n)
        ev_w[idx:end] = EV_CHARGE_W
    out["appliance_synthetic_ev_w"] = ev_w
    out["aggregate_w"] = out["aggregate_w"] + ev_w
    out["aggregate_wh"] = out["aggregate_w"] * (10.0 / 60.0)
    return out


def _safe_name(s: str) -> str:
    """Slugify an appliance name for use as a column name."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _cache_path(house_id: int) -> Path:
    return CACHE_DIR / f"refit_house_{house_id:02d}_{RESAMPLE_FREQ}.parquet"


def load_house(house_id: int, force_rebuild: bool = False) -> pd.DataFrame:
    """Load one REFIT house at 10-min resolution.

    On first call, reads the raw CSV (~7M rows), downsamples to 10-min mean,
    and caches as parquet. Subsequent calls hit the cache (< 1s).
    """
    cache = _cache_path(house_id)
    if cache.exists() and not force_rebuild:
        return pd.read_parquet(cache)

    csv_path = REFIT_DIR / f"House_{house_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Read raw — use int32 to halve memory (each col is whole-watt int)
    dtypes = {f"Appliance{i}": "int32" for i in range(1, 10)}
    dtypes["Aggregate"] = "int32"
    dtypes["Unix"]      = "int64"
    df = pd.read_csv(csv_path, parse_dates=["Time"], dtype=dtypes)
    df = df.set_index("Time").drop(columns=["Unix"])

    # Downsample to 10-min mean watts
    df10 = df.resample(RESAMPLE_FREQ).mean()

    # Rename per-house appliance channels using the map, build class buckets
    apl = HOUSE_APPLIANCES[house_id]
    appliance_cols = {}
    deferable_cols, semi_cols, non_cols = [], [], []
    for ch in range(1, 10):
        name = apl.get(ch, f"Channel {ch}")
        new = f"appliance_{_safe_name(name)}_w"
        # if the safe-name collides (e.g. House 4 two washing machines), add ch
        if new in appliance_cols.values():
            new = f"appliance_{_safe_name(name)}_ch{ch}_w"
        appliance_cols[f"Appliance{ch}"] = new
        cls = classify(name)
        if   cls == "deferable":        deferable_cols.append(new)
        elif cls == "semi_deferable":   semi_cols.append(new)
        else:                            non_cols.append(new)
    df10 = df10.rename(columns=appliance_cols)
    df10 = df10.rename(columns={"Aggregate": "aggregate_w"})

    # Class-level totals
    df10["deferable_w"]        = df10[deferable_cols].sum(axis=1) if deferable_cols else 0.0
    df10["semi_deferable_w"]   = df10[semi_cols].sum(axis=1)      if semi_cols else 0.0
    df10["non_controllable_w"] = df10[non_cols].sum(axis=1)       if non_cols else 0.0

    # Energy column (W → Wh per 10-min bucket)
    df10["aggregate_wh"] = df10["aggregate_w"] * (10.0 / 60.0)

    # Bring index back as 'time' column so it round-trips through parquet cleanly
    df10 = df10.reset_index().rename(columns={"Time": "time"})

    # Synthetic EV injection (BEFORE caching so it persists)
    df10 = _inject_synthetic_ev(df10, house_id)
    # If EV was added, re-bucket it into deferable_w and recompute class totals
    if "appliance_synthetic_ev_w" in df10.columns:
        df10["deferable_w"] = df10["deferable_w"] + df10["appliance_synthetic_ev_w"]

    # Deglitch the FINAL aggregate (incl. EV): REFIT does not clean the Aggregate
    # stream, so it carries meter spikes well above any physical household draw
    # (a single House-18 reading hit 24.9 kW). Legit load incl. the 7 kW EV stays
    # under ~14 kW, so clip at 15 kW. NaNs are preserved for downstream handling.
    df10["aggregate_w"] = df10["aggregate_w"].clip(upper=AGG_DEGLITCH_W)
    df10["aggregate_wh"] = df10["aggregate_w"] * (10.0 / 60.0)

    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    df10.to_parquet(cache, index=False)
    return df10


def load_houses(house_ids: Iterable[int],
                force_rebuild: bool = False) -> dict[int, pd.DataFrame]:
    """Batch-load. Returns dict {house_id: DataFrame}."""
    return {h: load_house(h, force_rebuild) for h in house_ids}


def coverage_report(df: pd.DataFrame) -> dict:
    """Quick sanity stats for one house's 10-min DataFrame."""
    n = len(df)
    nan_rate = float(df["aggregate_w"].isna().mean())
    zero_rate = float((df["aggregate_w"] == 0).mean())
    return {
        "n_rows": n,
        "days":   round((df["time"].iloc[-1] - df["time"].iloc[0]).days, 1),
        "aggregate_mean_w": float(df["aggregate_w"].mean()),
        "aggregate_p95_w": float(df["aggregate_w"].quantile(0.95)),
        "deferable_mean_w": float(df["deferable_w"].mean()),
        "non_controllable_mean_w": float(df["non_controllable_w"].mean()),
        "nan_rate":   round(nan_rate, 4),
        "zero_rate":  round(zero_rate, 4),
    }
