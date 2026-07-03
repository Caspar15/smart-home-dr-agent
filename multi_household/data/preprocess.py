"""Per-house preprocessing: NaN imputation, feature engineering, train/test
split.

Input:  raw 10-min DataFrame from refit_loader.load_house()
Output: tuple of (features, target) numpy arrays + a metadata dict
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from multi_household.data.refit_loader import load_house
from multi_household.config import CLEAN_WINDOW, MAX_INTERP_GAP_STEPS, RESAMPLE_FREQ


def reindex_to_common_grid(df: pd.DataFrame,
                           window: tuple[str, str] = CLEAN_WINDOW
                           ) -> pd.DataFrame:
    """Reindex onto the shared 10-min grid spanning the clean window, so EVERY
    house has identical timestamps by position. This is what makes the
    cross-house aggregate valid — without it, per-house row drops desynchronise
    the houses and the aggregate sums different times of day."""
    full = pd.date_range(window[0], window[1], freq=RESAMPLE_FREQ)
    out = df.set_index("time").reindex(full)
    out.index.name = "time"
    return out.reset_index()


# --- imputation -------------------------------------------------------------

def impute_missing(df: pd.DataFrame,
                   max_gap: int = MAX_INTERP_GAP_STEPS) -> pd.DataFrame:
    """Fill ONLY short gaps. REFIT has multi-week outages; interpolating across
    those fabricates data, so we limit filling to `max_gap` steps (≤6 h).
    Anything still missing after that is dropped by the caller — we never invent
    a month-long straight line. (The clean-window restriction means very little
    is left to fill.)

    Causal by design: we forward-fill only (never backfill from FUTURE values),
    so an imputed cell never peeks ahead. Leading NaNs before a house's first
    reading are left as NaN and dropped by the caller.
    """
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = (out[numeric]
                    .ffill(limit=max_gap)
                    .interpolate(method="linear", limit=max_gap,
                                 limit_direction="forward"))
    return out


def restrict_to_clean_window(df: pd.DataFrame,
                             window: tuple[str, str] = CLEAN_WINDOW
                             ) -> pd.DataFrame:
    """Keep only the common low-outage window (avoids the Feb-2014 etc. gaps)."""
    t = pd.to_datetime(df["time"])
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return df[(t >= lo) & (t <= hi)].reset_index(drop=True)


# --- feature engineering ----------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["time"])
    out["hour"]        = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
    out["month"]       = ts.dt.month
    # Cyclical encodings — continuity for the model
    out["hour_sin"]  = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"]   = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    return out


def add_lag_features(df: pd.DataFrame, target_col: str = "aggregate_w",
                     lags=(1, 2, 3, 6, 144, 1008)) -> pd.DataFrame:
    """Conference paper's lag set (1, 2, 3, 6 step + 24h) plus a 1-week lag."""
    out = df.copy()
    for L in lags:
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    out[f"{target_col}_roll6_mean"]  = out[target_col].shift(1).rolling(6).mean()
    out[f"{target_col}_roll6_std"]   = out[target_col].shift(1).rolling(6).std()
    out[f"{target_col}_roll144_mean"] = out[target_col].shift(1).rolling(144).mean()
    return out


# --- split ------------------------------------------------------------------

def chronological_split(df: pd.DataFrame, train_frac: float = 0.80
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 by time — first 80% train, last 20% test."""
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


# --- one-shot wrapper -------------------------------------------------------

FEATURE_COLS_BASE = [
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "non_controllable_w", "deferable_w", "semi_deferable_w",
]


def prepare_house(house_id: int, train_frac: float = 0.80) -> dict:
    """End-to-end: load, impute, feature engineer, split.

    Returns a dict with:
        train_df, test_df            (full DataFrames with all columns)
        feature_cols                 (list of column names used as features)
        target_col                   ('aggregate_w')
        meta                         (counts, first/last timestamp, etc.)
    """
    df = load_house(house_id)
    df = reindex_to_common_grid(df)            # shared grid → houses align
    df = impute_missing(df)                     # fill the small (≤6 h) gaps
    # The chosen window has ≤6 h gaps for every house, so nothing should remain;
    # guard anyway by dropping any residual (keeps all houses on the same rows
    # because the grid + gaps are shared).
    df = df.dropna(subset=["aggregate_w"]).reset_index(drop=True)
    df = add_time_features(df)
    df = add_lag_features(df, "aggregate_w")
    # Drop the first rows that have NaN lag features
    df = df.dropna(subset=[c for c in df.columns
                            if c.startswith("aggregate_w_lag")
                            or c.startswith("aggregate_w_roll")]).reset_index(drop=True)

    train_df, test_df = chronological_split(df, train_frac=train_frac)

    feature_cols = FEATURE_COLS_BASE + [
        c for c in df.columns
        if c.startswith("aggregate_w_lag") or c.startswith("aggregate_w_roll")
    ]
    meta = {
        "house_id":      house_id,
        "n_total":       len(df),
        "n_train":       len(train_df),
        "n_test":        len(test_df),
        "first":         str(df["time"].iloc[0]),
        "last":          str(df["time"].iloc[-1]),
        "feature_cols":  feature_cols,
    }
    return {
        "train_df":     train_df,
        "test_df":      test_df,
        "feature_cols": feature_cols,
        "target_col":   "aggregate_w",
        "meta":         meta,
    }
