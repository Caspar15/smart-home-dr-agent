"""Data-layer tests: causal imputation, chronological split, lag causality,
synthetic-EV injection, forecaster windowing. All synthetic — no REFIT needed."""
from __future__ import annotations
import numpy as np
import pandas as pd

from multi_household.data.preprocess import (
    impute_missing, chronological_split, add_lag_features,
)
from multi_household.data.refit_loader import _inject_synthetic_ev, EV_CHARGE_W
from multi_household.forecasting.per_house_lstm import make_windows


def _df(n=200, start="2014-05-01"):
    t = pd.date_range(start, periods=n, freq="10min")
    return pd.DataFrame({"time": t,
                         "aggregate_w": np.arange(n, dtype=float) + 100.0})


# --- imputation: causal, bounded ------------------------------------------

def test_impute_fills_short_gap_but_not_leading_nans():
    df = _df(50)
    df.loc[0:2, "aggregate_w"] = np.nan          # leading gap (no past)
    df.loc[20:22, "aggregate_w"] = np.nan        # interior 3-step gap
    out = impute_missing(df, max_gap=6)
    assert out["aggregate_w"].iloc[0:3].isna().all()      # never backfilled
    assert out["aggregate_w"].iloc[20:23].notna().all()   # short gap filled


def test_impute_does_not_bridge_long_gaps():
    df = _df(100)
    df.loc[30:60, "aggregate_w"] = np.nan        # 31-step gap > max_gap
    out = impute_missing(df, max_gap=6)
    # only the first max_gap steps after the gap start may be filled
    assert out["aggregate_w"].iloc[40:60].isna().any()


# --- split / lags: no leakage ----------------------------------------------

def test_chronological_split_keeps_time_order():
    df = _df(100)
    tr, te = chronological_split(df, train_frac=0.8)
    assert len(tr) == 80 and len(te) == 20
    assert tr["time"].iloc[-1] < te["time"].iloc[0]       # strictly earlier


def test_lag_features_are_strictly_past():
    df = add_lag_features(_df(50), "aggregate_w", lags=(1, 2))
    i = 30
    assert df["aggregate_w_lag1"].iloc[i] == df["aggregate_w"].iloc[i - 1]
    assert df["aggregate_w_lag2"].iloc[i] == df["aggregate_w"].iloc[i - 2]
    # rolling mean excludes the CURRENT value (shift(1) before rolling)
    expect = df["aggregate_w"].iloc[i - 6:i].mean()
    assert abs(df["aggregate_w_roll6_mean"].iloc[i] - expect) < 1e-9


# --- synthetic EV injection --------------------------------------------------

def test_ev_injection_deterministic_and_consistent():
    base = pd.DataFrame({
        "time": pd.date_range("2014-05-01", periods=7 * 144, freq="10min"),
        "aggregate_w": np.full(7 * 144, 500.0),
    })
    base["aggregate_wh"] = base["aggregate_w"] * (10.0 / 60.0)
    out1 = _inject_synthetic_ev(base.copy(), house_id=5)
    out2 = _inject_synthetic_ev(base.copy(), house_id=5)
    ev = out1["appliance_synthetic_ev_w"]
    assert (ev == out2["appliance_synthetic_ev_w"]).all()   # deterministic
    assert set(np.unique(ev)) <= {0.0, EV_CHARGE_W}         # 0 or 7 kW only
    assert ev.sum() > 0                                      # some nights charge
    # aggregate reflects the EV exactly
    assert np.allclose(out1["aggregate_w"], base["aggregate_w"] + ev)


def test_ev_injection_skips_non_ev_house():
    base = _df(288)
    out = _inject_synthetic_ev(base.copy(), house_id=2)      # not in EV_HOUSES
    assert "appliance_synthetic_ev_w" not in out.columns
    assert (out["aggregate_w"] == base["aggregate_w"]).all()


# --- forecaster windowing -----------------------------------------------------

def test_make_windows_shape_and_alignment():
    T, F, L = 60, 3, 10
    X = np.arange(T * F, dtype=np.float32).reshape(T, F)
    y = np.arange(T, dtype=np.float32)
    Xw, yw = make_windows(X, y, lookback=L)
    assert Xw.shape == (T - L, L, F) and yw.shape == (T - L,)
    # window i covers rows [i, i+L); the label is the NEXT step (row i+L)
    assert (Xw[0] == X[0:L]).all()
    assert yw[0] == y[L]
