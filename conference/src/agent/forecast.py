# -*- coding: utf-8 -*-
"""Precompute a 1-step-ahead LSTM forecast for the full chronological series.

The agent's State uses a load forecast (peak-risk signal). To avoid re-running
the LSTM every env step, we train once on the chronological train portion and
emit a prediction for every row, cached to cache/forecast_full.npz.

  forecast_full[t] = predicted Appliances at row t (NaN for the first `lookback`
                     rows where no window exists).

Run:  python -m src.agent.forecast
"""
from __future__ import annotations
import numpy as np

from src.config import CACHE, TARGET
from src.data.preprocess import build_dataset
from src.evaluation.retrain import _train_lstm_on_modified_target


def precompute(verbose=True):
    # chronological split so the forecast over the test period is honest
    split = build_dataset(verbose=False, split_mode="chronological", feature_set="env_lags")
    df = split["df_clean"]
    full_y = df[TARGET].to_numpy()
    n = len(df)

    # Train on the chronological target; the trainer returns predictions on the
    # TEST positions plus their end indices. We also want train-period values
    # for state continuity, so we additionally fill them with a cheap proxy.
    y_true, y_pred, end_idx_test = _train_lstm_on_modified_target(
        split, full_y, verbose=verbose)

    forecast = np.full(n, np.nan, dtype=float)
    forecast[end_idx_test] = y_pred
    # For the train period (state continuity), fall back to persistence
    # (previous actual) — agents are only evaluated on the test period anyway.
    persistence = np.r_[full_y[:1], full_y[:-1]]
    mask = np.isnan(forecast)
    forecast[mask] = persistence[mask]

    out = CACHE / "forecast_full.npz"
    np.savez_compressed(out, forecast=forecast,
                        test_start=int(end_idx_test.min()),
                        test_end=int(end_idx_test.max()))
    if verbose:
        from sklearn.metrics import r2_score
        print(f"[forecast] test R2={r2_score(y_true, y_pred):.3f}  saved -> {out}")
    return forecast


if __name__ == "__main__":
    precompute()
