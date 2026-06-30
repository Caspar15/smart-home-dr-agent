"""Post-hoc smoothing experiment.

Take the (already saved) 10-min retrain-per-DR predictions and re-evaluate
under three different post-hoc smoothing schemes, to quantify how much of
the paper's R² = 0.94 could come from this kind of trick:

  raw            : evaluate as-is (current pipeline result)
  ma3            : 3-step (30-min) centred moving average applied to BOTH
                   ground truth and predicted, then R² recomputed
  ma6            : 6-step (1-h) moving average
  ema_alpha_0.3  : exponential moving average with alpha = 0.3 applied to
                   the predicted series only (band-aid trick)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from src.config import (CACHE, RESULTS, DR_STRATEGIES, ALL_MODELS, TARGET,
                        RFR as RFR_CFG, SVR as SVR_CFG, KNN as KNN_CFG,
                        LSTM as LSTM_CFG, SEED, RESULT_SUFFIX)
from src.data.preprocess import build_dataset
from src.dr.strategies import apply_dr
from src.evaluation.retrain import _train_classical, _train_lstm_on_modified_target


def _ma(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average, padded at edges."""
    if w <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=w, center=True, min_periods=1).mean().to_numpy()


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return x.copy()
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def smooth_pair(y_true, y_pred, scheme: str):
    if scheme == "raw":
        return y_true, y_pred
    if scheme == "ma3":
        return _ma(y_true, 3), _ma(y_pred, 3)
    if scheme == "ma6":
        return _ma(y_true, 6), _ma(y_pred, 6)
    if scheme == "ema_alpha_0.3":
        return y_true, _ema(y_pred, 0.3)
    raise ValueError(scheme)


def main(fast: bool = True):
    split = build_dataset(verbose=False)
    df = split["df_clean"]
    full_y = df[TARGET].to_numpy()
    full_ts = df["date"].to_numpy()
    rng = np.random.default_rng(SEED)

    raw_preds = np.load(CACHE / f"preds_classical{RESULT_SUFFIX}.npz",
                         allow_pickle=False)

    feature_cols = split["feature_names"]
    scaler_X = MinMaxScaler()
    scaler_X.fit(df.iloc[split["train_idx"]][feature_cols].to_numpy())
    X_full = scaler_X.transform(df[feature_cols].to_numpy())
    X_tr = X_full[split["train_idx"]]
    X_te = X_full[split["test_idx"]]

    schemes = ["raw", "ma3", "ma6", "ema_alpha_0.3"]
    rows = []

    for strat in DR_STRATEGIES:
        t0 = time.time()
        y_dr_full = apply_dr(strat, full_y, full_ts)
        y_dr_tr = y_dr_full[split["train_idx"]]
        y_dr_te = y_dr_full[split["test_idx"]]
        sy = MinMaxScaler(); sy.fit(y_dr_tr.reshape(-1, 1))
        y_dr_tr_s = sy.transform(y_dr_tr.reshape(-1, 1)).ravel()

        per_model_preds = {}
        for name in ["linear_regression", "random_forest", "svr", "knn"]:
            yhat_s = _train_classical(name, X_tr, y_dr_tr_s, X_te,
                                      fast=fast, rng=rng)
            per_model_preds[name] = sy.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
        for name in ["persistence", "seasonal_naive", "ets"]:
            per_model_preds[name] = raw_preds[name]

        y_true_lstm, yhat_lstm, _ = _train_lstm_on_modified_target(
            split, y_dr_full, verbose=False)
        per_model_preds["lstm"] = yhat_lstm
        # LSTM uses fewer test rows -- realign by trimming.

        for model_name, yhat in per_model_preds.items():
            if model_name == "lstm":
                y_true_eval = y_true_lstm
            else:
                y_true_eval = y_dr_te
            for scheme in schemes:
                yt_s, yh_s = smooth_pair(y_true_eval, yhat, scheme)
                rows.append(dict(
                    model=model_name, DR=strat, scheme=scheme,
                    MAE=mean_absolute_error(yt_s, yh_s),
                    RMSE=float(np.sqrt(mean_squared_error(yt_s, yh_s))),
                    R2=max(-1.0, r2_score(yt_s, yh_s)),
                ))
        print(f"[{strat:18s}] done in {time.time()-t0:.1f}s")

    df_metrics = pd.DataFrame(rows).round(4)
    df_metrics.to_csv(RESULTS / "metrics_smoothed.csv", index=False)

    pivot = (df_metrics[df_metrics["model"] == "lstm"]
             .pivot(index="DR", columns="scheme", values="R2").round(3))
    print("\n=== LSTM R² under different post-hoc smoothing ===")
    print(pivot.to_string())

    pivot_rmse = (df_metrics[df_metrics["model"] == "lstm"]
                  .pivot(index="DR", columns="scheme", values="RMSE").round(2))
    print("\n=== LSTM RMSE under different post-hoc smoothing ===")
    print(pivot_rmse.to_string())


if __name__ == "__main__":
    main()
