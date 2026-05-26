"""LR / RFR / SVR / k-NN + Persistence / Seasonal Naïve / ETS baselines.

All models predict the (unscaled) target on the chronological test split.
Predictions are saved as numpy arrays keyed by model name so they can be
combined with DR strategies later.

Paper reference (Table 2 stats verified -> mean=97.69, std=102.52, max=1080).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import (CACHE, RESULTS, TARGET, RFR as RFR_CFG, SVR as SVR_CFG,
                        KNN as KNN_CFG, SEED, RESAMPLE_FREQ, RESULT_SUFFIX)
from src.data.preprocess import build_dataset

# Steps per day depends on sampling rate.
SEASON = 24 if RESAMPLE_FREQ == "1h" else 144


def _inverse_y(scaler_y, y_scaled: np.ndarray) -> np.ndarray:
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def fit_predict_sklearn(name: str, model, split) -> dict:
    t0 = time.time()
    model.fit(split["X_train"], split["y_train"])
    yhat_scaled = model.predict(split["X_test"])
    yhat = _inverse_y(split["scaler_y"], yhat_scaled)
    return dict(name=name, y_pred=yhat, fit_time=time.time() - t0)


def persistence_predict(split) -> dict:
    """y_hat[t] = y[t-1] using the FULL chronological series, then look up
    predictions at test positions only."""
    full_y = split["df_clean"][TARGET].to_numpy()
    full_pred = np.r_[full_y[:1], full_y[:-1]]   # shift by 1
    preds = full_pred[split["test_idx"]]
    return dict(name="persistence", y_pred=preds.astype(float), fit_time=0.0)


def seasonal_naive_predict(split, season: int = SEASON) -> dict:
    """y_hat[t] = y[t - season] over the full chronological series."""
    full_y = split["df_clean"][TARGET].to_numpy()
    full_pred = np.empty_like(full_y, dtype=float)
    full_pred[:season] = full_y[:season]
    full_pred[season:] = full_y[:-season]
    preds = full_pred[split["test_idx"]]
    return dict(name="seasonal_naive", y_pred=preds, fit_time=0.0)


def ets_predict(split, season: int = SEASON) -> dict:
    """Holt-Winters additive exponential smoothing on the chronological
    series. For random-split we still fit on a contiguous chronological
    window (last fit_len train indices' values), then produce a forecast
    that we evaluate at test positions. To avoid the cost of forecasting
    horizon = N_test * gap, we fit once and emit a fitted+forecast vector
    aligned with the full series, then index by test_idx."""
    full_y = split["df_clean"][TARGET].to_numpy()
    n = len(full_y)
    fit_len = min(n, 14 * season)   # 2 weeks
    series = pd.Series(full_y[:fit_len])
    t0 = time.time()
    try:
        model = ExponentialSmoothing(
            series, trend=None, seasonal="add", seasonal_periods=season,
            initialization_method="estimated"
        ).fit(optimized=True, use_brute=False)
        # In-sample fitted values for the fit window, then forecast beyond.
        fitted = np.asarray(model.fittedvalues, dtype=float)
        remaining = n - fit_len
        if remaining > 0:
            fc = np.asarray(model.forecast(steps=remaining), dtype=float)
            full_pred = np.concatenate([fitted, fc])
        else:
            full_pred = fitted[:n]
    except Exception as e:
        print(f"[ets] fallback to seasonal naive: {e}")
        return seasonal_naive_predict(split, season=season) | {"name": "ets"}
    preds = full_pred[split["test_idx"]]
    return dict(name="ets", y_pred=preds, fit_time=time.time() - t0)


def run_all(split, fast: bool = False) -> dict:
    results = {}

    # --- Linear Regression ---
    results["linear_regression"] = fit_predict_sklearn(
        "linear_regression", LinearRegression(), split)
    print(f"[lr]  done  ({results['linear_regression']['fit_time']:.1f}s)")

    # --- Random Forest ---
    rfr_cfg = dict(RFR_CFG)
    if fast:
        rfr_cfg["n_estimators"] = 100
    results["random_forest"] = fit_predict_sklearn(
        "random_forest", RandomForestRegressor(**rfr_cfg), split)
    print(f"[rfr] done  ({results['random_forest']['fit_time']:.1f}s)")

    # --- SVR ---  (slow: subsample train for tractability)
    svr_cfg = dict(SVR_CFG)
    rng = np.random.default_rng(SEED)
    max_n = 4000 if fast else 8000
    idx = rng.choice(len(split["X_train"]), size=min(max_n, len(split["X_train"])),
                     replace=False)
    svr_split = dict(split)
    svr_split["X_train"] = split["X_train"][idx]
    svr_split["y_train"] = split["y_train"][idx]
    results["svr"] = fit_predict_sklearn("svr", SVR(**svr_cfg), svr_split)
    print(f"[svr] done  ({results['svr']['fit_time']:.1f}s)")

    # --- k-NN ---
    results["knn"] = fit_predict_sklearn(
        "knn", KNeighborsRegressor(**KNN_CFG), split)
    print(f"[knn] done  ({results['knn']['fit_time']:.1f}s)")

    # --- Baselines ---
    results["persistence"] = persistence_predict(split)
    print("[pst] done")
    results["seasonal_naive"] = seasonal_naive_predict(split)
    print("[sn ] done")
    results["ets"] = ets_predict(split)
    print(f"[ets] done  ({results['ets']['fit_time']:.1f}s)")
    return results


def main(fast: bool = False):
    split = build_dataset()
    results = run_all(split, fast=fast)

    y_test = split["y_test_raw"]
    np.savez_compressed(
        CACHE / f"preds_classical{RESULT_SUFFIX}.npz",
        y_test=y_test,
        **{name: r["y_pred"] for name, r in results.items()},
    )

    # Quick sanity metrics (no DR yet).
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rows = []
    for name, r in results.items():
        yhat = r["y_pred"]
        rows.append(dict(
            model=name,
            MAE=mean_absolute_error(y_test, yhat),
            RMSE=np.sqrt(mean_squared_error(y_test, yhat)),
            R2=r2_score(y_test, yhat),
        ))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS / "sanity_classical_no_dr.csv", index=False)
    print("\n[classical] no-DR sanity metrics (test split):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller RFR/SVR for quick test")
    args = ap.parse_args()
    main(fast=args.fast)
