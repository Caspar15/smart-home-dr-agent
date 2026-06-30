# -*- coding: utf-8 -*-
"""Unified experiment runner for the conference paper (E1–E6).

A single core, run_grid(), powers every experiment by varying:
  split_mode      : "random" | "chronological"
  feature_set     : "env" | "env_lags" | "full"
  models          : subset of MODELS
  dr_list         : ["none"] or DR_STRATEGIES
  granularities   : subset of {"raw","ma3","ma6","ma12"}

Drivers:
  e1e2()  reproduction (Table II) + granularity study (Fig. A) + cnn_lstm std (E8)
  e3()    split-leakage study (Fig. B)
  e4()    vs Candanedo original method (Table III)
  e5()    feature ablation (Table V)
  e6()    architecture ablation (Table IV)

CLI:
  python -m src.evaluation.experiments --exp e1e2 [--fast]
  python -m src.evaluation.experiments --exp all  [--fast]
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (RESULTS, TARGET, SEED, DR_STRATEGIES)
from src.data.preprocess import build_dataset
from src.dr.strategies import apply_dr
from src.evaluation.retrain import (_train_classical, _train_lstm_on_modified_target,
                                    _metrics)
from src.forecasting.classical import (persistence_predict, seasonal_naive_predict,
                                       ets_predict)
from src.forecasting.lstm_cnn import train_lstm_v2

CLASSICAL = {"linear_regression", "random_forest", "svr", "knn", "gbm", "xgboost"}
NAIVE = {"persistence": persistence_predict,
         "seasonal_naive": seasonal_naive_predict,
         "ets": ets_predict}
GRAN = {"raw": 1, "ma3": 3, "ma6": 6, "ma12": 12}


def _ma(x, w):
    if w <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _fast_lstm_cfg(fast):
    return {"epochs": 6, "patience": 3} if fast else {}


def run_grid(split_mode, feature_set, models, dr_list, granularities,
             n_seeds_cnn=4, fast=False, tag=""):
    split = build_dataset(verbose=False, split_mode=split_mode, feature_set=feature_set)
    df = split["df_clean"]
    full_y = df[TARGET].to_numpy(); full_ts = df["date"].to_numpy()
    feature_cols = split["feature_names"]
    rng = np.random.default_rng(SEED)

    sx = MinMaxScaler().fit(df.iloc[split["train_idx"]][feature_cols].to_numpy())
    X_full = sx.transform(df[feature_cols].to_numpy())
    X_tr, X_te = X_full[split["train_idx"]], X_full[split["test_idx"]]

    rows = []
    for dr in dr_list:
        y_dr_full = full_y if dr == "none" else apply_dr(dr, full_y, full_ts)
        y_dr_te = y_dr_full[split["test_idx"]]
        sy = MinMaxScaler().fit(y_dr_full[split["train_idx"]].reshape(-1, 1))
        y_dr_tr_s = sy.transform(y_dr_full[split["train_idx"]].reshape(-1, 1)).ravel()

        for model in models:
            t0 = time.time()
            seed_r2 = {g: [] for g in granularities}   # for cnn_lstm std

            if model in CLASSICAL:
                yhat_s = _train_classical(model, X_tr, y_dr_tr_s, X_te, fast=fast, rng=rng)
                yhat = sy.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
                y_true = y_dr_te
            elif model in NAIVE:
                yhat = NAIVE[model](split)["y_pred"]      # raw prediction
                y_true = y_dr_te
            elif model == "lstm":
                y_true, yhat, _ = _train_lstm_on_modified_target(split, y_dr_full, verbose=False)
            elif model == "cnn_lstm":
                preds = []
                for k in range(n_seeds_cnn):
                    out = train_lstm_v2(split, y_full=y_dr_full,
                                        cfg={**_fast_lstm_cfg(fast), "_seed_offset": k},
                                        verbose=False)
                    preds.append(out["y_pred"]); y_true = out["y_true"]
                    for g in granularities:
                        w = GRAN[g]
                        seed_r2[g].append(_metrics(_ma(y_true, w), _ma(out["y_pred"], w))["R2"])
                yhat = np.mean(preds, axis=0)
            else:
                raise ValueError(model)

            for g in granularities:
                w = GRAN[g]
                m = _metrics(_ma(y_true, w), _ma(yhat, w))
                row = dict(experiment=tag, model=model, DR=dr, granularity=g,
                           split=split_mode, features=feature_set,
                           MAE=round(m["MAE"], 3), RMSE=round(m["RMSE"], 3),
                           R2=round(m["R2"], 4))
                if model == "cnn_lstm" and seed_r2[g]:
                    row["R2_std"] = round(float(np.std(seed_r2[g])), 4)
                rows.append(row)
            print(f"  [{tag}|{split_mode}|{feature_set}|{dr}|{model}] {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ----------------------------- drivers ------------------------------------- #

def e1e2(fast=False):
    """Reproduction (raw) + granularity sweep, random split, 7 DR, all models."""
    models = ["linear_regression", "random_forest", "svr", "knn", "gbm", "xgboost",
              "persistence", "seasonal_naive", "ets", "lstm", "cnn_lstm"]
    df = run_grid("random", "env_lags", models, DR_STRATEGIES,
                  ["raw", "ma3", "ma6", "ma12"], n_seeds_cnn=(2 if fast else 4),
                  fast=fast, tag="E1E2")
    df.to_csv(RESULTS / "E1E2_reproduction_granularity.csv", index=False)
    # E1 = raw slice
    df[df.granularity == "raw"].to_csv(RESULTS / "E1_reproduction.csv", index=False)
    df.to_csv(RESULTS / "E2_granularity.csv", index=False)
    print("\nsaved E1_reproduction.csv + E2_granularity.csv")
    return df


def e3(fast=False):
    """Split-leakage study: no-DR, 8 learning models, random vs chronological."""
    models = ["linear_regression", "knn", "svr", "random_forest", "gbm", "xgboost",
              "lstm", "cnn_lstm"]
    out = []
    for sm in ["random", "chronological"]:
        out.append(run_grid(sm, "env_lags", models, ["none"], ["raw", "ma6"],
                            n_seeds_cnn=(1 if fast else 2), fast=fast, tag="E3"))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RESULTS / "E3_split.csv", index=False)
    print("\nsaved E3_split.csv")
    return df


def e4(fast=False):
    """vs Candanedo original method (Table III).

    Match Candanedo's protocol as closely as possible: random CV-style split,
    environmental + time features, NO autoregressive lags, no DR. Under this
    protocol GBM should reproduce ~0.5–0.57. We then contrast with our honest
    chronological-split numbers (reported in E3) and the lag-augmented setup.
    """
    models = ["linear_regression", "svr", "random_forest", "gbm", "xgboost",
              "lstm", "cnn_lstm"]
    # Candanedo protocol: random split, env-only features
    a = run_grid("random", "env", models, ["none"], ["raw"],
                 n_seeds_cnn=(1 if fast else 3), fast=fast, tag="E4_candanedo_protocol")
    # Honest contrast: chronological split, env-only
    b = run_grid("chronological", "env", models, ["none"], ["raw"],
                 n_seeds_cnn=(1 if fast else 3), fast=fast, tag="E4_honest_chrono")
    df = pd.concat([a, b], ignore_index=True)
    df.to_csv(RESULTS / "E4_vs_candanedo.csv", index=False)
    print("\nsaved E4_vs_candanedo.csv")
    return df


def e5(fast=False):
    """Feature ablation: env / env_lags / full, no-DR, both splits, key models."""
    models = ["linear_regression", "random_forest", "gbm", "lstm"]
    out = []
    for fs in ["env", "env_lags", "full"]:
        for sm in ["random", "chronological"]:
            out.append(run_grid(sm, fs, models, ["none"], ["raw"],
                                fast=fast, tag="E5"))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RESULTS / "E5_features.csv", index=False)
    print("\nsaved E5_features.csv")
    return df


def e6(fast=False):
    """Architecture decomposition ladder (Table IV / Fig. E). no-DR, {raw, ma6}.

    Shows WHERE the improvement comes from, in order:
      (1) weak LSTM  ->  (2) tuned LSTM (capacity+training)
      ->  (3) + CNN front-end  ->  (4) + 4-seed ensemble
    plus loss/target variants on the CNN-LSTM.
    """
    from src.evaluation.retrain import _train_lstm_on_modified_target as _v1
    split = build_dataset(verbose=False, split_mode="random", feature_set="env_lags")
    full_y = split["df_clean"][TARGET].to_numpy()
    base = _fast_lstm_cfg(fast)
    rows = []

    def _add(variant, step, y_true, y_pred):
        for g in ["raw", "ma6"]:
            w = GRAN[g]; m = _metrics(_ma(y_true, w), _ma(y_pred, w))
            rows.append(dict(experiment="E6", step=step, variant=variant,
                             granularity=g, **{k: round(v, 4) for k, v in m.items()}))

    # (1) weak LSTM: small + short + few epochs
    weak_cfg = {"hidden": 32, "layers": 1, "lookback": 12, "epochs": 15,
                "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.0}
    yt, yp, _ = _v1(split, full_y, cfg={**weak_cfg, **base}, verbose=False)
    _add("LSTM (weak: h32, lb12, 15ep)", 1, yt, yp)

    # (2) tuned LSTM (current LSTM_CFG: h128, lb36, 60ep, cosine)
    yt, yp, _ = _v1(split, full_y, cfg=base or None, verbose=False)
    _add("LSTM (tuned: h128, lb36, 60ep)", 2, yt, yp)

    # (3) + CNN front-end (single CNN-LSTM, MSE, no log)
    out = train_lstm_v2(split, y_full=full_y,
                        cfg={**base, "loss": "mse", "use_log": False}, verbose=False)
    _add("CNN-LSTM (single)", 3, out["y_true"], out["y_pred"])

    # (4) + 4-seed ensemble
    preds = []; yt2 = None
    for k in range(2 if fast else 4):
        o = train_lstm_v2(split, y_full=full_y,
                          cfg={**base, "loss": "mse", "use_log": False, "_seed_offset": k},
                          verbose=False)
        preds.append(o["y_pred"]); yt2 = o["y_true"]
    _add("CNN-LSTM + ensemble", 4, yt2, np.mean(preds, axis=0))

    # loss / target variants (on single CNN-LSTM) — design-choice ablation
    o = train_lstm_v2(split, y_full=full_y,
                      cfg={**base, "loss": "huber", "use_log": False}, verbose=False)
    _add("CNN-LSTM (Huber loss)", 5, o["y_true"], o["y_pred"])
    o = train_lstm_v2(split, y_full=full_y,
                      cfg={**base, "loss": "mse", "use_log": True}, verbose=False)
    _add("CNN-LSTM (log target)", 6, o["y_true"], o["y_pred"])

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "E6_architecture.csv", index=False)
    print("\nsaved E6_architecture.csv")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["e1e2", "e3", "e4", "e5", "e6", "all"])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    funcs = {"e1e2": e1e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}
    if a.exp == "all":
        for fn in funcs.values():
            fn(fast=a.fast)
    else:
        funcs[a.exp](fast=a.fast)
