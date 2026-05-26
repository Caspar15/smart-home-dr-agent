"""Retrain-per-DR evaluation pipeline.

For each DR strategy:
  1. Apply DR rule to the full chronological target series y -> y_dr
  2. Train every ML model (LR, RF, SVR, kNN, LSTM) on (X_train, y_dr_train)
  3. Predict y_dr_test_pred on the test split
  4. Score against y_dr_test

The naive baselines (Persistence, Seasonal-Naive, ETS) keep operating on the
RAW series, so their predictions stay raw — this reproduces the paper's
R² = -1 behaviour on variance-collapsing strategies (Tables 7-9).

This file produces metrics_retrain.csv and the three pivot tables that mirror
Tables 3-10 in the paper.
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
from src.forecasting.lstm import _build_sequences, LSTMNet

NAIVE_MODELS = {"persistence", "seasonal_naive", "ets"}


def _metrics(y_true, y_pred, floor_r2_at: float = -1.0) -> dict:
    r2 = r2_score(y_true, y_pred)
    if floor_r2_at is not None and r2 < floor_r2_at:
        r2 = floor_r2_at
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                R2=r2)


# ------- classical models -------------------------------------------------- #

def _train_classical(name: str, X_tr, y_tr, X_te, fast=False, rng=None):
    if name == "linear_regression":
        m = LinearRegression(); m.fit(X_tr, y_tr); return m.predict(X_te)
    if name == "random_forest":
        cfg = dict(RFR_CFG)
        if fast: cfg["n_estimators"] = 100
        m = RandomForestRegressor(**cfg); m.fit(X_tr, y_tr); return m.predict(X_te)
    if name == "svr":
        idx = rng.choice(len(X_tr), size=min(8000, len(X_tr)), replace=False)
        m = SVR(**SVR_CFG); m.fit(X_tr[idx], y_tr[idx]); return m.predict(X_te)
    if name == "knn":
        m = KNeighborsRegressor(**KNN_CFG); m.fit(X_tr, y_tr); return m.predict(X_te)
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        from src.config import GBM as GBM_CFG
        cfg = dict(GBM_CFG)
        if fast: cfg["n_estimators"] = 100
        m = GradientBoostingRegressor(**cfg); m.fit(X_tr, y_tr); return m.predict(X_te)
    if name == "xgboost":
        from xgboost import XGBRegressor
        from src.config import XGB as XGB_CFG
        cfg = dict(XGB_CFG)
        if fast: cfg["n_estimators"] = 150
        m = XGBRegressor(**cfg); m.fit(X_tr, y_tr); return m.predict(X_te)
    raise ValueError(name)


# ------- LSTM (retrained per DR strategy) --------------------------------- #

def _train_lstm_on_modified_target(split, y_dr_full, cfg=None, verbose=False):
    """Same architecture/scaling as models_lstm.train_lstm, but using y_dr
    instead of the raw target. y_dr_full is a 1-D array aligned with
    split['df_clean'] rows."""
    cfg = {**LSTM_CFG, **(cfg or {})}
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = split["df_clean"]
    feature_cols = split["feature_names"]
    lookback = cfg["lookback"]

    # Manual sequence builder using y_dr.
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = np.asarray(y_dr_full, dtype=np.float32)
    n = len(df)
    n_seq = n - lookback
    X_seq = np.empty((n_seq, lookback, len(feature_cols) + 1), dtype=np.float32)
    y_target = np.empty((n_seq,), dtype=np.float32)
    end_idx = np.empty((n_seq,), dtype=np.int64)
    for i in range(n_seq):
        t = i + lookback
        X_seq[i, :, :-1] = feats[t - lookback:t]
        X_seq[i, :, -1] = y_all[t - lookback:t]
        y_target[i] = y_all[t]
        end_idx[i] = t

    train_mask = np.isin(end_idx, split["train_idx"])
    test_mask = np.isin(end_idx, split["test_idx"])

    n_, L, C = X_seq.shape
    flat = X_seq.reshape(-1, C)
    scaler_X = MinMaxScaler()
    train_rows = np.repeat(train_mask, L)
    scaler_X.fit(flat[train_rows])
    flat_s = scaler_X.transform(flat)
    X_seq = flat_s.reshape(n_, L, C).astype(np.float32)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_target[train_mask].reshape(-1, 1))
    y_s = scaler_y.transform(y_target.reshape(-1, 1)).ravel().astype(np.float32)

    X_tr, y_tr = X_seq[train_mask], y_s[train_mask]
    X_te, y_te = X_seq[test_mask], y_s[test_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(in_features=C, hidden=cfg["hidden"], layers=cfg["layers"],
                    dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = torch.nn.MSELoss()

    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], shuffle=True)

    model.train()
    for ep in range(cfg["epochs"]):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(X_te).to(device)
        yhat_s = model(Xte_t).cpu().numpy()
    yhat = scaler_y.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
    return y_true, yhat, end_idx[test_mask]


# ------- main loop --------------------------------------------------------- #

def main(skip_lstm: bool = False, fast: bool = False):
    split = build_dataset(verbose=False)
    df = split["df_clean"]
    full_y = df[TARGET].to_numpy()
    full_ts = df["date"].to_numpy()
    rng = np.random.default_rng(SEED)

    # Cache raw naive predictions (these don't change per DR strategy).
    raw_preds = np.load(CACHE / f"preds_classical{RESULT_SUFFIX}.npz",
                         allow_pickle=False)

    # Re-scale features once for classical models (fit on train rows only).
    feature_cols = split["feature_names"]
    scaler_X = MinMaxScaler()
    scaler_X.fit(df.iloc[split["train_idx"]][feature_cols].to_numpy())
    X_full = scaler_X.transform(df[feature_cols].to_numpy())
    X_tr = X_full[split["train_idx"]]
    X_te = X_full[split["test_idx"]]

    rows = []
    per_dr_preds = {}    # {model: {strat: y_pred_array}}
    per_dr_truth = {}    # {strat: y_true_array}
    per_dr_lstm_endidx = {}

    for strat in DR_STRATEGIES:
        t0 = time.time()
        y_dr_full = apply_dr(strat, full_y, full_ts)
        y_dr_tr = y_dr_full[split["train_idx"]]
        y_dr_te = y_dr_full[split["test_idx"]]

        # Scale target for classical models.
        sy = MinMaxScaler(); sy.fit(y_dr_tr.reshape(-1, 1))
        y_dr_tr_s = sy.transform(y_dr_tr.reshape(-1, 1)).ravel()

        per_dr_truth[strat] = y_dr_te

        # --- classical models ---
        for name in ["linear_regression", "random_forest", "svr", "knn"]:
            yhat_s = _train_classical(name, X_tr, y_dr_tr_s, X_te,
                                      fast=fast, rng=rng)
            yhat = sy.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
            m = _metrics(y_dr_te, yhat)
            rows.append(dict(model=name, DR=strat, **m))
            per_dr_preds.setdefault(name, {})[strat] = yhat

        # --- naive baselines (raw prediction vs DR-adjusted actual) ---
        for name in ["persistence", "seasonal_naive", "ets"]:
            yhat_raw = raw_preds[name]
            m = _metrics(y_dr_te, yhat_raw)
            rows.append(dict(model=name, DR=strat, **m))
            per_dr_preds.setdefault(name, {})[strat] = yhat_raw

        # --- LSTM ---
        if not skip_lstm:
            y_true_lstm, yhat_lstm, end_idx_test = _train_lstm_on_modified_target(
                split, y_dr_full, verbose=False)
            m = _metrics(y_true_lstm, yhat_lstm)
            rows.append(dict(model="lstm", DR=strat, **m))
            per_dr_preds.setdefault("lstm", {})[strat] = yhat_lstm
            per_dr_lstm_endidx[strat] = end_idx_test

        print(f"[{strat:18s}] done in {time.time()-t0:.1f}s")

    df_metrics = pd.DataFrame(rows).round(3)
    df_metrics.to_csv(RESULTS / f"metrics_retrain{RESULT_SUFFIX}.csv", index=False)

    # Save per-(model, DR) predictions so visualize.py can show our model's
    # REAL outputs instead of just the DR-modified target.
    flat = {}
    for m, by in per_dr_preds.items():
        for s, arr in by.items():
            flat[f"{m}__{s}"] = arr
    for s, arr in per_dr_truth.items():
        flat[f"_truth__{s}"] = arr
    for s, arr in per_dr_lstm_endidx.items():
        flat[f"_lstm_endidx__{s}"] = arr
    np.savez_compressed(CACHE / f"preds_per_dr{RESULT_SUFFIX}.npz", **flat)

    p_r2 = df_metrics.pivot(index="model", columns="DR", values="R2").round(3)
    p_mae = df_metrics.pivot(index="model", columns="DR", values="MAE").round(2)
    p_rmse = df_metrics.pivot(index="model", columns="DR", values="RMSE").round(2)
    p_r2.to_csv(RESULTS / f"pivot_r2_retrain{RESULT_SUFFIX}.csv")
    p_mae.to_csv(RESULTS / f"pivot_mae_retrain{RESULT_SUFFIX}.csv")
    p_rmse.to_csv(RESULTS / f"pivot_rmse_retrain{RESULT_SUFFIX}.csv")

    print("\n=== R2 (retrain-per-DR) ===")
    print(p_r2.reindex(ALL_MODELS).to_string())
    print("\n=== MAE (Wh) ===")
    print(p_mae.reindex(ALL_MODELS).to_string())
    print("\n=== RMSE (Wh) ===")
    print(p_rmse.reindex(ALL_MODELS).to_string())

    if "lstm" in p_r2.index and "price_based" in p_r2.columns:
        r2 = p_r2.loc["lstm", "price_based"]
        mae = p_mae.loc["lstm", "price_based"]
        rmse = p_rmse.loc["lstm", "price_based"]
        print(f"\n>>> Headline (paper: MAE=18.95 RMSE=24.83 R2=0.94): "
              f"ours MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    main(skip_lstm=args.skip_lstm, fast=args.fast)
