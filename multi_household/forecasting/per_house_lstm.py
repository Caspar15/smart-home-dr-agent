"""Per-house CNN-LSTM forecaster — same architecture as the conference paper
but lighter (smaller hidden, fewer epochs) so 17 houses can be trained
in under 30 minutes on a single GPU/CPU.

Each house gets its OWN model — no weight sharing, no centralized server.
This is the "Local-only" training paradigm and the baseline for any future
FL comparison.
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from multi_household.config import CACHE_DIR
from multi_household.data.preprocess import prepare_house


MODEL_DIR = CACHE_DIR.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --- architecture -----------------------------------------------------------

class CNNLSTM(nn.Module):
    """Two Conv1D → two LSTM → small head. v2: bumped capacity."""
    def __init__(self, n_features: int, lookback: int = 48,
                 conv_filters: int = 64, hidden: int = 128,
                 dropout: float = 0.25):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.conv1 = nn.Conv1d(n_features, conv_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=hidden,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, L, F) → permute for conv1d which wants (B, F, L)
        h = torch.relu(self.conv1(x.permute(0, 2, 1)))
        h = torch.relu(self.conv2(h))
        h = h.permute(0, 2, 1)         # back to (B, L, F)
        _, (h_n, _) = self.lstm(h)
        h_last = h_n[-1]               # last layer's hidden
        return self.head(h_last).squeeze(-1)


# --- windowing --------------------------------------------------------------

def make_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    """X: (T, F), y: (T,) → (N, L, F), (N,) where N = T - L."""
    N = len(X) - lookback
    if N <= 0:
        raise ValueError("series shorter than lookback")
    Xw = np.empty((N, lookback, X.shape[1]), dtype=np.float32)
    yw = np.empty((N,), dtype=np.float32)
    for i in range(N):
        Xw[i] = X[i:i+lookback]
        yw[i] = y[i+lookback]
    return Xw, yw


# --- train one house --------------------------------------------------------

def train_one_house(house_id: int,
                    lookback: int = 48,
                    epochs: int = 15,
                    batch_size: int = 128,
                    lr: float = 1e-3,
                    device: str | None = None,
                    val_frac: float = 0.15,
                    patience: int = 5,
                    verbose: bool = True) -> dict:
    """Train + save model for one house with validation-based early stopping.
    Returns dict with paths + metrics."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = prepare_house(house_id)
    train_df = data["train_df"]
    test_df  = data["test_df"]
    fcols    = data["feature_cols"]
    tcol     = data["target_col"]

    # Hold out last `val_frac` of TRAIN as validation (chronological).
    n_full_train = len(train_df)
    n_val   = int(n_full_train * val_frac)
    tr_part = train_df.iloc[:-n_val] if n_val > 0 else train_df
    val_part = train_df.iloc[-n_val:] if n_val > 0 else train_df.iloc[:0]

    # Standardize features + target (fit on tr_part only — no leakage)
    fx_scaler = StandardScaler().fit(tr_part[fcols].values)
    ty_scaler = StandardScaler().fit(tr_part[[tcol]].values)

    def _scale(df_part):
        X = fx_scaler.transform(df_part[fcols].values).astype(np.float32)
        y = ty_scaler.transform(df_part[[tcol]].values).astype(np.float32).ravel()
        return X, y

    Xtr, ytr = _scale(tr_part)
    Xva, yva = _scale(val_part)
    Xte, yte = _scale(test_df)

    Xtrw, ytrw = make_windows(Xtr, ytr, lookback)
    Xvaw, yvaw = make_windows(Xva, yva, lookback) if len(Xva) > lookback else (None, None)
    Xtew, ytew = make_windows(Xte, yte, lookback)

    model = CNNLSTM(n_features=len(fcols), lookback=lookback).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    Xtrw_t = torch.from_numpy(Xtrw).to(device)
    ytrw_t = torch.from_numpy(ytrw).to(device)
    if Xvaw is not None:
        Xvaw_t = torch.from_numpy(Xvaw).to(device)
        yvaw_t = torch.from_numpy(yvaw).to(device)
    n_train = len(Xtrw_t)

    history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xtrw_t[idx], ytrw_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        train_loss = total / n_train

        # Validation
        val_loss = float("nan")
        if Xvaw is not None:
            model.eval()
            with torch.no_grad():
                # batched val to avoid OOM
                preds = []
                for i in range(0, len(Xvaw_t), 1024):
                    preds.append(model(Xvaw_t[i:i+1024]))
                pv = torch.cat(preds)
                val_loss = float(loss_fn(pv, yvaw_t))

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

        history.append((train_loss, val_loss))
        if verbose and (ep == 0 or (ep+1) % 5 == 0 or ep == epochs-1):
            print(f"   epoch {ep+1:3d}/{epochs} "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"(best={best_val:.4f}, bad={bad_epochs})")
        if bad_epochs >= patience:
            if verbose:
                print(f"   early stop at epoch {ep+1} (val not improving)")
            break

    # Roll back to best validation state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval on test
    model.eval()
    with torch.no_grad():
        Xtew_t = torch.from_numpy(Xtew).to(device)
        ytew_t = torch.from_numpy(ytew).to(device)
        preds_scaled = model(Xtew_t).cpu().numpy()
    # Inverse scale to Watts
    preds_w = ty_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    truth_w = ty_scaler.inverse_transform(ytew.reshape(-1, 1)).ravel()
    err = preds_w - truth_w
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float((err**2).sum())
    ss_tot = float(((truth_w - truth_w.mean())**2).sum())
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    # Save model + scaler stats + meta
    model_path = MODEL_DIR / f"house_{house_id:02d}.pt"
    meta_path  = MODEL_DIR / f"house_{house_id:02d}_meta.json"
    torch.save({
        "model_state": model.state_dict(),
        "n_features": len(fcols),
        "lookback": lookback,
        "fx_mean": fx_scaler.mean_.tolist(),
        "fx_scale": fx_scaler.scale_.tolist(),
        "ty_mean": float(ty_scaler.mean_[0]),
        "ty_scale": float(ty_scaler.scale_[0]),
        "feature_cols": fcols,
        "target_col": tcol,
    }, model_path)

    meta = {
        "house_id": house_id,
        "n_train_windows": len(Xtrw),
        "n_test_windows":  len(Xtew),
        "lookback": lookback,
        "epochs": epochs,
        "mae_wh":   round(mae, 2),
        "rmse_wh":  round(rmse, 2),
        "r2":       round(r2, 4),
        "final_train_loss": round(history[-1][0], 4),
        "final_val_loss":   round(history[-1][1], 4) if not np.isnan(history[-1][1]) else None,
        "best_val_loss":    round(best_val, 4) if best_val < float("inf") else None,
        "epochs_actually_run": len(history),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"   → House {house_id}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}")
        print(f"   saved {model_path}")
    return meta


# --- load + inference -------------------------------------------------------

def load_forecaster(house_id: int, device: str | None = None):
    """Return (model, ckpt_dict) loaded from disk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_DIR / f"house_{house_id:02d}.pt",
                      map_location=device, weights_only=False)
    model = CNNLSTM(n_features=ckpt["n_features"],
                    lookback=ckpt["lookback"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_window(model, ckpt, window: np.ndarray, device: str | None = None) -> float:
    """Predict one-step-ahead for a single window of features.
    `window`: (lookback, n_features) raw (unscaled). Returns Watts (unscaled)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fx_mean  = np.array(ckpt["fx_mean"],  dtype=np.float32)
    fx_scale = np.array(ckpt["fx_scale"], dtype=np.float32)
    w_scaled = ((window - fx_mean) / fx_scale).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(w_scaled[None, ...]).to(device)
        pred_scaled = model(x).cpu().numpy()[0]
    return float(pred_scaled * ckpt["ty_scale"] + ckpt["ty_mean"])


def predict_24h_recursive(model, ckpt, feature_history: np.ndarray,
                           target_col_idx: int | None = None,
                           lag_col_indices: list[int] | None = None,
                           horizon_steps: int = 144,
                           device: str | None = None) -> np.ndarray:
    """Recursive 24-hour-ahead forecast.

    Strategy: feed predicted ŷ(t+1) back into the input window for the
    next iteration. Lag features in the input that point at the target
    column are updated each step; other features (time, weather) are
    assumed known (we use the value at the prediction's target step).

    Args:
        feature_history: (lookback, n_features) raw window ending at t.
        target_col_idx: index of `aggregate_w` in the feature vector
                         (None if the target is NOT in the feature window;
                          most CNN-LSTMs don't include past target directly,
                          they have lag columns instead).
        lag_col_indices: indices of `aggregate_w_lag1` columns that should
                          be updated with the new prediction each iteration.

    For simplicity and honesty, this does pure RECURSIVE rollout: predict,
    shift window, predict again. Error compounds — this is realistic
    and we don't try to hide it.

    Returns array of `horizon_steps` predictions in Watts (unscaled).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fx_mean  = np.array(ckpt["fx_mean"],  dtype=np.float32)
    fx_scale = np.array(ckpt["fx_scale"], dtype=np.float32)
    ty_mean  = float(ckpt["ty_mean"])
    ty_scale = float(ckpt["ty_scale"])
    lookback = ckpt["lookback"]

    window = feature_history.copy().astype(np.float32)
    preds_w = np.zeros(horizon_steps, dtype=np.float32)

    with torch.no_grad():
        for h in range(horizon_steps):
            w_scaled = (window - fx_mean) / fx_scale
            x = torch.from_numpy(w_scaled[None, ...]).to(device)
            p_scaled = float(model(x).cpu().numpy()[0])
            p_w = p_scaled * ty_scale + ty_mean
            preds_w[h] = p_w

            # Shift window: drop oldest row, append a new row that re-uses the
            # last row's non-target features and substitutes p_w into lag1
            # (and shifts other lag features by one step).
            new_row = window[-1].copy()
            if lag_col_indices is not None and len(lag_col_indices) > 0:
                # lag_col_indices is sorted [lag1_idx, lag2_idx, lag3_idx, ...]
                # so we shift lag2 ← lag1, lag1 ← p_w
                for i in reversed(range(1, len(lag_col_indices))):
                    new_row[lag_col_indices[i]] = window[-1, lag_col_indices[i-1]]
                new_row[lag_col_indices[0]] = p_w
            window = np.vstack([window[1:], new_row[None, :]])
    return preds_w
