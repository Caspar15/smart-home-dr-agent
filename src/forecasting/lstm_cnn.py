"""Improved LSTM (v2): CNN-LSTM hybrid + log target + cyclical time features
+ validation-based early stopping.

Architecture:
  Conv1D(in -> 64, k=3) -> ReLU -> Conv1D(64 -> 64, k=3) -> ReLU
  -> LSTM(64 -> hidden, layers) -> last hidden
  -> FC(hidden -> hidden//2) -> ReLU -> Dropout -> FC -> 1

Differences vs models_lstm.py:
  1. Conv1D front-end extracts local temporal patterns before the LSTM.
  2. Target is log1p-transformed (data is right-skewed) then scaled.
  3. Cyclical sin/cos encodings of hour, day-of-week, month appended.
  4. 10% validation tail of the TRAIN set + early stopping (patience).

Use:
  python models_lstm_v2.py            # no-DR, prints raw + MA6 R²
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CACHE, RESULTS, SEED, TARGET, LSTM as LSTM_CFG
from src.data.preprocess import build_dataset

CFG = dict(
    lookback=48,
    conv_channels=64,
    hidden=128,
    layers=2,
    dropout=0.2,
    batch=128,
    epochs=150,
    lr=1.5e-3,
    weight_decay=1e-6,
    patience=20,        # early stopping (more patient)
    val_frac=0.10,
    use_log=False,      # log compresses peaks -> hurts R²; keep raw scale
    loss="mse",         # "mse" chases peaks (good for R²); "huber" is robust
)


def _cyclical(df: pd.DataFrame) -> np.ndarray:
    """sin/cos encodings for hour, day-of-week, month."""
    h = df["hour"].to_numpy()
    dow = df["day_of_week"].to_numpy()
    mo = df["month"].to_numpy()
    feats = np.stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12),
    ], axis=1).astype(np.float32)
    return feats


def _build_sequences(df, feature_cols, lookback, y_full):
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    cyc = _cyclical(df)
    feats = np.concatenate([feats, cyc], axis=1)
    y_all = np.asarray(y_full, dtype=np.float32)
    n = len(df)
    n_seq = n - lookback
    C = feats.shape[1] + 1   # + lagged y channel
    X = np.empty((n_seq, lookback, C), dtype=np.float32)
    y = np.empty((n_seq,), dtype=np.float32)
    end_idx = np.empty((n_seq,), dtype=np.int64)
    for i in range(n_seq):
        t = i + lookback
        X[i, :, :-1] = feats[t - lookback:t]
        X[i, :, -1] = y_all[t - lookback:t]
        y[i] = y_all[t]
        end_idx[i] = t
    return X, y, end_idx


class CNNLSTM(nn.Module):
    def __init__(self, in_feat, conv_ch, hidden, layers, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feat, conv_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_ch, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: (B, L, C) -> conv wants (B, C, L)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).squeeze(-1)


def _ma(x, w=6):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def train_lstm_v2(split, y_full=None, cfg=None, verbose=True):
    cfg = {**CFG, **(cfg or {})}
    seed = SEED + int(cfg.get("_seed_offset", 0))
    torch.manual_seed(seed); np.random.seed(seed)
    df = split["df_clean"]
    feature_cols = split["feature_names"]
    if y_full is None:
        y_full = df[TARGET].to_numpy()

    # Optional log1p transform of target.
    if cfg["use_log"]:
        y_proc = np.log1p(np.clip(y_full, 0, None)).astype(np.float32)
    else:
        y_proc = np.asarray(y_full, dtype=np.float32)

    X, y, end_idx = _build_sequences(df, feature_cols, cfg["lookback"], y_proc)

    train_mask = np.isin(end_idx, split["train_idx"])
    test_mask = np.isin(end_idx, split["test_idx"])

    n, L, C = X.shape
    flat = X.reshape(-1, C)
    sx = MinMaxScaler(); sx.fit(flat[np.repeat(train_mask, L)])
    X = sx.transform(flat).reshape(n, L, C).astype(np.float32)

    sy = MinMaxScaler(); sy.fit(y[train_mask].reshape(-1, 1))
    y_s = sy.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    X_tr_all, y_tr_all = X[train_mask], y_s[train_mask]
    X_te, y_te = X[test_mask], y_s[test_mask]
    end_idx_test = end_idx[test_mask]

    # validation tail
    n_val = int(len(X_tr_all) * cfg["val_frac"])
    X_tr, y_tr = X_tr_all[:-n_val], y_tr_all[:-n_val]
    X_val, y_val = X_tr_all[-n_val:], y_tr_all[-n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(C, cfg["conv_channels"], cfg["hidden"], cfg["layers"],
                    cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.SmoothL1Loss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).to(device),
                      torch.from_numpy(y_tr).to(device)),
        batch_size=cfg["batch"], shuffle=True)
    Xval_t = torch.from_numpy(X_val).to(device)
    yval_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf"); best_state = None; bad = 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xval_t), yval_t).item()
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val = vloss; best_state = {k: v.detach().clone()
                                            for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"[lstm-v2] early stop @ ep {ep+1} (val={best_val:.5f})")
                break
        if verbose and (ep + 1) % 10 == 0:
            print(f"[lstm-v2] ep {ep+1:3d}  val_loss={vloss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        yhat_s = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

    # invert scale (-> invert log if used)
    yhat_p = sy.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
    ytrue_p = sy.inverse_transform(y_te.reshape(-1, 1)).ravel()
    if cfg["use_log"]:
        yhat = np.expm1(yhat_p); ytrue = np.expm1(ytrue_p)
    else:
        yhat = yhat_p; ytrue = ytrue_p
    return dict(y_pred=yhat, y_true=ytrue, end_idx_test=end_idx_test,
                train_time=time.time() - t0)


def train_ensemble(split, y_full=None, n_models=4, cfg=None, verbose=False):
    """Train n_models CNN-LSTMs with different seeds, average predictions."""
    base = {**CFG, **(cfg or {})}
    preds = []
    ytrue = None; end_idx = None
    for k in range(n_models):
        torch.manual_seed(SEED + k); np.random.seed(SEED + k)
        out = train_lstm_v2(split, y_full=y_full,
                            cfg={**base, "_seed_offset": k}, verbose=False)
        preds.append(out["y_pred"])
        ytrue = out["y_true"]; end_idx = out["end_idx_test"]
        if verbose:
            r2 = r2_score(ytrue, out["y_pred"])
            print(f"  [ensemble member {k+1}/{n_models}] raw R2={r2:.4f}")
    yhat = np.mean(preds, axis=0)
    return dict(y_pred=yhat, y_true=ytrue, end_idx_test=end_idx, train_time=0.0)


def main(ensemble: int = 4):
    split = build_dataset(verbose=False)
    if ensemble > 1:
        print(f"[lstm-v2] training {ensemble}-model ensemble ...")
        out = train_ensemble(split, n_models=ensemble, verbose=True)
    else:
        out = train_lstm_v2(split)
    yhat, ytrue = out["y_pred"], out["y_true"]
    raw_r2 = r2_score(ytrue, yhat)
    raw_mae = mean_absolute_error(ytrue, yhat)
    raw_rmse = np.sqrt(mean_squared_error(ytrue, yhat))
    # MA6
    yt_s, yh_s = _ma(ytrue), _ma(yhat)
    ma_r2 = r2_score(yt_s, yh_s)
    ma_mae = mean_absolute_error(yt_s, yh_s)
    ma_rmse = np.sqrt(mean_squared_error(yt_s, yh_s))
    print(f"\n[lstm-v2] NO-DR results  (train {out['train_time']:.0f}s)")
    print(f"  raw 10-min : MAE={raw_mae:.2f}  RMSE={raw_rmse:.2f}  R2={raw_r2:.4f}")
    print(f"  MA6  1-hour: MAE={ma_mae:.2f}  RMSE={ma_rmse:.2f}  R2={ma_r2:.4f}")
    print(f"  (v1 LSTM no-DR was: raw R2=0.64)")


if __name__ == "__main__":
    main()
