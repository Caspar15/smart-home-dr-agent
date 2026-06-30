"""Autoregressive LSTM in PyTorch.

Input  : last L=lookback timesteps of (env features + previous Appliances)
Target : Appliances at t

The paper says LSTM was used to capture "long-term relationships in time series
data" and "utilize past consumption behavior" — i.e. autoregressive with
environmental covariates. Architecture details (layers/units/dropout) are not
specified in the paper, so we use a reasonable default and grid-tune if needed.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.config import CACHE, RESULTS, LSTM as LSTM_CFG, SEED, TARGET
from src.data.preprocess import build_dataset


def _build_sequences(df_clean: pd.DataFrame, feature_cols: list[str], lookback: int):
    """Build (X_seq, y_target, end_idx_in_full_df) over the FULL chronological
    series. Each sample is a length-`lookback` window of [features + past y]
    ending immediately before the row whose y is the target."""
    feats = df_clean[feature_cols].to_numpy(dtype=np.float32)
    y_all = df_clean[TARGET].to_numpy(dtype=np.float32)
    # Append lagged y as extra column (use past-only — append AFTER constructing
    # windows by shifting).
    n = len(df_clean)
    # For row t (target), window covers rows [t-lookback, t-1] inclusive.
    # Channels = features at those rows + y at those rows.
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
    return X_seq, y_target, end_idx


class LSTMNet(nn.Module):
    def __init__(self, in_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_lstm(split, cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = {**LSTM_CFG, **(cfg or {})}
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = split["df_clean"]
    feature_cols = split["feature_names"]

    # Build full chronological sequences first.
    X_seq, y_seq, end_idx = _build_sequences(df, feature_cols, cfg["lookback"])

    # Scale features and target using the TRAIN positions (from split).
    train_mask = np.isin(end_idx, split["train_idx"])
    test_mask = np.isin(end_idx, split["test_idx"])

    # Reshape for scaler fit: features (n*lookback x ch).
    n, L, C = X_seq.shape
    flat = X_seq.reshape(-1, C)
    scaler_X = MinMaxScaler()
    # Fit on training rows only.
    train_rows = np.repeat(train_mask, L)
    scaler_X.fit(flat[train_rows])
    flat_scaled = scaler_X.transform(flat)
    X_seq = flat_scaled.reshape(n, L, C).astype(np.float32)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_seq[train_mask].reshape(-1, 1))
    y_seq_s = scaler_y.transform(y_seq.reshape(-1, 1)).ravel().astype(np.float32)

    X_train, y_train = X_seq[train_mask], y_seq_s[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq_s[test_mask]
    end_idx_test = end_idx[test_mask]
    if verbose:
        print(f"[lstm] sequences: train {X_train.shape}  test {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(in_features=C, hidden=cfg["hidden"], layers=cfg["layers"],
                    dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    ds = TensorDataset(Xtr, ytr)
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True)

    t0 = time.time()
    model.train()
    for ep in range(cfg["epochs"]):
        epoch_loss = 0.0; nb = 0
        for xb, yb in loader:
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); nb += 1
        sched.step()
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"[lstm] ep {ep+1:3d}/{cfg['epochs']}  loss={epoch_loss/nb:.5f}")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_test).to(device)
        yhat_s = model(Xte).cpu().numpy()
    yhat = scaler_y.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    return dict(
        name="lstm",
        y_pred=yhat,
        y_true=y_true,
        end_idx_test=end_idx_test,
        train_time=train_time,
        cfg=cfg,
    )


def main():
    split = build_dataset()
    out = train_lstm(split)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yhat, ytrue = out["y_pred"], out["y_true"]
    mae = mean_absolute_error(ytrue, yhat)
    rmse = np.sqrt(mean_squared_error(ytrue, yhat))
    r2 = r2_score(ytrue, yhat)
    print(f"\n[lstm] no-DR test metrics: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}"
          f"  train_time={out['train_time']:.1f}s")
    np.savez_compressed(
        CACHE / "preds_lstm.npz",
        y_pred=yhat, y_true=ytrue,
        end_idx_test=out["end_idx_test"],
    )
    pd.DataFrame([dict(model="lstm", MAE=mae, RMSE=rmse, R2=r2)]).round(3) \
        .to_csv(RESULTS / "sanity_lstm_no_dr.csv", index=False)


if __name__ == "__main__":
    main()
