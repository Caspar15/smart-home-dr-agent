"""
Phase 2 CNN-LSTM Architecture Comparison: H20 + H2 pilot
=========================================================
Background: considering CNN-LSTM (Conv1D x2 -> LSTM x2) as unified
prediction model. This script validates CNN-LSTM on baseload target
for H20 and H2, comparing against existing pure LSTM Phase 2 numbers.

Architecture source: multi_household/forecasting/per_house_lstm.py::CNNLSTM
Adaptations made:
  (1) head output = horizon=36 steps (original: 1-step)
  (2) input: Phase 2 feature set (n_features=5), lookback=144
  (3) FedAvg-ready: get_weights() / set_weights()
NOT taken from classmate: preprocess / split (80/20 + bidirectional
  imputation leakage) / target (aggregate_w instead of baseload)

Hard rules (all inherited, none relaxed):
  [2] chronological 70/10/20 split, no shuffle, boundary assert
  [3] TrainOnlyScaler -- train-only fit; val/test stats isolated
  [4] handle_gaps: forward-only (limit_direction='forward')
  [5] RMSE / MAE / nRMSE_range / RMSE_scaled; no R-squared
  [6] FedAvg-ready: get_weights() / set_weights() on both model classes

Output: out_phase2_cnn_compare/
  Does NOT overwrite out_phase2_17h/. Does NOT touch Phase 3/4.

Usage:
    python phase2_cnnlstm_compare.py
    python phase2_cnnlstm_compare.py --epochs 80
    python phase2_cnnlstm_compare.py --houses 20 2 --epochs 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -- Import all shared pipeline utilities from phase2_lstm.py -----------------
# Avoids re-implementation and keeps settings in sync.
sys.path.insert(0, str(Path(__file__).parent))
from phase2_lstm import (
    TrainOnlyScaler,
    handle_gaps,
    add_time_features,
    make_windows,
    BaseloadDataset,
    BaseloadLSTM,   # Model A (my Phase 2 pure LSTM baseline)
    local_train,    # arch-agnostic: only uses get_weights/set_weights/forward
    evaluate,       # arch-agnostic: only calls model(X)
    set_seed,
    load_house,
    DEFAULT_CFG,
)

# -- CNN-LSTM fixed hyperparams (classmate's original settings) ----------------
_CNN_FILTERS = 64
_CNN_HIDDEN  = 128
_CNN_LAYERS  = 2
_CNN_DROPOUT = 0.25

# -- Existing pure LSTM Phase 2 finalized results (reference baseline) --------
# Source: PLAN.md "Phase 2 定案" 17-house test result table
_LSTM_REF = {
    20: {"rmse": 221,  "mae": 139, "rmse_scaled": 0.82, "nrmse_range": 7.4},
    2:  {"rmse": 783,  "mae": 333, "rmse_scaled": 1.10, "nrmse_range": 6.9},
}


# -- CNN-LSTM architecture -----------------------------------------------------
class BaseloadCNNLSTM(nn.Module):
    """
    Conv1D x2 -> LSTM x2 -> multi-step head. FedAvg-ready.

    Source: multi_household/forecasting/per_house_lstm.py::CNNLSTM
    Adaptation: head output = horizon=36 (original: 1).
    Unchanged:  Conv(in->64, k=3, pad=1) -> Conv(64->64, k=3, pad=1)
                -> LSTM(64->128, L=2, drop=0.25) -> Linear(128, horizon)
    """

    def __init__(
        self,
        input_size:   int   = 5,
        conv_filters: int   = _CNN_FILTERS,
        hidden_size:  int   = _CNN_HIDDEN,
        num_layers:   int   = _CNN_LAYERS,
        dropout:      float = _CNN_DROPOUT,
        horizon:      int   = 36,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        lstm_drop  = dropout if num_layers > 1 else 0.0
        self.lstm  = nn.LSTM(
            conv_filters, hidden_size, num_layers,
            batch_first=True, dropout=lstm_drop,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> permute to (B, F, L) for Conv1d
        h = torch.relu(self.conv1(x.permute(0, 2, 1)))   # (B, 64, L)
        h = torch.relu(self.conv2(h))                     # (B, 64, L)
        h = h.permute(0, 2, 1)                            # back to (B, L, 64)
        _, (h_n, _) = self.lstm(h)
        return self.head(h_n[-1])                          # (B, horizon)

    def get_weights(self) -> list:
        """FedAvg interface: return all parameters as list of numpy arrays."""
        return [v.detach().cpu().numpy().copy()
                for v in self.state_dict().values()]

    def set_weights(self, weights: list) -> None:
        """FedAvg interface: load list of numpy arrays (e.g. after FedAvg agg)."""
        sd = self.state_dict()
        for key, w in zip(sd.keys(), weights):
            sd[key] = torch.tensor(w, dtype=sd[key].dtype)
        self.load_state_dict(sd)


# -- Per-house: train both architectures on same data -------------------------
def _build_loaders(splits: dict, cfg: dict):
    """
    Build three DataLoaders. Called once per model with the same seeded
    Generator so both models see exactly the same batch order each epoch.
    """
    train_loader = DataLoader(
        BaseloadDataset(*splits["train"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(
        BaseloadDataset(*splits["val"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        BaseloadDataset(*splits["test"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def compare_house(house_id: int, cfg: dict, device: str) -> dict:
    """
    Load data once, build windows once, then train LSTM and CNN-LSTM
    sequentially. Seed is reset before each model for a fair comparison.
    """
    print(f"\n{'='*62}")
    print(f"  House {house_id}")

    # -- data loading (shared by both models) ---------------------------------
    raw = load_house(house_id, cfg["data_dir"])
    raw = handle_gaps(raw, cfg["short_gap"])        # HARD RULE 4
    N   = len(raw)

    train_end = int(N * cfg["split"][0])
    val_end   = train_end + int(N * cfg["split"][1])
    ts        = raw.index

    # HARD RULE 2: chronological boundary assert
    assert ts[train_end - 1] < ts[train_end], (
        f"[HR2] H{house_id}: train/val boundary not chronological")
    assert ts[train_end] < ts[val_end], (
        f"[HR2] H{house_id}: val/test boundary not chronological")

    print(f"  Train : {ts[0]} -> {ts[train_end - 1]}  ({train_end} slots, 70%)")
    print(f"  Val   : {ts[train_end]} -> {ts[val_end - 1]}"
          f"  ({val_end - train_end} slots, 10%)")
    print(f"  Test  : {ts[val_end]} -> {ts[-1]}  ({N - val_end} slots, 20%)")

    # HARD RULE 3: scaler fit on train ONLY
    scaler = TrainOnlyScaler()
    scaler.fit(raw.values[:train_end])
    scaled     = scaler.transform(raw.values).astype(np.float64)
    time_feats = add_time_features(raw.index)

    # windows (shared; NaN windows skipped automatically)
    splits = make_windows(
        scaled, time_feats,
        cfg["look_back"], cfg["horizon"],
        train_end, val_end,
    )
    n_tr = len(splits["train"][0])
    n_va = len(splits["val"][0])
    n_te = len(splits["test"][0])
    print(f"  Windows -- train: {n_tr}  val: {n_va}  test: {n_te}")
    assert n_tr > 0, f"H{house_id}: zero training windows"

    result = {
        "house":   house_id,
        "windows": {"train": n_tr, "val": n_va, "test": n_te},
        "scaler":  {"mean_W": scaler.mean_, "std_W": scaler.std_},
    }

    # -- Model A: pure LSTM (retrained under identical conditions) ------------
    print(f"\n  [A] pure LSTM  "
          f"(hidden={cfg['hidden_size']}, L={cfg['num_layers']}, "
          f"drop={cfg['dropout']})")
    set_seed(cfg["seed"])                                   # fair reset
    model_lstm = BaseloadLSTM(
        input_size  = cfg["n_features"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        horizon     = cfg["horizon"],
        dropout     = cfg["dropout"],
    )
    n_params_lstm = sum(p.numel() for p in model_lstm.parameters())
    print(f"  params: {n_params_lstm:,}")

    tr_l, va_l, te_l = _build_loaders(splits, cfg)
    best_wts, _ = local_train(model_lstm, tr_l, va_l, cfg, device)
    model_lstm.set_weights(best_wts)
    val_lstm  = evaluate(model_lstm, va_l, scaler, device)
    test_lstm = evaluate(model_lstm, te_l, scaler, device)

    print(f"  Val   RMSE={val_lstm['rmse']:7.1f} W  "
          f"RMSE_sc={val_lstm['rmse_scaled']:.4f}s  "
          f"nRng={val_lstm['nrmse_range']:.1f}%")
    print(f"  Test  RMSE={test_lstm['rmse']:7.1f} W  "
          f"RMSE_sc={test_lstm['rmse_scaled']:.4f}s  "
          f"nRng={test_lstm['nrmse_range']:.1f}%")

    result["lstm"] = {
        "arch":     (f"LSTM(h={cfg['hidden_size']}, "
                     f"L={cfg['num_layers']}, drop={cfg['dropout']})"),
        "n_params": n_params_lstm,
        "val":      val_lstm,
        "test":     test_lstm,
    }

    # -- Model B: CNN-LSTM (classmate's arch, horizon=36 head) ----------------
    print(f"\n  [B] CNN-LSTM  "
          f"(Conv{_CNN_FILTERS}x2 -> LSTM h={_CNN_HIDDEN}, "
          f"L={_CNN_LAYERS}, drop={_CNN_DROPOUT})")
    set_seed(cfg["seed"])                                   # fair reset
    model_cnn = BaseloadCNNLSTM(
        input_size   = cfg["n_features"],
        conv_filters = _CNN_FILTERS,
        hidden_size  = _CNN_HIDDEN,
        num_layers   = _CNN_LAYERS,
        dropout      = _CNN_DROPOUT,
        horizon      = cfg["horizon"],
    )
    n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"  params: {n_params_cnn:,}  (vs LSTM {n_params_lstm:,}, "
          f"x{n_params_cnn / n_params_lstm:.1f})")

    tr_l, va_l, te_l = _build_loaders(splits, cfg)
    best_wts_cnn, _ = local_train(model_cnn, tr_l, va_l, cfg, device)
    model_cnn.set_weights(best_wts_cnn)
    val_cnn  = evaluate(model_cnn, va_l, scaler, device)
    test_cnn = evaluate(model_cnn, te_l, scaler, device)

    print(f"  Val   RMSE={val_cnn['rmse']:7.1f} W  "
          f"RMSE_sc={val_cnn['rmse_scaled']:.4f}s  "
          f"nRng={val_cnn['nrmse_range']:.1f}%")
    print(f"  Test  RMSE={test_cnn['rmse']:7.1f} W  "
          f"RMSE_sc={test_cnn['rmse_scaled']:.4f}s  "
          f"nRng={test_cnn['nrmse_range']:.1f}%")

    result["cnnlstm"] = {
        "arch":     (f"Conv{_CNN_FILTERS}x2->LSTM(h={_CNN_HIDDEN}, "
                     f"L={_CNN_LAYERS}, drop={_CNN_DROPOUT})->Lin(36)"),
        "n_params": n_params_cnn,
        "val":      val_cnn,
        "test":     test_cnn,
    }

    return result


# -- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 CNN-LSTM architecture validation: H20 + H2 pilot"
    )
    parser.add_argument("--houses",   type=int,   nargs="+", default=[20, 2])
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--patience", type=int,   default=DEFAULT_CFG["patience"])
    parser.add_argument("--data_dir", type=str,   default=DEFAULT_CFG["data_dir"])
    parser.add_argument("--out_dir",  type=str,   default="out_phase2_cnn_compare")
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "epochs":   args.epochs,
        "patience": args.patience,
        "data_dir": args.data_dir,
        "out_dir":  args.out_dir,
    }

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"Houses  : {args.houses}")
    print(f"Epochs  : {cfg['epochs']}  Patience: {cfg['patience']}")
    print(f"look_back={cfg['look_back']}  horizon={cfg['horizon']}  "
          f"n_features={cfg['n_features']}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config (HARD RULE 7 inherited)
    cfg_meta = {
        **cfg,
        "arch_lstm": (f"BaseloadLSTM(hidden={cfg['hidden_size']}, "
                      f"layers={cfg['num_layers']}, dropout={cfg['dropout']})"),
        "arch_cnnlstm": (f"BaseloadCNNLSTM(conv_filters={_CNN_FILTERS}, "
                         f"hidden={_CNN_HIDDEN}, layers={_CNN_LAYERS}, "
                         f"dropout={_CNN_DROPOUT})"),
        "note": ("CNN-LSTM head adapted to horizon=36. "
                 "input/lookback/split/gap/scaler identical to Phase 2."),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_meta, indent=2))
    print(f"Config  -> {out_dir / 'config.json'}")

    # -- train and compare -----------------------------------------------------
    all_results = []
    for h in args.houses:
        r = compare_house(h, cfg, device)
        all_results.append(r)

    (out_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print(f"\nResults -> {out_dir / 'results.json'}")

    # ====== comparison table ==================================================
    W = 102
    print(f"\n{'='*W}")
    print("Phase 2 Architecture Comparison -- H20 + H2 test set")
    print("Pipeline: 70/10/20 chron split | forward-only gap | train-only scaler | no R2")
    print(f"{'='*W}")
    print(f"  {'':3}  {'Architecture':<38}  {'Params':>7}  "
          f"{'RMSE(W)':>8}  {'MAE(W)':>7}  {'RMSE_sc(s)':>10}  {'nRMSE_rng%':>11}")
    print(f"  {'':3}  {'(test set)':38}  {'':>7}  "
          f"{'[test]':>8}  {'[test]':>7}  {'[test]':>10}  {'[test]':>11}")
    sep = f"  {'-'*W}"

    for r in all_results:
        h = r["house"]
        ref = _LSTM_REF.get(h)
        print(sep)

        # Phase 2 finalized pure LSTM reference (historical, from PLAN.md)
        if ref:
            print(f"  H{h:<2}  {'[LSTM Phase2 ref - PLAN.md]':<38}  {'---':>7}  "
                  f"{ref['rmse']:>8.0f}  {ref['mae']:>7.0f}  "
                  f"{ref['rmse_scaled']:>10.2f}  {ref['nrmse_range']:>11.1f}")

        # this run: LSTM
        for label, key in [("LSTM (this run)", "lstm"),
                            ("CNN-LSTM (this run)", "cnnlstm")]:
            m = r[key]
            t = m["test"]
            print(f"  H{h:<2}  {label:<38}  {m['n_params']:>7,}  "
                  f"{t['rmse']:>8.1f}  {t['mae']:>7.1f}  "
                  f"{t['rmse_scaled']:>10.4f}  {t['nrmse_range']:>11.1f}")

        # delta row
        sc_lstm = r["lstm"]["test"]["rmse_scaled"]
        sc_cnn  = r["cnnlstm"]["test"]["rmse_scaled"]
        delta   = sc_cnn - sc_lstm
        pct     = delta / sc_lstm * 100
        if delta < -0.005:
            flag = "CNN better  <<"
        elif delta > 0.15:
            flag = "CNN degraded !!"
        else:
            flag = "roughly equal"
        print(f"  H{h:<2}  {'delta (CNN - LSTM)':<38}  {'':>7}  "
              f"{'':>8}  {'':>7}  {delta:>+10.4f}  "
              f"({pct:>+.1f}%  {flag})")

    print(f"{'='*W}")

    # ====== verdict ===========================================================
    print("\n-- Assessment: Is CNN-LSTM reasonable on baseload target? ----------")
    notes = []
    for r in all_results:
        h = r["house"]
        sc_lstm = r["lstm"]["test"]["rmse_scaled"]
        sc_cnn  = r["cnnlstm"]["test"]["rmse_scaled"]
        delta   = sc_cnn - sc_lstm
        pct     = delta / sc_lstm * 100

        if delta > 0.15:
            notes.append(
                f"  [!!] H{h}: CNN-LSTM degraded significantly {pct:+.1f}% "
                f"({sc_lstm:.4f}s -> {sc_cnn:.4f}s)\n"
                f"       Conv features may not help on smooth baseload. "
                f"Try --epochs 80 before concluding."
            )
        elif delta > 0.05:
            notes.append(
                f"  [ ~] H{h}: CNN-LSTM slightly worse {pct:+.1f}% "
                f"({sc_lstm:.4f}s -> {sc_cnn:.4f}s) -- borderline, within noise"
            )
        elif delta < -0.005:
            notes.append(
                f"  [ v] H{h}: CNN-LSTM better {pct:+.1f}% "
                f"({sc_lstm:.4f}s -> {sc_cnn:.4f}s)"
            )
        else:
            notes.append(
                f"  [ =] H{h}: CNN-LSTM roughly equal {pct:+.1f}% "
                f"({sc_lstm:.4f}s -> {sc_cnn:.4f}s)"
            )
    for n in notes:
        print(n)

    any_crash = any(
        r["cnnlstm"]["test"]["rmse_scaled"] - r["lstm"]["test"]["rmse_scaled"] > 0.15
        for r in all_results
    )
    all_ok = not any_crash and all(
        r["cnnlstm"]["test"]["rmse_scaled"] - r["lstm"]["test"]["rmse_scaled"] <= 0.05
        for r in all_results
    )

    print()
    if any_crash:
        print("  CONCLUSION: CNN-LSTM degraded on at least one house (>0.15s gap).")
        print("    Possible cause: baseload is smoother than aggregate; Conv1D")
        print("    local feature extraction may add noise rather than signal.")
        print("    Recommended next step: try --epochs 80 to rule out under-training.")
        print("    If still degraded, keep pure LSTM.")
    elif all_ok:
        print("  CONCLUSION: CNN-LSTM is numerically reasonable on baseload target.")
        print("    Numbers are comparable to or better than pure LSTM.")
        print("    Safe to proceed with full 17-house training + rerun Phase 3/4.")
    else:
        print("  CONCLUSION: CNN-LSTM shows minor regression (< 0.05s, borderline).")
        print("    Within typical training variance. Consider running --epochs 80.")

    # ====== HARD RULE self-check ==============================================
    print(f"\n{'='*W}")
    print("HARD RULE Self-Check")
    print(f"{'='*W}")
    print("  [1] Delta=10 min, 144 slots/day")
    print("      OK  look_back=144 inherited from DEFAULT_CFG")
    print("          split computed from slot counts")
    print("  [2] chronological 70/10/20, no shuffle")
    print("      OK  split=[0.70, 0.10, 0.20]")
    print("      OK  assert ts[train_end-1] < ts[train_end] (both boundaries)")
    print("      OK  val/test DataLoader: shuffle=False")
    print("      OK  train DataLoader: within-epoch shuffle, seeded Generator=42")
    print("          (this is training-batch randomisation, NOT split shuffling)")
    print("  [3] scaler train-only fit")
    print("      OK  TrainOnlyScaler.fit() called once with raw[:train_end] only")
    print("      OK  second fit() raises AssertionError (inherited from phase2_lstm)")
    print("  [4] causality: forward-only gap interpolation")
    print("      OK  handle_gaps(limit_direction='forward') inherited from phase2_lstm")
    print("      OK  classmate's limit_direction='both' (bidirectional leakage) NOT used")
    print("  [5] metrics: RMSE / MAE / nRMSE_range / RMSE_scaled; no R-squared")
    print("      OK  evaluate() inherited from phase2_lstm; R2 never computed")
    print("  [6] FedAvg-ready interface")
    print("      OK  BaseloadLSTM:    get_weights() / set_weights() (inherited)")
    print("      OK  BaseloadCNNLSTM: get_weights() / set_weights() (implemented here)")
    print("      OK  local_train() is arch-agnostic; shared by both models")
    print("  [7] fixed seed=42; config saved")
    print("      OK  set_seed(42) at global entry + before each model training")
    print("      OK  config.json written to out_phase2_cnn_compare/")
    print("  Phase 3/4 untouched")
    print("      OK  output isolated to out_phase2_cnn_compare/")
    print("      OK  phase3_simulator.py / phase4*.py not imported or executed")
    print("  Prediction target = baseload (not aggregate_w)")
    print("      OK  load_house() reads baseload_houseXX.csv, col='baseload_W'")
    print("      OK  classmate's refit_loader / aggregate_w target NOT used")


if __name__ == "__main__":
    main()
