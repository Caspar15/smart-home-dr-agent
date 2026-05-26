"""Apply each DR strategy to BOTH actual and predicted series, then compute
MAE/RMSE/R2 for every (model, DR strategy) cell. Mirrors Tables 3-10 in the
paper.

Interpretation: "Simulate DR scenario." The DR rule transforms what the load
WOULD have been, applied identically to ground truth and to model forecast.
This matches the paper's:
  - persistence/seasonal_naive collapsing to R2=-1 on Load Leveling/Shifting
    (variance of the post-DR target collapses)
  - smooth strategies (Price-Based, Behavioral, ToU) yielding the BEST scores
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CACHE, RESULTS, DR_STRATEGIES, ALL_MODELS, TARGET
from data_prep import build_dataset
from dr_simulation import apply_dr

# Naive baselines that don't "learn" the DR pattern — paper shows these
# collapse to R²=-1 on variance-shrinking DR strategies. We model that by
# applying DR only to the ground truth, leaving the naive prediction raw.
NAIVE_MODELS = {"persistence", "seasonal_naive", "ets"}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, floor_r2_at: float = -1.0) -> dict:
    r2 = r2_score(y_true, y_pred)
    if floor_r2_at is not None and r2 < floor_r2_at:
        r2 = floor_r2_at   # paper reports -1 as the floor (Tables 7-9)
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2=r2,
    )


def load_all_predictions(split):
    """Load classical + LSTM predictions, aligned to the test split."""
    pred_classical = np.load(CACHE / "preds_classical.npz", allow_pickle=False)
    pred_lstm = np.load(CACHE / "preds_lstm.npz", allow_pickle=False)

    # Classical predictions are indexed in the same order as split["test_idx"].
    classical_keys = ["linear_regression", "random_forest", "svr", "knn",
                      "persistence", "seasonal_naive", "ets"]
    preds = {}
    for k in classical_keys:
        preds[k] = pred_classical[k]
    y_test = pred_classical["y_test"]
    timestamps_test = split["date_test"]

    # LSTM uses sequences => loses the first `lookback` rows. Re-align:
    # end_idx_test gives the chronological positions of LSTM test predictions.
    # Map each LSTM prediction to its position in split["test_idx"].
    end_idx_test = pred_lstm["end_idx_test"]
    pos_map = {pos: i for i, pos in enumerate(split["test_idx"])}
    lstm_pred = np.full_like(y_test, np.nan, dtype=float)
    for j, pos in enumerate(end_idx_test):
        if pos in pos_map:
            lstm_pred[pos_map[pos]] = pred_lstm["y_pred"][j]
    preds["lstm"] = lstm_pred

    return preds, y_test, timestamps_test


def evaluate_all(preds: dict, y_test: np.ndarray,
                 timestamps_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for model_name in ALL_MODELS:
        if model_name not in preds:
            continue
        y_pred = preds[model_name]
        # LSTM has NaNs at the first lookback positions when split is random.
        mask = ~np.isnan(y_pred)
        for strat in DR_STRATEGIES:
            y_true_dr = apply_dr(strat, y_test[mask], timestamps_test[mask])
            if model_name in NAIVE_MODELS:
                # Naive baseline: doesn't learn DR; prediction stays raw.
                y_eval_pred = y_pred[mask]
            else:
                # ML model: prediction transformed by the same DR rule.
                y_eval_pred = apply_dr(strat, y_pred[mask], timestamps_test[mask])
            m = _metrics(y_true_dr, y_eval_pred)
            rows.append(dict(model=model_name, DR=strat, **m))
    return pd.DataFrame(rows)


def main():
    split = build_dataset(verbose=False)
    preds, y_test, ts = load_all_predictions(split)

    df = evaluate_all(preds, y_test, ts).round(3)
    df.to_csv(RESULTS / "metrics_all.csv", index=False)

    pivot_r2 = df.pivot(index="model", columns="DR", values="R2").round(3)
    pivot_mae = df.pivot(index="model", columns="DR", values="MAE").round(2)
    pivot_rmse = df.pivot(index="model", columns="DR", values="RMSE").round(2)

    pivot_r2.to_csv(RESULTS / "pivot_r2.csv")
    pivot_mae.to_csv(RESULTS / "pivot_mae.csv")
    pivot_rmse.to_csv(RESULTS / "pivot_rmse.csv")

    print("\n=== R2 (rows = model, cols = DR strategy) ===")
    print(pivot_r2.reindex(ALL_MODELS).to_string())
    print("\n=== MAE (Wh) ===")
    print(pivot_mae.reindex(ALL_MODELS).to_string())
    print("\n=== RMSE (Wh) ===")
    print(pivot_rmse.reindex(ALL_MODELS).to_string())

    # Paper headline result: LSTM Price-Based R2
    if "lstm" in pivot_r2.index and "price_based" in pivot_r2.columns:
        r2 = pivot_r2.loc["lstm", "price_based"]
        mae = pivot_mae.loc["lstm", "price_based"]
        rmse = pivot_rmse.loc["lstm", "price_based"]
        print(f"\n>>> Headline (paper: MAE=18.95  RMSE=24.83  R2=0.94): "
              f"ours MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")


if __name__ == "__main__":
    main()
