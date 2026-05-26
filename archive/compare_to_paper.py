"""Side-by-side comparison of our reproduction to the paper's Tables 3-10.

Outputs:
  results/comparison_<model>.csv  -- per-model paper vs ours
  results/comparison_overview.csv -- single wide table for the writeup
"""
from __future__ import annotations
import pandas as pd

from config import RESULTS, DR_STRATEGIES

# Paper values, transcribed from the article.
PAPER = {
    "linear_regression": {  # Table 3
        "peak_clipping":    (33.72, 43.51, 0.78),
        "valley_filling":   (34.01, 42.83, 0.80),
        "load_shifting":    (31.97, 41.56, 0.82),
        "load_leveling":    (37.65, 45.78, 0.71),
        "tou_optimization": (32.44, 42.12, 0.81),
        "price_based":      (31.55, 40.93, 0.83),
        "behavioral_dr":    (32.85, 42.36, 0.80),
    },
    "random_forest": {  # Table 4
        "peak_clipping":    (26.41, 34.55, 0.86),
        "valley_filling":   (26.17, 33.98, 0.87),
        "load_shifting":    (25.66, 33.11, 0.88),
        "load_leveling":    (28.75, 36.22, 0.83),
        "tou_optimization": (25.87, 33.57, 0.87),
        "price_based":      (25.13, 32.44, 0.89),
        "behavioral_dr":    (25.60, 33.30, 0.88),
    },
    "svr": {  # Table 5
        "peak_clipping":    (30.71, 39.42, 0.81),
        "valley_filling":   (30.33, 38.91, 0.82),
        "load_shifting":    (29.51, 37.86, 0.83),
        "load_leveling":    (32.45, 40.63, 0.78),
        "tou_optimization": (29.02, 37.55, 0.83),
        "price_based":      (28.66, 36.73, 0.85),
        "behavioral_dr":    (28.95, 36.89, 0.84),
    },
    "knn": {  # Table 6
        "peak_clipping":    (31.67, 39.05, 0.81),
        "valley_filling":   (30.94, 38.11, 0.82),
        "load_shifting":    (29.80, 36.79, 0.83),
        "load_leveling":    (33.82, 41.60, 0.76),
        "tou_optimization": (29.52, 36.44, 0.84),
        "price_based":      (29.66, 36.62, 0.83),
        "behavioral_dr":    (29.44, 36.18, 0.84),
    },
    "persistence": {  # Table 7
        "peak_clipping":    (16.247, 19.497, 0.743),
        "valley_filling":   (18.392, 22.07, 0.69),
        "load_shifting":    (35.000, 42.00, -1.0),
        "load_leveling":    (35.000, 42.00, -1.0),
        "tou_optimization": (17.607, 21.128, 0.706),
        "price_based":      (14.124, 16.948, 0.745),
        "behavioral_dr":    (18.995, 22.794, 0.632),
    },
    "seasonal_naive": {  # Table 8
        "peak_clipping":    (15.091, 18.109, 0.628),
        "valley_filling":   (15.825, 18.991, 0.679),
        "load_shifting":    (35.000, 42.000, -1.0),
        "load_leveling":    (35.000, 42.000, -1.0),
        "tou_optimization": (15.753, 18.903, 0.655),
        "price_based":      (16.736, 20.084, 0.718),
        "behavioral_dr":    (15.198, 18.238, 0.677),
    },
    "ets": {  # Table 9
        "peak_clipping":    (15.07, 18.084, 0.732),
        "valley_filling":   (17.24, 20.688, 0.631),
        "load_shifting":    (34.00, 42.00, -1.0),
        "load_leveling":    (34.00, 42.00, -1.0),
        "tou_optimization": (18.389, 22.067, 0.725),
        "price_based":      (17.264, 20.717, 0.691),
        "behavioral_dr":    (16.596, 19.915, 0.681),
    },
    "lstm": {  # Table 10
        "peak_clipping":    (20.33, 26.84, 0.91),
        "valley_filling":   (20.17, 26.12, 0.92),
        "load_shifting":    (19.84, 25.59, 0.93),
        "load_leveling":    (21.47, 28.76, 0.89),
        "tou_optimization": (19.78, 25.21, 0.93),
        "price_based":      (18.95, 24.83, 0.94),
        "behavioral_dr":    (19.67, 25.47, 0.93),
    },
}


def main():
    ours = pd.read_csv(RESULTS / "metrics_retrain.csv")
    rows = []
    for model, by_dr in PAPER.items():
        for strat, (mae_p, rmse_p, r2_p) in by_dr.items():
            sub = ours[(ours["model"] == model) & (ours["DR"] == strat)]
            if sub.empty:
                continue
            mae_o = float(sub["MAE"].iloc[0])
            rmse_o = float(sub["RMSE"].iloc[0])
            r2_o = float(sub["R2"].iloc[0])
            rows.append(dict(
                model=model, DR=strat,
                MAE_paper=mae_p, MAE_ours=round(mae_o, 2), MAE_diff=round(mae_o - mae_p, 2),
                RMSE_paper=rmse_p, RMSE_ours=round(rmse_o, 2), RMSE_diff=round(rmse_o - rmse_p, 2),
                R2_paper=r2_p, R2_ours=round(r2_o, 3), R2_diff=round(r2_o - r2_p, 3),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_overview.csv", index=False)

    print("\n=== Side-by-side: paper vs ours (LSTM, the headline model) ===")
    lstm = df[df["model"] == "lstm"].drop(columns=["model"])
    print(lstm.to_string(index=False))

    print("\n=== R^2 column-by-column (all models) ===")
    r2_compare = df.pivot_table(index="model", columns="DR",
                                values=["R2_paper", "R2_ours"])
    print(r2_compare.to_string())

    # Summary statistics
    print("\n=== Aggregate gap ===")
    for model in PAPER.keys():
        sub = df[df["model"] == model]
        print(f"  {model:18s}  ΔR² mean={sub['R2_diff'].mean():+.3f}  "
              f"ΔMAE mean={sub['MAE_diff'].mean():+.2f}  "
              f"ΔRMSE mean={sub['RMSE_diff'].mean():+.2f}")


if __name__ == "__main__":
    main()
