"""Full DR evaluation with the improved v2 CNN-LSTM ensemble.

For each of the 7 DR strategies:
  1. apply the DR rule to the full target series
  2. train a 4-model CNN-LSTM ensemble on the DR-modified target
  3. evaluate predictions under three smoothing granularities:
       raw  (10-min), ma6 (1-hour), ma12 (2-hour)

Outputs:
  results/metrics_lstm_v2.csv         long format (DR x scheme)
  results/lstm_v2_vs_paper.csv        side-by-side with paper Table 10
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CACHE, RESULTS, DR_STRATEGIES, TARGET
from src.data.preprocess import build_dataset
from src.dr.strategies import apply_dr
from src.forecasting.lstm_cnn import train_ensemble

# Paper Table 10 (LSTM).
PAPER_LSTM = {
    "peak_clipping":    (20.33, 26.84, 0.91),
    "valley_filling":   (20.17, 26.12, 0.92),
    "load_shifting":    (19.84, 25.59, 0.93),
    "load_leveling":    (21.47, 28.76, 0.89),
    "tou_optimization": (19.78, 25.21, 0.93),
    "price_based":      (18.95, 24.83, 0.94),
    "behavioral_dr":    (19.67, 25.47, 0.93),
}

N_ENSEMBLE = 4


def _ma(x, w):
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _metrics(yt, yp, floor=-1.0):
    r2 = r2_score(yt, yp)
    return (mean_absolute_error(yt, yp),
            float(np.sqrt(mean_squared_error(yt, yp))),
            max(floor, r2))


def main(n_ensemble: int = N_ENSEMBLE):
    split = build_dataset(verbose=False)
    df = split["df_clean"]
    full_y = df[TARGET].to_numpy()
    full_ts = df["date"].to_numpy()

    rows = []
    schemes = {"raw": 1, "ma6": 6, "ma12": 12}

    for strat in DR_STRATEGIES:
        t0 = time.time()
        y_dr_full = apply_dr(strat, full_y, full_ts)
        out = train_ensemble(split, y_full=y_dr_full, n_models=n_ensemble,
                             verbose=False)
        yt, yp = out["y_true"], out["y_pred"]
        for sname, w in schemes.items():
            mae, rmse, r2 = _metrics(_ma(yt, w), _ma(yp, w))
            rows.append(dict(DR=strat, scheme=sname, MAE=round(mae, 2),
                             RMSE=round(rmse, 2), R2=round(r2, 3)))
        print(f"[{strat:18s}] done in {time.time()-t0:.0f}s  "
              f"(raw R2={_metrics(yt, yp)[2]:.3f}, "
              f"ma6={_metrics(_ma(yt,6),_ma(yp,6))[2]:.3f}, "
              f"ma12={_metrics(_ma(yt,12),_ma(yp,12))[2]:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "metrics_lstm_v2.csv", index=False)

    # Side-by-side vs paper (use ma6 and ma12).
    cmp_rows = []
    for strat, (mae_p, rmse_p, r2_p) in PAPER_LSTM.items():
        for sch in ["raw", "ma6", "ma12"]:
            sub = res[(res["DR"] == strat) & (res["scheme"] == sch)].iloc[0]
            cmp_rows.append(dict(
                DR=strat, scheme=sch,
                R2_ours=sub["R2"], R2_paper=r2_p, R2_diff=round(sub["R2"] - r2_p, 3),
                MAE_ours=sub["MAE"], MAE_paper=mae_p,
                RMSE_ours=sub["RMSE"], RMSE_paper=rmse_p,
            ))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(RESULTS / "lstm_v2_vs_paper.csv", index=False)

    print("\n=== LSTM v2 R² by smoothing scheme vs paper ===")
    pivot = res.pivot(index="DR", columns="scheme", values="R2")[["raw", "ma6", "ma12"]]
    pivot["paper"] = pd.Series({k: v[2] for k, v in PAPER_LSTM.items()})
    print(pivot.to_string())

    print("\n=== headline: Price-Based ===")
    pb = res[res["DR"] == "price_based"]
    print(pb.to_string(index=False))
    print(f"paper price_based: MAE=18.95 RMSE=24.83 R2=0.94")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_ENSEMBLE)
    args = ap.parse_args()
    main(n_ensemble=args.n)
