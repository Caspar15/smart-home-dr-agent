"""Per-house one-step forecast accuracy: CNN-LSTM vs persistence.

The manuscript reports this comparison (the learned forecaster loses to
persistence, and the DR result is insensitive to that) but no artifact backed
it — the numbers only ever existed in a terminal log. This writes them out.

Scoring window
--------------
The first `lookback` steps of the rollout are filled with true persistence
because the model has no history yet, so scoring them would compare
persistence against itself. We drop them and score the remaining
T - lookback steps, which is what the manuscript's "1,968 paired samples"
refers to (2016 - 48).

Both predictors are one-step-ahead on the aggregate: the LSTM output stored by
`compute_all_forecasts`, and persistence y_hat(t) = y(t-1), exactly the
`forecast_mode="persistence"` arm of the rollout.

Run:    python -m multi_household.experiments.forecast_eval
Writes: reports/multi_household/forecast_eval.json
"""
from __future__ import annotations
import sys, json
import numpy as np

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import compute_all_forecasts, REPORTS

TEST_STEPS = 2016


def main() -> None:
    print(f"[forecast_eval] {len(CLEAN_HOUSES)} houses x {TEST_STEPS} steps")
    data = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=TEST_STEPS)

    per_house: list[dict] = []
    lookback = None
    for h in sorted(data):
        d = data[h]
        y = d["test_df"]["aggregate_w"].values.astype(np.float64)
        yhat_lstm = np.asarray(d["forecast_w"], dtype=np.float64)

        # persistence, built exactly as rollout does it
        yhat_pers = np.roll(y, 1)
        yhat_pers[0] = y[0]

        # The model has no history for the first `lookback` steps; the rollout
        # fills them with persistence, so scoring them is self-comparison.
        # Find where the two series genuinely diverge. Exact != is wrong here:
        # the stored forecast is float32 and the target is float64, so the
        # identical head still differs in the last bit.
        same = np.isclose(yhat_lstm, yhat_pers, rtol=1e-4, atol=1e-2)
        lb = int(np.argmin(same)) if not same.all() else 0
        lookback = lb if lookback is None else min(lookback, lb)

        sl = slice(lb, None)
        mae_l = float(np.mean(np.abs(y[sl] - yhat_lstm[sl])))
        mae_p = float(np.mean(np.abs(y[sl] - yhat_pers[sl])))
        rmse_l = float(np.sqrt(np.mean((y[sl] - yhat_lstm[sl]) ** 2)))
        rmse_p = float(np.sqrt(np.mean((y[sl] - yhat_pers[sl]) ** 2)))
        per_house.append({
            "house": h, "n_scored": int(len(y) - lb),
            "mae_lstm_w": round(mae_l, 2), "mae_persistence_w": round(mae_p, 2),
            "rmse_lstm_w": round(rmse_l, 2), "rmse_persistence_w": round(rmse_p, 2),
            "lstm_wins": bool(mae_l < mae_p),
        })
        print(f"  H{h:02d}: MAE lstm {mae_l:7.2f} W | persistence {mae_p:7.2f} W"
              f"  {'LSTM' if mae_l < mae_p else 'persistence'} wins")

    wins = sum(r["lstm_wins"] for r in per_house)
    out = {
        "note": "one-step-ahead aggregate forecast on the headline test window; "
                "the first lookback steps are excluded because the rollout fills "
                "them with persistence for both arms",
        "n_houses": len(per_house),
        "test_steps": TEST_STEPS,
        "scored_steps_per_house": per_house[0]["n_scored"] if per_house else None,
        "lookback_steps_excluded": lookback,
        "mean_mae_lstm_w": round(float(np.mean([r["mae_lstm_w"] for r in per_house])), 2),
        "mean_mae_persistence_w": round(float(np.mean([r["mae_persistence_w"] for r in per_house])), 2),
        "mean_rmse_lstm_w": round(float(np.mean([r["rmse_lstm_w"] for r in per_house])), 2),
        "mean_rmse_persistence_w": round(float(np.mean([r["rmse_persistence_w"] for r in per_house])), 2),
        "lstm_wins_n_houses": int(wins),
        "per_house": per_house,
    }
    path = REPORTS / "forecast_eval.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  mean MAE: CNN-LSTM {out['mean_mae_lstm_w']} W vs "
          f"persistence {out['mean_mae_persistence_w']} W "
          f"(LSTM wins {wins}/{len(per_house)} houses)")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
