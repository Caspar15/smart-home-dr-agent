"""Train a per-house CNN-LSTM forecaster for every clean house.

Run:  python -m multi_household.experiments.train_all [--epochs 10]
"""
from __future__ import annotations
import sys, io, time, argparse
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.config import CLEAN_HOUSES
from multi_household.forecasting.per_house_lstm import (
    train_one_house, MODEL_DIR,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--houses", nargs="+", type=int, default=CLEAN_HOUSES)
    args = ap.parse_args()

    print(f"Training {len(args.houses)} per-house forecasters, "
          f"{args.epochs} epochs each ...\n")

    results = []
    grand_t0 = time.time()
    for h in args.houses:
        t0 = time.time()
        meta = train_one_house(h, epochs=args.epochs, lookback=args.lookback,
                               verbose=False)
        dt = time.time() - t0
        results.append({**meta, "time_s": round(dt, 1)})
        print(f"  House {h:2d}: MAE={meta['mae_wh']:6.1f} Wh, "
              f"RMSE={meta['rmse_wh']:6.1f}, "
              f"R²={meta['r2']:+.3f}   ({dt:.0f}s)")

    print(f"\nTotal time: {time.time()-grand_t0:.0f}s")
    print(f"Models saved in: {MODEL_DIR}")

    import json
    summary = MODEL_DIR / "training_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
