"""One-shot: build the Parquet cache for all 17 clean REFIT houses.

Run:  python -m multi_household.experiments.pre_cache
"""
from __future__ import annotations
import sys, io, time
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.config import CLEAN_HOUSES
from multi_household.data.refit_loader import load_house, coverage_report


def main():
    print(f"Caching {len(CLEAN_HOUSES)} clean houses ...\n")
    for h in CLEAN_HOUSES:
        t0 = time.time()
        df = load_house(h)
        dt = time.time() - t0
        rep = coverage_report(df)
        print(f"  House {h:2d}: {rep['n_rows']:>6d} rows, "
              f"{rep['days']} days, "
              f"agg_mean={rep['aggregate_mean_w']:.0f} W, "
              f"def_mean={rep['deferable_mean_w']:.0f} W, "
              f"nan={rep['nan_rate']:.2%}  "
              f"({'from cache' if dt < 1 else f'built {dt:.0f}s'})")
    print("\nDone.")


if __name__ == "__main__":
    main()
