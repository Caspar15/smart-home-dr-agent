"""Smoke test for REFIT loader.

Run:  python -m multi_household.experiments.test_loader

Loads ONE house (slow first time, fast after), prints a sanity summary,
then loads the appliance classification for all 17 clean houses and
prints which deferable appliances each one has.
"""
from __future__ import annotations
import sys, io, time
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.config import CLEAN_HOUSES
from multi_household.data.appliance_map import (
    HOUSE_APPLIANCES, appliance_summary,
)
from multi_household.data.refit_loader import load_house, coverage_report


def main():
    print("=" * 70)
    print("1. Loading House 1 at 10-min resolution ...")
    t0 = time.time()
    df = load_house(1)
    print(f"   loaded in {time.time()-t0:.1f}s — {len(df):,} rows")
    print()
    print("   Columns:")
    for c in df.columns:
        print(f"     {c}")
    print()
    print("   Coverage report:")
    rep = coverage_report(df)
    for k, v in rep.items():
        print(f"     {k:28s} {v}")
    print()
    print("   First 3 rows (selected columns):")
    show = ["time", "aggregate_w", "deferable_w",
            "non_controllable_w", "aggregate_wh"]
    print(df[show].head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("2. Appliance classification for all 17 clean houses\n")
    print(f"   {'House':<6}{'#def':<6}{'#semi':<7}{'#nc':<5} deferable appliances")
    print(f"   {'-'*6}{'-'*6}{'-'*7}{'-'*5} {'-'*40}")
    for h in CLEAN_HOUSES:
        s = appliance_summary(h)
        defs = ", ".join(name for _, name in s["deferable"])
        print(f"   {h:<6}{len(s['deferable']):<6}"
              f"{len(s['semi_deferable']):<7}{len(s['non_controllable']):<5} {defs}")


if __name__ == "__main__":
    main()
