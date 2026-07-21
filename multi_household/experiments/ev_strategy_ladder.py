"""EV scheduling baseline ladder — does our fixed stagger beat dumb methods?

Reviewer question this answers: "your peak reduction comes from a 2-hour
stagger heuristic — would random placement or a trivial greedy do as well?"

All arms share the SAME window, seed, appliance agent and — critically — the
SAME nightly accept-draw stream (the accept RNG is consumed identically in
every strategy), so the comparison is paired: the same recommendations get
accepted everywhere, only the placement differs.

    no_dr      — baseline demand
    random     — accepted blocks placed uniformly in the 10 h overnight span
    edf        — earliest-deadline-first greedy (min-overlap, 8 h comfort cap)
    stagger    — ours (fixed 2 h offsets)
    (MPC perfect-foresight bound: see mpc_ladder.json — same window)

Run:  python -m multi_household.experiments.ev_strategy_ladder --days 14
Writes: reports/multi_household/ev_strategy_ladder.json
"""
from __future__ import annotations
import sys, argparse, json, random
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
import torch

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import compute_all_forecasts, rollout, REPORTS

SEED = 42


def _reseed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--accept", type=float, default=0.85)
    args = ap.parse_args()

    print(f"[1/2] Loading data ({args.days} d) ...")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=args.days * 144)
    T = min(len(hd[h]["test_df"]) for h in hd)
    dem = sum(hd[h]["test_df"]["aggregate_w"].values[:T].astype(float) for h in hd)
    rows = {"no_dr": {"peak_kw": round(float(dem.max()) / 1000, 2),
                      "p95_kw": round(float(np.percentile(dem, 95)) / 1000, 2)}}

    print(f"[2/2] Strategies (paired accept stream, accept={args.accept}) ...")
    for strat in ("random", "edf", "stagger"):
        _reseed()
        r = rollout(hd, mode="coordinated", user_accept=args.accept,
                    verbose=False, ev_smart=True, ev_strategy=strat)
        agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0) / 1000.0
        rows[strat] = {"peak_kw": round(float(agg.max()), 2),
                       "p95_kw": round(float(np.percentile(agg, 95)), 2)}
        b = rows["no_dr"]
        rows[strat]["peak_red_pct"] = round(
            100 * (b["peak_kw"] - rows[strat]["peak_kw"]) / b["peak_kw"], 2)
        rows[strat]["p95_red_pct"] = round(
            100 * (b["p95_kw"] - rows[strat]["p95_kw"]) / b["p95_kw"], 2)
        print(f"      {strat:8s} peak {rows[strat]['peak_kw']:6.2f} kW "
              f"(−{rows[strat]['peak_red_pct']}%)  "
              f"P95 {rows[strat]['p95_kw']:6.2f} (−{rows[strat]['p95_red_pct']}%)")

    try:
        mpc = json.loads((REPORTS / "mpc_ladder.json").read_text())
        rows["mpc_bound"] = mpc["mpc_bound"]
    except Exception:
        pass

    out = {"seed": SEED, "accept": args.accept, "days": args.days,
           "note": ("paired accept stream across strategies; MPC bound is the "
                    "perfect-foresight LP relaxation from mpc_ladder.json"),
           "rows": rows}
    p = REPORTS / "ev_strategy_ladder.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
