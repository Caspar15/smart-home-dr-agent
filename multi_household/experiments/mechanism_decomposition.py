"""Mechanism decomposition — the factorial that answers "where does the peak
reduction actually come from?" honestly.

The headline coordinated result mixes three ingredients: the synthetic EV
scenario itself, the per-house appliance rule agent, and the EV advisory
coordinator. The audit showed the informal chain (subtracting the EV column,
flipping --no-ev-smart) is a characterization, not a causal decomposition —
`ev_smart=False` still let the AGENT control the EV, and "natural REFIT" was
an after-the-fact column subtraction. This script runs the explicit factorial
on a single common window, seed and accept rate:

    row            synthetic EV   appliance agent      EV coordinator
    natural        no             off                  off
    ev_no_dr       yes            off                  off
    agent_with_ev  yes            on (incl. EV col)    off
    agent_no_ev    yes            on (excl. EV col)    off
    coord_only     yes            off                  on
    full           yes            on (excl. EV col)    on   ← headline

`natural` / `ev_no_dr` are pure demand statistics (no controller, so no
forecasters needed); `natural` uses the REAL no-EV pipeline (inject_ev=False
caches), not a column subtraction.

Run:  python -m multi_household.experiments.mechanism_decomposition --days 14
Writes: reports/multi_household/mechanism_decomposition.json
"""
from __future__ import annotations
import sys, argparse, json, random
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
import torch

from multi_household.config import CLEAN_HOUSES
from multi_household.data.preprocess import prepare_house
from multi_household.experiments.rollout import (
    compute_all_forecasts, rollout, REPORTS,
)

SEED = 42


def _reseed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def _stats(agg_w: np.ndarray) -> dict:
    return {"peak_kw": round(float(agg_w.max()) / 1000, 2),
            "p95_kw":  round(float(np.percentile(agg_w, 95)) / 1000, 2)}


def _demand_only(days: int, inject_ev: bool) -> dict:
    agg = None
    for h in CLEAN_HOUSES:
        d = prepare_house(h, inject_ev=inject_ev)["test_df"]
        a = d["aggregate_w"].values[:days * 144].astype(float)
        agg = a if agg is None else agg + a
    return _stats(agg)


def _rollout_row(hd, accept, **kw) -> dict:
    _reseed()
    r = rollout(hd, mode="coordinated", user_accept=accept, verbose=False, **kw)
    agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0)
    row = _stats(agg)
    ag = r["agent_state"]
    row["appliance_decisions"] = int(sum(ag[h].n_recommendations for h in ag))
    row["ev_decisions"] = int(sum(1 for x in r["recommendations"]
                                  if x.event_type == "ev_advisory"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--accept", type=float, default=0.85)
    args = ap.parse_args()

    print("[1/3] demand-only rows (no controller) ...")
    rows = {
        "natural":  {**_demand_only(args.days, inject_ev=False),
                     "ev": False, "agent": "off", "coordinator": "off"},
        "ev_no_dr": {**_demand_only(args.days, inject_ev=True),
                     "ev": True, "agent": "off", "coordinator": "off"},
    }
    print(f"    natural  {rows['natural']}")
    print(f"    ev_no_dr {rows['ev_no_dr']}")

    print("[2/3] controller rows (shared forecasts) ...")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=args.days * 144)
    rows["agent_with_ev"] = {**_rollout_row(hd, args.accept, ev_smart=False),
                             "ev": True, "agent": "on (incl. EV)",
                             "coordinator": "off"}
    rows["agent_no_ev"] = {**_rollout_row(hd, args.accept, ev_smart=False,
                                          exclude_ev_from_agent=True),
                           "ev": True, "agent": "on (excl. EV)",
                           "coordinator": "off"}
    rows["coord_only"] = {**_rollout_row(hd, args.accept, ev_smart=True,
                                         appliance_agent=False),
                          "ev": True, "agent": "off", "coordinator": "on"}
    rows["full"] = {**_rollout_row(hd, args.accept, ev_smart=True),
                    "ev": True, "agent": "on (excl. EV)", "coordinator": "on"}

    base = rows["ev_no_dr"]
    for k, v in rows.items():
        if v.get("ev"):
            v["peak_red_pct"] = round(100 * (base["peak_kw"] - v["peak_kw"])
                                      / base["peak_kw"], 1)
            v["p95_red_pct"] = round(100 * (base["p95_kw"] - v["p95_kw"])
                                     / base["p95_kw"], 1)

    print("[3/3] table (accept = %.2f, seed = %d):" % (args.accept, SEED))
    hdr = f"{'row':<14}{'peak kW':>9}{'P95 kW':>9}{'peak%':>8}{'P95%':>8}"
    print("    " + hdr)
    for k, v in rows.items():
        print(f"    {k:<14}{v['peak_kw']:>9.2f}{v['p95_kw']:>9.2f}"
              f"{v.get('peak_red_pct', float('nan')):>8}"
              f"{v.get('p95_red_pct', float('nan')):>8}")

    out = REPORTS / "mechanism_decomposition.json"
    out.write_text(json.dumps(
        {"seed": SEED, "accept": args.accept, "days": args.days, "rows": rows},
        indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
