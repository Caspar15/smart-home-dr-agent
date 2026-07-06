"""Multi-seed robustness: the headline + accept sweep with error bars.

All headline numbers so far are single-seed (seed 42). This runs the
coordinated rollout across several seeds — reseeding BOTH the appliance-level
accept sampling AND the nightly EV-advisory accept decisions — and reports
mean ± std of peak / P95 per accept level. This is what makes the numbers
paper-grade.

Run:  python -m multi_household.experiments.multiseed --days 14
Writes: reports/multi_household/multiseed_results.json
        figures/multi_household/multiseed_accept.png
"""
from __future__ import annotations
import sys, argparse, json
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import (
    compute_all_forecasts, rollout, REPORTS,
)
from multi_household.experiments.metrics import FIGS

SEEDS = [41, 42, 43, 44, 45]
ACCEPTS = [0.0, 0.5, 0.85, 1.0]


def _reseed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run(days: int, seeds: list[int], accepts: list[float]) -> dict:
    print(f"[1/3] Loading data + forecasts ({days} d) ...")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=days * 144)

    # Baseline aggregate is deterministic (no user sampling)
    any_h = sorted(hd)[0]
    T = min(len(hd[h]["test_df"]) for h in hd)
    demand_agg = sum(hd[h]["test_df"]["aggregate_w"].values[:T] for h in hd) / 1000.0
    base_peak, base_p95 = float(demand_agg.max()), float(np.percentile(demand_agg, 95))
    print(f"      baseline: peak {base_peak:.2f} kW  P95 {base_p95:.2f} kW")

    print(f"[2/3] {len(accepts)} accept levels × {len(seeds)} seeds ...")
    results = {}
    for acc in accepts:
        peaks, p95s = [], []
        for seed in seeds:
            _reseed(seed)
            r = rollout(hd, mode="coordinated", user_accept=acc, verbose=False,
                        ev_smart=True, ev_seed=seed)
            agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0) / 1000.0
            peaks.append(float(agg.max()))
            p95s.append(float(np.percentile(agg, 95)))
        key = f"accept={acc:.2f}"
        results[key] = {
            "peak_kw_mean": round(float(np.mean(peaks)), 2),
            "peak_kw_std":  round(float(np.std(peaks)), 3),
            "p95_kw_mean":  round(float(np.mean(p95s)), 2),
            "p95_kw_std":   round(float(np.std(p95s)), 3),
            "peak_red_pct_mean": round(100 * (base_peak - float(np.mean(peaks))) / base_peak, 2),
            "p95_red_pct_mean":  round(100 * (base_p95 - float(np.mean(p95s))) / base_p95, 2),
            "peaks": [round(p, 2) for p in peaks],
        }
        print(f"      {key}: peak {results[key]['peak_kw_mean']}±{results[key]['peak_kw_std']} kW"
              f"  P95 {results[key]['p95_kw_mean']}±{results[key]['p95_kw_std']} kW")

    return {"seeds": seeds, "days": days,
            "baseline": {"peak_kw": round(base_peak, 2), "p95_kw": round(base_p95, 2)},
            "coordinated": results}


def plot(res: dict, out_path) -> None:
    acc = [float(k.split("=")[1]) for k in res["coordinated"]]
    rows = list(res["coordinated"].values())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.errorbar(acc, [r["peak_kw_mean"] for r in rows],
                yerr=[r["peak_kw_std"] for r in rows],
                fmt="o-", lw=2, capsize=5, color="#1F3A5F",
                label=f"Peak (mean±std over {len(res['seeds'])} seeds)")
    ax.errorbar(acc, [r["p95_kw_mean"] for r in rows],
                yerr=[r["p95_kw_std"] for r in rows],
                fmt="s--", lw=2, capsize=5, color="#2A7F6F", label="P95")
    ax.axhline(res["baseline"]["peak_kw"], color="#9E4A32", lw=1.4,
               ls=(0, (5, 3)))
    ax.text(0.02, res["baseline"]["peak_kw"] + 0.4,
            f"No-DR peak {res['baseline']['peak_kw']} kW",
            fontsize=9.5, color="#9E4A32", style="italic")
    ax.set_xlabel("User accept rate"); ax.set_ylabel("kW")
    ax.set_title("Acceptance drives the peak — robust across seeds",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--accept-rates", nargs="+", type=float, default=ACCEPTS)
    args = ap.parse_args()

    res = run(args.days, args.seeds, args.accept_rates)

    print(f"[3/3] Saving ...")
    out = REPORTS / "multiseed_results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  saved {out}")
    fig_path = FIGS / "multiseed_accept.png"
    plot(res, fig_path)
    print(f"  saved {fig_path}")


if __name__ == "__main__":
    main()
