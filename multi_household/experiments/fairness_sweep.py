"""Fairness constraint sweep — the Jain vs peak-shaving trade-off (R16).

Problem: recommendations concentrate on the houses that can actually shave
(Jain 0.34). Mechanism: a per-house DAILY recommendation budget — once a house
has received B new recommendations today, the agent stays quiet there until
tomorrow, spreading the burden across houses.

Scope note (honest): the budget governs the appliance-level agent
recommendations (what the Jain metric measures). The nightly EV advisory is a
separate coordinator channel and is not budgeted — one reschedule per EV per
night is already minimal.

Run:  python -m multi_household.experiments.fairness_sweep --days 14
Writes: reports/multi_household/fairness_sweep.json
        figures/multi_household/fairness_tradeoff.png
"""
from __future__ import annotations
import sys, argparse, json
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import compute_all_forecasts, rollout, REPORTS
from multi_household.experiments.ablations import _save_and_summarize, _reseed
from multi_household.experiments.metrics import FIGS

BUDGETS = [None, 6, 4, 2, 1]          # recs/house/day; None = unlimited (today)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--accept", type=float, default=0.85)
    args = ap.parse_args()

    print(f"[1/3] Loading data ({args.days} d) ...")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=args.days * 144)

    print(f"[2/3] Sweeping daily rec budget {BUDGETS} (accept={args.accept}) ...")
    rows = []
    for b in BUDGETS:
        tag = f"budget={'inf' if b is None else b}"
        _reseed()
        r = rollout(hd, mode="coordinated", user_accept=args.accept,
                    verbose=False, ev_smart=True, fairness_budget=b)
        row = _save_and_summarize(r, "coordinated", f"fairness_{tag}")
        row["tag"] = tag
        row["n_skipped_by_fairness"] = int(sum(
            r["agent_state"][h].n_skipped_by_fairness for h in r["houses"]))
        rows.append(row)
        print(f"    {tag:12s} jain={row['fairness']:.3f}  "
              f"p95_red={row['p95_reduction']:+.2f}%  peak={row['peak_kw']:.2f}kW  "
              f"recs={row['total_recs']}  skipped={row['n_skipped_by_fairness']}")

    print(f"[3/3] Saving ...")
    out = REPORTS / "fairness_sweep.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved {out}")

    # trade-off plot: Jain (x) vs P95 reduction (y), annotated by budget
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = [r["fairness"] for r in rows]
    ys = [r["p95_reduction"] for r in rows]
    ax.plot(xs, ys, "o-", lw=2, color="#1F3A5F")
    for r in rows:
        ax.annotate(r["tag"].replace("budget=", "B="),
                    (r["fairness"], r["p95_reduction"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("Jain fairness of recommendations (1 = perfectly even)")
    ax.set_ylabel("Grid P95 reduction (%)")
    ax.set_title("Fairness budget trade-off — spreading the burden vs shaving",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fairness_tradeoff.png", dpi=150)
    plt.close(fig)
    print(f"  saved {FIGS / 'fairness_tradeoff.png'}")


if __name__ == "__main__":
    main()
