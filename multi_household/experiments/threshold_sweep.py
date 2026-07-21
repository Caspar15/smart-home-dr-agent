"""Grid-threshold sensitivity — the curve promised by the threshold protocol.

The congestion trigger is FROZEN at the train-window p85 (17.7 kW). This sweep
shows the coordinated result does not hinge on that particular value: each row
re-runs the coordinated rollout with a different trigger (same window, seed,
accept rate, EV advisory). The legacy test-calibrated 18 kW is one row.

Run:  python -m multi_household.experiments.threshold_sweep --days 14
Writes: reports/multi_household/threshold_sweep.json
        figures/multi_household/threshold_sweep.png
"""
from __future__ import annotations
import sys, argparse, json, random
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_household.config import CLEAN_HOUSES, GRID_THRESHOLD_W
from multi_household.experiments.rollout import compute_all_forecasts, rollout, REPORTS
from multi_household.experiments.metrics import FIGS

SEED = 42
THRESHOLDS_W = [15000.0, 16000.0, 17000.0, GRID_THRESHOLD_W,
                18000.0, 19000.0, 20000.0]


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
    base_peak, base_p95 = dem.max() / 1000, np.percentile(dem, 95) / 1000
    print(f"      baseline peak {base_peak:.2f} / P95 {base_p95:.2f} kW")

    print(f"[2/2] Sweeping trigger {sorted(set(t/1000 for t in THRESHOLDS_W))} kW ...")
    rows = []
    for g in THRESHOLDS_W:
        _reseed()
        r = rollout(hd, mode="coordinated", user_accept=args.accept,
                    verbose=False, ev_smart=True, grid_threshold_w=g)
        agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0) / 1000.0
        ag = r["agent_state"]
        rows.append({
            "threshold_kw": round(g / 1000, 3),
            "is_frozen_protocol_value": abs(g - GRID_THRESHOLD_W) < 1e-6,
            "peak_kw": round(float(agg.max()), 2),
            "p95_kw": round(float(np.percentile(agg, 95)), 2),
            "peak_red_pct": round(100 * (base_peak - agg.max()) / base_peak, 2),
            "p95_red_pct": round(100 * (base_p95 - np.percentile(agg, 95)) / base_p95, 2),
            "appliance_decisions": int(sum(ag[h].n_recommendations for h in ag)),
        })
        print(f"      G={g/1000:5.2f} kW → peak {rows[-1]['peak_kw']} "
              f"P95 {rows[-1]['p95_kw']} (decisions {rows[-1]['appliance_decisions']})")

    out = {"seed": SEED, "accept": args.accept,
           "baseline": {"peak_kw": round(base_peak, 2), "p95_kw": round(base_p95, 2)},
           "rows": rows}
    p = REPORTS / "threshold_sweep.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {p}")

    xs = [r["threshold_kw"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(xs, [r["peak_kw"] for r in rows], "o-", color="#1F3A5F", label="Peak (kW)")
    ax.plot(xs, [r["p95_kw"] for r in rows], "s--", color="#2A7F6F", label="P95 (kW)")
    fx = GRID_THRESHOLD_W / 1000
    ax.axvline(fx, color="#9E4A32", ls=":", lw=1.4)
    ax.text(fx + 0.05, ax.get_ylim()[0] + 0.5, f"frozen {fx:.1f} kW\n(train p85)",
            fontsize=9, color="#9E4A32")
    ax.set_xlabel("Congestion trigger G (kW)")
    ax.set_ylabel("Coordinated result (kW)")
    ax.set_title("Threshold sensitivity — result does not hinge on the frozen trigger",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "threshold_sweep.png", dpi=150)
    print(f"saved {FIGS / 'threshold_sweep.png'}")


if __name__ == "__main__":
    main()
