"""Run ablations: forecast quality (LSTM vs persistence) and user accept rate.

Each ablation re-runs the rollout with one knob varied, summarizes a single
row of metrics, and saves a comparison table + plot.

Run:
   python -m multi_household.experiments.ablations --days 30
"""
from __future__ import annotations
import sys, io, argparse, json
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the canonical rollout (rollout.py uses --seed 42). We RE-SEED before
# every rollout so each ablation row is independently reproducible and the
# accept=0.85 row lines up exactly with the headline result.
SEED = 42


def _reseed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import (
    compute_all_forecasts, rollout, REPORTS,
)
from multi_household.experiments.metrics import (
    compute_metrics, REPORTS as METRIC_REPORTS, FIGS,
)


def _save_and_summarize(r: dict, mode: str, tag: str) -> dict:
    """Save .npz files using the same naming the metrics module expects, then
    compute metrics and return a summary row."""
    out_dir = REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"rollout_{mode}.npz"
    np.savez(npz,
             served=np.stack([r["served_w"][h] for h in r["houses"]]),
             demand=np.stack([r["demand_w"][h] for h in r["houses"]]),
             forecast=np.stack([r["forecast_w"][h] for h in r["houses"]]),
             timestamps=r["timestamps"].astype("datetime64[s]"),
             houses=np.array(r["houses"]))
    # wait_log
    wait_path = out_dir / f"rollout_{mode}_waitlog.json"
    wait_path.write_text(json.dumps(
        {int(h): r["agent_state"][h].wait_log for h in r["houses"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    # recs (so comfort metrics work)
    from dataclasses import asdict
    recs_path = out_dir / f"rollout_{mode}_recs.json"
    recs_path.write_text(json.dumps(
        [asdict(rr) for rr in r["recommendations"]],
        ensure_ascii=False, indent=2), encoding="utf-8")

    s, _ = compute_metrics(npz)
    return {
        "tag":           tag,
        "user_saving":   s["user"]["avg_saving_pct"],
        "p95_reduction": s["grid"]["p95_reduction_pct"],
        "peak_kw":       s["grid"]["agg_served_peak_kw"],
        "rebound_mean":  s["rebound"]["off_peak_rebound_mean_kw"],
        "rebound_p95":   s["rebound"]["off_peak_rebound_p95_kw"],
        "total_recs":    s["comfort"]["total_recommendations"],
        "fairness":      s["comfort"]["fairness_jain"],
        "defer_mean_min":s["comfort"]["defer_wait_mean_min"],
        "n_defers":      s["comfort"]["n_defers_completed"],
    }


def ablate_forecast(houses_data: dict, accept: float) -> list[dict]:
    """LSTM vs persistence forecast — compare aggregate-side DR effectiveness."""
    print("\n=== Ablation 1: forecast source (LSTM vs persistence) ===")
    rows = []
    for fm in ("lstm", "persistence"):
        print(f"\n  --- forecast = {fm} ---")
        _reseed()
        r = rollout(houses_data, mode="coordinated",
                    user_accept=accept, forecast_mode=fm, verbose=False)
        row = _save_and_summarize(r, "coordinated", f"forecast={fm}")
        rows.append(row)
        print(f"    saving={row['user_saving']:+.2f}%  "
              f"p95_red={row['p95_reduction']:+.2f}%  "
              f"rebound_p95={row['rebound_p95']:+.3f} kW")
    return rows


def ablate_accept_rate(houses_data: dict, rates: list[float]) -> list[dict]:
    """Sweep user accept rate — show system's sensitivity to user cooperation."""
    print("\n=== Ablation 2: user accept rate ===")
    rows = []
    for r_in in rates:
        print(f"\n  --- accept_rate = {r_in:.2f} ---")
        _reseed()
        r = rollout(houses_data, mode="coordinated",
                    user_accept=r_in, forecast_mode="lstm", verbose=False)
        row = _save_and_summarize(r, "coordinated", f"accept={r_in:.2f}")
        rows.append(row)
        print(f"    saving={row['user_saving']:+.2f}%  "
              f"p95_red={row['p95_reduction']:+.2f}%  "
              f"defer_mean={row['defer_mean_min']:.0f} min")
    return rows


def plot_accept_curve(rows: list[dict], out_path: Path):
    rates = [float(r["tag"].split("=")[1]) for r in rows]
    saving = [r["user_saving"] for r in rows]
    p95 = [r["p95_reduction"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(rates, saving, "o-", color="#124163", lw=2, label="User cost saving (%)")
    ax1.set_xlabel("User accept rate")
    ax1.set_ylabel("Cost saving (%)", color="#124163")
    ax2 = ax1.twinx()
    ax2.plot(rates, p95, "s--", color="#c47a3d", lw=2, label="Grid P95 削減 (%)")
    ax2.set_ylabel("Grid P95 削減 (%)", color="#c47a3d")
    ax1.set_title("Sensitivity to user accept rate (coordinated mode)",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forecast_compare(rows: list[dict], out_path: Path):
    labels = [r["tag"].split("=")[1] for r in rows]
    metrics_to_show = [("p95_reduction", "P95 削減 (%)", "#124163"),
                       ("rebound_p95",   "Rebound P95 (kW)", "#c47a3d"),
                       ("user_saving",   "Cost saving (%)", "#75BDA7")]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (k, lab, col) in enumerate(metrics_to_show):
        vals = [r[k] for r in rows]
        ax.bar(x + (i-1)*w, vals, w, color=col, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (i-1)*w, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Forecast source ablation — does LSTM beat persistence?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--accept-rates", nargs="+", type=float,
                    default=[0.0, 0.5, 0.85, 1.0])
    args = ap.parse_args()

    n_test_steps = args.days * 144

    print(f"\n[1/3] Loading data + running per-house LSTM ...")
    houses_data = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=n_test_steps)

    print(f"\n[2/3] Running ablations ...")
    forecast_rows = ablate_forecast(houses_data, accept=0.85)
    accept_rows   = ablate_accept_rate(houses_data, args.accept_rates)

    print(f"\n[3/3] Saving plots + tables ...")
    out_table = REPORTS / "ablation_results.json"
    out_table.write_text(json.dumps({
        "forecast":     forecast_rows,
        "accept_rate":  accept_rows,
    }, indent=2), encoding="utf-8")
    print(f"  saved {out_table}")

    plot_accept_curve(accept_rows, FIGS / "ablation_accept_rate.png")
    print(f"  saved {FIGS / 'ablation_accept_rate.png'}")

    plot_forecast_compare(forecast_rows, FIGS / "ablation_forecast.png")
    print(f"  saved {FIGS / 'ablation_forecast.png'}")

    print("\n=== Ablation tables ===")
    print(f"\nForecast source:")
    for r in forecast_rows:
        print(f"  {r['tag']:24s} saving={r['user_saving']:+.2f}%  "
              f"p95_red={r['p95_reduction']:+.2f}%  "
              f"peak={r['peak_kw']:.2f}kW  rebound_p95={r['rebound_p95']:+.3f}kW")
    print(f"\nAccept rate (coordinated):")
    for r in accept_rows:
        print(f"  {r['tag']:24s} saving={r['user_saving']:+.2f}%  "
              f"p95_red={r['p95_reduction']:+.2f}%  "
              f"recs={r['total_recs']}  defer_mean={r['defer_mean_min']:.0f}min")


if __name__ == "__main__":
    main()
