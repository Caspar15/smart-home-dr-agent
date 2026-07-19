"""Compute user-facing + grid-facing metrics from saved rollouts.

Inputs are .npz files saved by rollout.py (one per mode). Outputs:
  • reports/multi_household/metrics_summary.json
  • figures/multi_household/agg_loadcurve.png         (3-day aggregate curve)
  • figures/multi_household/rebound_peak.png          (histogram)
  • figures/multi_household/per_house_savings.png     (bar)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_household.config import (
    PEAK_HOURS_LOCAL, OFFPEAK_HOURS,
    PEAK_PRICE_GBP, MID_PRICE_GBP, OFFPEAK_PRICE_GBP,
)

REPORTS = Path(__file__).resolve().parents[2] / "reports" / "multi_household"
FIGS    = Path(__file__).resolve().parents[2] / "figures" / "multi_household"
FIGS.mkdir(parents=True, exist_ok=True)


def hour_of(timestamps: np.ndarray) -> np.ndarray:
    return pd.to_datetime(timestamps).hour.values


def tou_price_vec(hours: np.ndarray) -> np.ndarray:
    p = np.full_like(hours, MID_PRICE_GBP, dtype=float)
    p[(hours >= OFFPEAK_HOURS[0]) & (hours < OFFPEAK_HOURS[1])] = OFFPEAK_PRICE_GBP
    p[(hours >= PEAK_HOURS_LOCAL[0]) & (hours < PEAK_HOURS_LOCAL[1])] = PEAK_PRICE_GBP
    return p


def cost_gbp(load_w: np.ndarray, hours: np.ndarray) -> float:
    """Compute electricity cost in GBP given Watts per 10-min step and
    fixed-ToU prices."""
    energy_kwh = load_w * (10.0 / 60.0) / 1000.0       # W → kWh per step
    return float((energy_kwh * tou_price_vec(hours)).sum())


def _load_recs(mode: str, base: Path = REPORTS) -> list[dict]:
    path = base / f"rollout_{mode}_recs.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _jain_index(values) -> float | None:
    """Jain's fairness index: 1 = perfectly equal; 1/N = totally concentrated.
    Returns None (report as n/a) when there is nothing to distribute — a
    zero-event mode is NOT 'perfectly fair', it is undefined."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or (v == 0).all():
        return None
    num = v.sum() ** 2
    den = (len(v) * (v ** 2).sum())
    return float(num / den) if den > 0 else None


def compute_metrics(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    served   = data["served"]          # (H, T)
    demand   = data["demand"]          # (H, T)
    forecast = data["forecast"]
    ts       = pd.to_datetime(data["timestamps"].astype(str))
    houses   = data["houses"]
    H, T     = served.shape
    hours    = ts.hour.values
    mode     = npz_path.stem.replace("rollout_", "")

    # Aggregate (sum across houses)
    served_agg = served.sum(axis=0)
    demand_agg = demand.sum(axis=0)

    # Per-house cost
    per_house = []
    for i, h in enumerate(houses):
        c_dem = cost_gbp(demand[i], hours)
        c_ser = cost_gbp(served[i], hours)
        per_house.append({
            "house": int(h),
            "demand_kwh":    float(demand[i].sum() * (10/60)/1000.0),
            "served_kwh":    float(served[i].sum() * (10/60)/1000.0),
            "cost_baseline_gbp": round(c_dem, 2),
            "cost_after_gbp":    round(c_ser, 2),
            "saving_pct":   round(100*(c_dem - c_ser)/c_dem if c_dem > 0 else 0, 2),
        })

    # Grid-level
    p95_dem = float(np.percentile(demand_agg, 95))
    p95_ser = float(np.percentile(served_agg, 95))
    peak_dem = float(demand_agg.max())
    peak_ser = float(served_agg.max())

    # Rebound: max served during off-peak window
    is_off = (hours >= OFFPEAK_HOURS[0]) & (hours < OFFPEAK_HOURS[1])
    off_peak_max_dem = float(demand_agg[is_off].max()) if is_off.any() else 0.0
    off_peak_max_ser = float(served_agg[is_off].max()) if is_off.any() else 0.0

    # Peak window mean (17-22)
    is_peak = (hours >= PEAK_HOURS_LOCAL[0]) & (hours < PEAK_HOURS_LOCAL[1])
    peak_win_mean_dem = float(demand_agg[is_peak].mean()) if is_peak.any() else 0.0
    peak_win_mean_ser = float(served_agg[is_peak].mean()) if is_peak.any() else 0.0

    # ---------------- Release-chunk wait metrics --------------------------
    # NOTE semantics: each wait-log entry is one banked ENERGY CHUNK (one
    # 10-min slice of a deferred cycle) from banking to release. This is an
    # engineering buffer-residency metric, NOT "how long a user waited for
    # their appliance cycle" — do not present it as user wait time.
    wait_path = npz_path.parent / f"rollout_{mode}_waitlog.json"
    chunk_waits = []
    if wait_path.exists():
        try:
            wait_logs = json.loads(wait_path.read_text(encoding="utf-8"))
            for h_str, log in wait_logs.items():
                for entry in log:
                    if isinstance(entry, list) and len(entry) >= 1:
                        chunk_waits.append(entry[0])      # wait_steps
        except Exception:
            pass

    if chunk_waits:
        wait_arr = np.array(chunk_waits, dtype=float)
        chunk_wait_mean_min = float(wait_arr.mean() * 10.0)
        chunk_wait_p95_min  = float(np.percentile(wait_arr, 95) * 10.0)
        chunk_wait_max_min  = float(wait_arr.max() * 10.0)
    else:
        chunk_wait_mean_min = chunk_wait_p95_min = chunk_wait_max_min = 0.0

    # ---------------- Comfort + fairness metrics (DECISION-LEVEL) ---------
    # Only typed decision events count: new_defer / declined_defer (appliance)
    # and ev_advisory (nightly EV reschedule). continuation / force_release
    # are trace messages — counting them used to inflate every number here
    # (445 "recommendations" that were really 48 appliance decisions).
    recs = _load_recs(mode, base=npz_path.parent)
    surface_total = len(recs)
    apl_dec = [r for r in recs
               if r.get("event_type") in ("new_defer", "declined_defer")]
    ev_dec = [r for r in recs if r.get("event_type") == "ev_advisory"]

    def _per_house_counts(events):
        cnt = {int(h): 0 for h in houses}
        acc = {int(h): 0 for h in houses}
        for r in events:
            hh = int(r.get("house_id", -1))
            if hh in cnt:
                cnt[hh] += 1
                if r.get("accepted"):
                    acc[hh] += 1
        return cnt, acc

    apl_cnt, apl_acc = _per_house_counts(apl_dec)
    ev_cnt,  ev_acc  = _per_house_counts(ev_dec)
    tot_cnt = {h: apl_cnt[h] + ev_cnt[h] for h in apl_cnt}

    n_apl, n_apl_acc = sum(apl_cnt.values()), sum(apl_acc.values())
    n_ev,  n_ev_acc  = sum(ev_cnt.values()),  sum(ev_acc.values())
    n_tot, n_tot_acc = n_apl + n_ev, n_apl_acc + n_ev_acc

    # Four fairness scopes (Jain), each answering a different question:
    #   appliance      — how evenly appliance recommendations spread (all houses)
    #   ev             — how evenly EV advisories spread (EV owners only)
    #   total          — all decision burden over all houses
    #   total_active   — conditional on having ≥1 decision this window
    ev_owner_counts   = ([ev_cnt[h] for h in ev_cnt if ev_cnt[h] > 0]
                         if n_ev else [])
    active_counts     = [c for c in tot_cnt.values() if c > 0]
    fairness = {
        "jain_appliance":      _jain_index(list(apl_cnt.values())),
        "jain_ev":             _jain_index(ev_owner_counts),
        "jain_total":          _jain_index(list(tot_cnt.values())),
        "jain_total_active":   _jain_index(active_counts),
    }
    tot_counts_arr = np.array(list(tot_cnt.values()))
    max_dec_per_house = int(tot_counts_arr.max()) if len(tot_counts_arr) else 0
    avg_dec_per_house = float(tot_counts_arr.mean()) if len(tot_counts_arr) else 0.0

    # ---------------- Rebound distribution --------------------------------
    is_off = (hours >= OFFPEAK_HOURS[0]) & (hours < OFFPEAK_HOURS[1])
    # Per-step rebound relative to demand at the SAME hour-of-day distribution
    rebound_per_step_kw = (served_agg - demand_agg) / 1000.0
    off_peak_rebound = rebound_per_step_kw[is_off]
    rebound_mean_kw = float(off_peak_rebound.mean()) if is_off.any() else 0.0
    rebound_p95_kw  = float(np.percentile(off_peak_rebound, 95)) if is_off.any() else 0.0
    rebound_max_kw  = float(off_peak_rebound.max()) if is_off.any() else 0.0

    total_base_gbp  = sum(p["cost_baseline_gbp"] for p in per_house)
    total_after_gbp = sum(p["cost_after_gbp"]    for p in per_house)
    # Billing note: costs use the FIXED ToU tariff only — the coordinated
    # surcharge is a control signal and is NOT billed. Both saving conventions
    # are reported: household-mean (unweighted) and total-bill (weighted).
    user_block = {
        "total_cost_baseline_gbp":  round(total_base_gbp, 2),
        "total_cost_after_gbp":     round(total_after_gbp, 2),
        "mean_household_saving_pct": round(np.mean([p["saving_pct"] for p in per_house]), 2),
        "weighted_total_saving_pct": round(100*(total_base_gbp - total_after_gbp)/total_base_gbp
                                           if total_base_gbp > 0 else 0, 2),
        "tariff_note": "fixed ToU only; dynamic surcharge not billed",
        "per_house":                per_house,
    }

    if n_tot == 0:
        comfort_block = None            # no decision events (e.g. baseline) → n/a
    else:
        comfort_block = {
            "appliance_decisions":       n_apl,
            "appliance_accepted":        n_apl_acc,
            "appliance_accept_rate":     round(n_apl_acc/n_apl, 3) if n_apl else None,
            "ev_decisions":              n_ev,
            "ev_accepted":               n_ev_acc,
            "ev_accept_rate":            round(n_ev_acc/n_ev, 3) if n_ev else None,
            "total_decisions":           n_tot,
            "total_accepted":            n_tot_acc,
            "total_accept_rate":         round(n_tot_acc/n_tot, 3),
            "surface_messages_total":    surface_total,   # incl. trace events
            "avg_decisions_per_house":   round(avg_dec_per_house, 2),
            "max_decisions_per_house":   max_dec_per_house,
            "per_house_decisions":       {str(h): tot_cnt[h] for h in sorted(tot_cnt)},
            "fairness":                  {k: (round(v, 4) if v is not None else None)
                                          for k, v in fairness.items()},
            "release_chunk_wait_mean_min": round(chunk_wait_mean_min, 1),
            "release_chunk_wait_p95_min":  round(chunk_wait_p95_min, 1),
            "release_chunk_wait_max_min":  round(chunk_wait_max_min, 1),
            "n_release_chunks":            len(chunk_waits),
        }

    summary = {
        "n_houses":  int(H),
        "n_steps":   int(T),
        "days":      round(T/144, 3),
        "user": user_block,
        "comfort": comfort_block,
        "rebound": {
            "off_peak_rebound_mean_kw":      round(rebound_mean_kw, 3),
            "off_peak_rebound_p95_kw":       round(rebound_p95_kw, 3),
            "off_peak_rebound_max_kw":       round(rebound_max_kw, 3),
        },
        "grid": {
            "agg_demand_p95_kw":          round(p95_dem/1000, 2),
            "agg_served_p95_kw":          round(p95_ser/1000, 2),
            "p95_reduction_pct":          round(100*(p95_dem - p95_ser)/p95_dem if p95_dem > 0 else 0, 2),
            "agg_demand_peak_kw":         round(peak_dem/1000, 2),
            "agg_served_peak_kw":         round(peak_ser/1000, 2),
            "peak_window_mean_demand_kw": round(peak_win_mean_dem/1000, 2),
            "peak_window_mean_served_kw": round(peak_win_mean_ser/1000, 2),
            "peak_window_reduction_pct":  round(100*(peak_win_mean_dem - peak_win_mean_ser)/peak_win_mean_dem if peak_win_mean_dem > 0 else 0, 2),
            "off_peak_max_demand_kw":     round(off_peak_max_dem/1000, 2),
            "off_peak_max_served_kw":     round(off_peak_max_ser/1000, 2),
            "rebound_increase_kw":        round((off_peak_max_ser - off_peak_max_dem)/1000, 2),
        },
        "energy_conservation": {
            "demand_total_mwh":  round(demand_agg.sum()*(10/60)/1e6, 3),
            "served_total_mwh":  round(served_agg.sum()*(10/60)/1e6, 3),
            "diff_pct":          round(100*(served_agg.sum() - demand_agg.sum())/demand_agg.sum(), 3),
        },
    }
    return summary, (ts, served, demand, served_agg, demand_agg, hours)


def plot_aggregate_curve(ts, demand_agg, served_agg_dict, out_path: Path,
                         days: int = 3):
    """3-day aggregate load curve overlay (baseline vs each DR mode)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(len(ts), days * 144)
    ax.plot(ts[:n], demand_agg[:n]/1000, color="#9AA7AD", lw=1.0,
            label="Demand (no DR)")
    colors = {"independent": "#75BDA7", "coordinated": "#124163"}
    for mode, arr in served_agg_dict.items():
        ax.plot(ts[:n], arr[:n]/1000, color=colors.get(mode, "#888"), lw=1.2,
                label=f"Served ({mode})")
    ax.set_title(f"Aggregate household load — first {days} days",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Aggregate load (kW)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hour_of_day_profile(ts, demand_agg, served_agg_dict, out_path: Path):
    """Mean load by hour-of-day for each mode — shows where DR shifts energy."""
    hours = ts.hour.values
    fig, ax = plt.subplots(figsize=(10, 4.5))
    by_hour_dem = np.array([demand_agg[hours == h].mean()/1000 if (hours == h).any() else 0
                            for h in range(24)])
    ax.plot(range(24), by_hour_dem, "o-", color="#9AA7AD", lw=2,
            label="Demand (no DR)")
    colors = {"independent": "#75BDA7", "coordinated": "#124163"}
    for mode, arr in served_agg_dict.items():
        by_hour = np.array([arr[hours == h].mean()/1000 if (hours == h).any() else 0
                            for h in range(24)])
        ax.plot(range(24), by_hour, "o-", color=colors.get(mode, "#888"), lw=2,
                label=f"Served ({mode})")
    # Mark peak / off-peak windows
    ax.axvspan(17, 22, color="#c47a3d", alpha=0.10, label="Peak ToU (17-22)")
    ax.axvspan( 0,  6, color="#2d6e6e", alpha=0.10, label="Off-peak ToU (0-6)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean aggregate load (kW)")
    ax.set_title("Mean load by hour-of-day — see where energy gets shifted",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_house_savings(summary_by_mode: dict, out_path: Path):
    """Bar of per-house savings under coordinated mode."""
    rows = summary_by_mode["coordinated"]["user"]["per_house"]
    houses = [r["house"] for r in rows]
    savings = [r["saving_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([f"H{h:02d}" for h in houses], savings, color="#124163")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cost saving (%)")
    ax.set_title("Per-household cost saving — coordinated mode",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rebound_check(metrics_by_mode: dict, out_path: Path):
    """DR-induced rebound only — uses MEAN and P95 rebound (not max which is
    swamped by natural REFIT data spikes). Bar chart with peak-window reduction
    on one side, rebound on the other.
    """
    modes = [m for m, s in metrics_by_mode.items()
             if m != "baseline" and isinstance(s, dict) and "grid" in s]
    if not modes or "baseline" not in metrics_by_mode:
        return
    base_pk  = metrics_by_mode["baseline"]["grid"]["peak_window_mean_served_kw"]

    peak_window_reduction = [base_pk - metrics_by_mode[m]["grid"]["peak_window_mean_served_kw"]
                              for m in modes]
    rebound_mean = [metrics_by_mode[m]["rebound"]["off_peak_rebound_mean_kw"]
                    for m in modes]
    rebound_p95  = [metrics_by_mode[m]["rebound"]["off_peak_rebound_p95_kw"]
                    for m in modes]

    x = np.arange(len(modes)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - w, peak_window_reduction, w, color="#124163",
                   label="Peak-window mean shaved (kW, ↑ good)")
    bars2 = ax.bar(x,     rebound_mean, w, color="#75BDA7",
                   label="Off-peak rebound, mean (kW, ↓ good)")
    bars3 = ax.bar(x + w, rebound_p95,  w, color="#c47a3d",
                   label="Off-peak rebound, P95 (kW, ↓ good)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("kW (relative to no-DR baseline)")
    ax.set_title("DR shaving vs rebound — coordinated should shave more, rebound less",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bs, vs in [(bars1, peak_window_reduction),
                   (bars2, rebound_mean),
                   (bars3, rebound_p95)]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width()/2,
                    v + (0.02 if v >= 0 else -0.05),
                    f"{v:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily_timeline(metrics_by_mode: dict, ts, out_path: Path):
    """Per-day timeline: recommendations issued + peak-window mean served vs
    demand baseline. Shows where DR actually helped."""
    days = []
    seen_days = set()
    for ts_val in ts:
        d = ts_val.date()
        if d not in seen_days:
            seen_days.add(d); days.append(d)
    n_days = len(days)
    if n_days < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"independent": "#75BDA7", "coordinated": "#124163"}
    for mode in ("independent", "coordinated"):
        if mode not in metrics_by_mode: continue
        recs = _load_recs(mode)
        per_day = [0] * n_days
        for r in recs:
            if r.get("event_type") not in ("new_defer", "declined_defer",
                                           "ev_advisory"):
                continue                     # decisions only, not trace msgs
            d_idx = r["timestep"] // 144
            if d_idx < n_days:
                per_day[d_idx] += 1
        ax.bar(range(n_days), per_day, color=colors.get(mode, "#888"), alpha=0.7,
               label=f"{mode} decisions/day")
    ax.set_xlabel("Day index")
    ax.set_ylabel("User decisions per day")
    ax.set_title("DR activity over the rollout window",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fairness(metrics_by_mode: dict, out_path: Path):
    """Per-house DECISION burden (appliance + EV, stacked) for coordinated
    mode, with the four Jain scopes in the title."""
    if "coordinated" not in metrics_by_mode:
        return
    s = metrics_by_mode["coordinated"]
    if not s.get("comfort"):
        return
    rows = s["user"]["per_house"]
    houses = [r["house"] for r in rows]
    from collections import Counter
    recs = _load_recs("coordinated")
    apl = Counter(r["house_id"] for r in recs
                  if r.get("event_type") in ("new_defer", "declined_defer"))
    ev = Counter(r["house_id"] for r in recs
                 if r.get("event_type") == "ev_advisory")
    apl_c = [apl.get(h, 0) for h in houses]
    ev_c  = [ev.get(h, 0) for h in houses]
    f4 = s["comfort"]["fairness"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [f"H{h:02d}" for h in houses]
    ax.bar(x, apl_c, color="#2d6e6e", label="Appliance decisions")
    ax.bar(x, ev_c, bottom=apl_c, color="#c47a3d", label="EV advisory decisions")
    ax.set_ylabel("User decisions received")
    ax.set_title(f"Per-household decision burden — coordinated "
                 f"(Jain appl={f4['jain_appliance']}, ev={f4['jain_ev']}, "
                 f"total={f4['jain_total']})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def data_quality_report() -> dict:
    """Per-house NaN disclosure on the common clean-window grid (reviewers ask).
    nan_rate = share of 10-min slots with no real aggregate reading BEFORE
    imputation; max_gap = longest consecutive missing run (steps of 10 min)."""
    from multi_household.config import CLEAN_HOUSES
    from multi_household.data.refit_loader import load_house
    from multi_household.data.preprocess import reindex_to_common_grid
    out = {}
    for h in CLEAN_HOUSES:
        df = reindex_to_common_grid(load_house(h))
        nan = df["aggregate_w"].isna().to_numpy()
        # longest consecutive NaN run
        max_gap = gap = 0
        for v in nan:
            gap = gap + 1 if v else 0
            max_gap = max(max_gap, gap)
        out[f"house_{h:02d}"] = {
            "n_slots":       int(len(df)),
            "nan_rate_pct":  round(100.0 * float(nan.mean()), 3),
            "max_gap_steps": int(max_gap),
            "max_gap_hours": round(max_gap / 6.0, 1),
        }
    rates = [v["nan_rate_pct"] for v in out.values()]
    out["_overall"] = {"mean_nan_rate_pct": round(float(np.mean(rates)), 3),
                       "worst_house_pct":   round(float(np.max(rates)), 3)}
    return out


def main():
    modes = ["baseline", "independent", "coordinated"]
    summaries = {}
    served_agg_by_mode = {}
    demand_agg_ref = None
    ts_ref = None

    for mode in modes:
        p = REPORTS / f"rollout_{mode}.npz"
        if not p.exists():
            print(f"missing {p}, skipping")
            continue
        s, (ts, served, demand, served_agg, demand_agg, hours) = compute_metrics(p)
        summaries[mode] = s
        served_agg_by_mode[mode] = served_agg
        if demand_agg_ref is None:
            demand_agg_ref = demand_agg
            ts_ref = ts

    # Data-quality disclosure (pre-imputation NaN per house on the clean window)
    try:
        summaries["data_quality"] = data_quality_report()
        dq = summaries["data_quality"]["_overall"]
        print(f"data quality: mean NaN {dq['mean_nan_rate_pct']}% "
              f"(worst house {dq['worst_house_pct']}%)")
    except Exception as e:                       # noqa: BLE001
        print(f"data-quality report skipped: {e}")

    out = REPORTS / "metrics_summary.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"saved {out}")

    print("\n=== SUMMARY ===")
    for mode, s in summaries.items():
        if mode == "data_quality":
            continue
        print(f"\n[{mode}]")
        print(f"  user.mean_household_saving  = {s['user']['mean_household_saving_pct']:.2f} %")
        print(f"  user.weighted_total_saving  = {s['user']['weighted_total_saving_pct']:.2f} %")
        print(f"  grid.p95_reduction_pct      = {s['grid']['p95_reduction_pct']:.2f} %")
        print(f"  grid.peak_window_reduce%    = {s['grid']['peak_window_reduction_pct']:.2f} %")
        print(f"  grid.peak_kw                = {s['grid']['agg_served_peak_kw']:.2f} kW  "
              f"(vs baseline {summaries['baseline']['grid']['agg_served_peak_kw']:.2f})"
              if 'baseline' in summaries else "")
        print(f"  rebound.mean_kw             = {s['rebound']['off_peak_rebound_mean_kw']:+.3f} kW")
        print(f"  rebound.p95_kw              = {s['rebound']['off_peak_rebound_p95_kw']:+.3f} kW")
        c = s.get("comfort")
        if c is None:
            print(f"  comfort                     = n/a (no decision events)")
        else:
            print(f"  comfort.appliance_decisions = {c['appliance_decisions']} "
                  f"(accept {c['appliance_accept_rate']})")
            print(f"  comfort.ev_decisions        = {c['ev_decisions']} "
                  f"(accept {c['ev_accept_rate']})")
            print(f"  comfort.total_decisions     = {c['total_decisions']} "
                  f"(surface msgs {c['surface_messages_total']})")
            f4 = c["fairness"]
            print(f"  comfort.jain a/ev/tot/act   = "
                  f"{f4['jain_appliance']} / {f4['jain_ev']} / "
                  f"{f4['jain_total']} / {f4['jain_total_active']}")
            print(f"  comfort.chunk_wait_mean     = {c['release_chunk_wait_mean_min']:.1f} min "
                  f"(buffer residency, not user wait)")
        print(f"  energy diff %               = {s['energy_conservation']['diff_pct']:+.3f} %")

    # Figures
    if ts_ref is not None:
        # Only overlay independent + coordinated against demand baseline
        overlay = {m: served_agg_by_mode[m] for m in ["independent", "coordinated"]
                   if m in served_agg_by_mode}
        plot_aggregate_curve(ts_ref, demand_agg_ref, overlay,
                             FIGS / "agg_loadcurve.png", days=3)
        print(f"saved {FIGS / 'agg_loadcurve.png'}")

        plot_hour_of_day_profile(ts_ref, demand_agg_ref, overlay,
                                 FIGS / "hour_of_day_profile.png")
        print(f"saved {FIGS / 'hour_of_day_profile.png'}")

    if "coordinated" in summaries:
        plot_per_house_savings(summaries, FIGS / "per_house_savings.png")
        print(f"saved {FIGS / 'per_house_savings.png'}")

    if len(summaries) >= 2:
        plot_rebound_check(summaries, FIGS / "rebound_peak.png")
        print(f"saved {FIGS / 'rebound_peak.png'}")

    if "coordinated" in summaries:
        plot_fairness(summaries, FIGS / "fairness.png")
        print(f"saved {FIGS / 'fairness.png'}")

    if ts_ref is not None and len(summaries) >= 2:
        plot_daily_timeline(summaries, ts_ref, FIGS / "daily_timeline.png")
        print(f"saved {FIGS / 'daily_timeline.png'}")


if __name__ == "__main__":
    main()
