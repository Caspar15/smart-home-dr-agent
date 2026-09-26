"""What sets the aggregate peak, and how stable is it.

Why this exists
---------------
The reference run's peak (32.41 kW) turned out to be decided by a single
night: two vehicles that plugged in at the same time both rejected their
proposals and landed on a third that had already been placed. Across ten
acceptance seeds the peak is bimodal — most seeds sit near the full-acceptance
value, a few jump when several rejections coincide on one night — while the
95th percentile barely moves. The manuscript reports this, so it needs an
artifact behind it rather than a notebook observation.

Also records the peak-to-average ratio (PAR), the metric the demand-response
scheduling literature most often uses, so the paper can speak that language.
With energy conserved the mean is unchanged across arms, so PAR moves with the
peak; it adds comparability, not a new signal.

Run:    python -m multi_household.experiments.peak_analysis
Writes: reports/multi_household/peak_analysis.json
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.experiments.rollout import REPORTS
from multi_household.data.preprocess import prepare_house
from multi_household.aggregator.ev_coordinator import advisory_ev_schedule, EV_ACCEPT_SEED

T = 2016
SPLIT_KW = 30.0          # separates the two modes of the ten-seed peak


def _load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def par_by_arm() -> dict:
    out = {}
    for arm in ("baseline", "independent", "coordinated"):
        z = np.load(REPORTS / f"rollout_{arm}.npz", allow_pickle=True)
        s = z["served"].sum(0) / 1000.0
        out[arm] = {"peak_kw": round(float(s.max()), 2),
                    "mean_kw": round(float(s.mean()), 2),
                    "par": round(float(s.max() / s.mean()), 2),
                    "load_factor": round(float(s.mean() / s.max()), 3)}
    raw = {}
    for arm in ("baseline", "coordinated"):
        z = np.load(REPORTS / f"rollout_{arm}.npz", allow_pickle=True)
        s = z["served"].sum(0)
        raw[arm] = s.max() / s.mean()
    # from the unrounded ratios; the rounded 3.90 / 3.14 would give 19.49%
    out["par_reduction_pct"] = round(float(100 * (raw["baseline"] - raw["coordinated"])
                                           / raw["baseline"]), 2)
    return out


def reference_night() -> dict:
    """Which vehicles were charging at the coordinated peak, and who rejected."""
    z = np.load(REPORTS / "rollout_coordinated.npz", allow_pickle=True)
    ts = pd.DatetimeIndex(pd.to_datetime(z["timestamps"]))
    houses = [int(h) for h in z["houses"]]
    tot = z["served"].sum(0)
    k = int(np.argmax(tot))
    recs = _load("rollout_coordinated_recs.json")
    ev = [r for r in recs if r.get("event_type") == "ev_advisory"]
    night = ts[k].date() if ts[k].hour >= 12 else (ts[k] - pd.Timedelta(days=1)).date()
    on_night = [r for r in ev if ts[r["timestep"]].date() == night]
    heavy = [h for h, v in zip(houses, z["served"][:, k]) if v > 6000]
    return {
        "peak_kw": round(float(tot[k] / 1000), 2),
        "peak_time": str(ts[k])[:16],
        "night": str(night),
        "houses_above_6kw_at_peak": heavy,
        "proposals_that_night": [
            {"house": int(r["house_id"]),
             "plug_in": str(ts[r["timestep"]])[11:16],
             "accepted": bool(r["accepted"])} for r in sorted(on_night, key=lambda r: r["timestep"])],
        "ev_proposals_total": len(ev),
        "ev_proposals_rejected": sum(1 for r in ev if not r["accepted"]),
    }


def seed_mechanism() -> list:
    """EV coordination alone, per acceptance seed: the peak and how many
    vehicles rejected on the night it falls on."""
    z = np.load(REPORTS / "rollout_baseline.npz", allow_pickle=True)
    houses = [int(h) for h in z["houses"]]
    ts = pd.DatetimeIndex(pd.to_datetime(z["timestamps"]))[:T]
    base, ev = {}, {}
    for i, h in enumerate(houses):
        b = prepare_house(h, inject_ev=False)["test_df"]["aggregate_w"].to_numpy(float)[:T]
        base[h] = b
        e = np.clip(z["demand"][i][:T] - b, 0, None)
        e = np.where(e > 1000, e, 0.0).astype(np.float32)       # the 7 kW blocks
        if e.sum() > 0:
            ev[h] = e
    BASE = sum(base.values())
    rows = []
    for sd in list(range(41, 51)) + [EV_ACCEPT_SEED]:
        oa, sa, dec = advisory_ev_schedule(ev, ts, accept_rate=0.85, seed=sd, strategy="edf")
        tot = BASE.copy()
        for h, v in ev.items():
            tot = tot + v - oa[h] + sa[h]
        k = int(np.argmax(tot))
        night = ts[k].date() if ts[k].hour >= 12 else (ts[k] - pd.Timedelta(days=1)).date()
        rej = [d for d in dec if not d["accepted"] and d["night"] == str(night)]
        # A rejection only matters if its charging window overlaps others. Count
        # vehicles charging at once, and how many of them are rejected blocks
        # left at their natural time.
        charging = np.zeros(T, dtype=int)
        rejected_now = np.zeros(T, dtype=int)
        for h, v in ev.items():
            live = (v - oa[h] + sa[h]) > 1000
            charging += live
            rejected_now += live & (oa[h] == 0) & (v > 1000)
        rows.append({"seed": sd,
                     "peak_kw": round(float(tot[k] / 1000), 2),
                     "p95_kw": round(float(np.percentile(tot, 95) / 1000), 2),
                     "peak_night": str(night),
                     "rejections_on_peak_night": len(rej),
                     "peak_time": str(ts[k])[5:16],
                     "base_kw_at_peak": round(float(BASE[k] / 1000), 2),
                     "ev_kw_at_peak": round(float((tot[k] - BASE[k]) / 1000), 2),
                     "max_vehicles_charging_at_once": int(charging.max()),
                     "vehicles_charging_at_peak": int(charging[k]),
                     "of_which_rejected_at_peak": int(rejected_now[k])})
    return rows


def main() -> None:
    ms = _load("multiseed_results.json")["coordinated"]
    mpc = _load("mpc_ladder.json")
    peaks = ms["accept=0.85"]["peaks"]
    low = [p for p in peaks if p < SPLIT_KW]
    high = [p for p in peaks if p >= SPLIT_KW]
    base_pk = mpc["no_dr"]["peak_kw"]
    span = base_pk - mpc["mpc_bound"]["peak_kw"]

    def share(pk):
        return round(100 * (base_pk - pk) / span, 1)

    def red(pk):
        return round(100 * (base_pk - pk) / base_pk, 1)

    out = {
        "note": ("peak = single 10-min maximum; it is bimodal across acceptance "
                 "seeds because it depends on whether rejections coincide on one "
                 "night. P95 summarises the top 5% of slots and is stable."),
        "par": par_by_arm(),
        "ten_seed_peak": {
            "seeds": ms.get("seeds", list(range(41, 51))) if isinstance(ms, dict) else None,
            "n_low_mode": len(low), "low_min_kw": min(low), "low_max_kw": max(low),
            "low_reduction_pct_range": [red(max(low)), red(min(low))],
            "n_high_mode": len(high), "high_values_kw": high,
            "high_reduction_pct_range": [red(max(high)), red(min(high))] if high else None,
            "p95_mean_kw": ms["accept=0.85"]["p95_kw_mean"],
            "p95_sd_kw": ms["accept=0.85"]["p95_kw_std"],
            "p95_sd_pct_of_mean": round(100 * ms["accept=0.85"]["p95_kw_std"]
                                        / ms["accept=0.85"]["p95_kw_mean"], 1),
            "peak_sd_pct_of_mean": round(100 * ms["accept=0.85"]["peak_kw_std"]
                                         / ms["accept=0.85"]["peak_kw_mean"], 1),
        },
        "bound_share_pct": {
            "full_acceptance": share(mpc["rule_100"]["peak_kw"]),
            "low_mode": [share(max(low)), share(min(low))],
            "ten_seed_mean": share(ms["accept=0.85"]["peak_kw_mean"]),
            "reference_run": share(mpc["rule_85"]["peak_kw"]),
        },
        "reference_night": reference_night(),
        "ev_only_by_seed": seed_mechanism(),
    }
    dst = REPORTS / "peak_analysis.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p = out["par"]
    print(f"PAR: baseline {p['baseline']['par']}  coordinated {p['coordinated']['par']}  "
          f"(-{p['par_reduction_pct']}%)")
    t = out["ten_seed_peak"]
    print(f"ten seeds @0.85: {t['n_low_mode']} in {t['low_min_kw']}-{t['low_max_kw']} kW, "
          f"{t['n_high_mode']} at {t['high_values_kw']};  P95 sd {t['p95_sd_pct_of_mean']}% "
          f"vs peak sd {t['peak_sd_pct_of_mean']}%")
    print(f"share of bound: {out['bound_share_pct']}")
    r = out["reference_night"]
    print(f"reference peak {r['peak_kw']} kW at {r['peak_time']}: {r['proposals_that_night']}")
    for row in out["ev_only_by_seed"]:
        print(f"  seed {row['seed']:>8}: peak {row['peak_kw']:5.2f} at {row['peak_time']} "
              f"= base {row['base_kw_at_peak']:5.2f} + EV {row['ev_kw_at_peak']:5.2f}  | rejected that night "
              f"{row['rejections_on_peak_night']}  | max EVs at once "
              f"{row['max_vehicles_charging_at_once']}  at peak {row['vehicles_charging_at_peak']}"
              f" ({row['of_which_rejected_at_peak']} rejected)")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
