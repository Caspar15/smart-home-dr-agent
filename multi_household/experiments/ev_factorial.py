"""EV factorial stress test — does the coordination result survive changes in
EV penetration, charger power, session length and arrival spread?

The reference scenario fixes ONE EV configuration (5 houses, 7 kW, 4 h,
arrivals clustered in 21:00-23:50). A reviewer can reasonably ask whether the
reported improvement is an artefact of that single, highly synchronised
configuration. This experiment varies all four factors:

    24 settings = 3 EV-house counts {3, 5, 10}
                x 2 charger powers  {3.6, 7.0} kW
                x 2 session lengths {2 h, 4 h}
                x 2 arrival spreads {clustered 21:00-23:50, dispersed 17:00-23:50}
    x 5 load-generation seeds (1001-1005)   -> which nights charge, plug-in times
    x 3 acceptance seeds       (41-43)      -> which proposals the user accepts
    = 360 PAIRED scenarios at p = 0.85.

PAIRED means: inside one scenario every strategy sees the SAME charging events
and the SAME accept draws (advisory_ev_schedule consumes one uniform per block
unconditionally); only the placement rule differs. Each strategy is scored
against `immediate` (charge on arrival, no DR) from the SAME scenario, so every
reported reduction is a within-scenario difference, never a cross-run one.

Strategies: immediate | random | stagger | edf (the promoted main method).
Placement calls the production coordinator (aggregator.ev_coordinator), not a
re-implementation, so the factorial characterises the shipped system.

The appliance controller and the learned forecaster are NOT used here: the
factorial isolates EV placement, and mechanism_decomposition.json already shows
the appliance rules move the aggregate peak by ~0.1%.

Invariants asserted on EVERY scenario (any violation aborts the run):
  * energy conservation    sum(served) == sum(immediate)  (<=1e-6 relative)
  * arrival feasibility    no block starts before it is plugged in
  * comfort cap            start delay <= 8 h (48 steps)
  * zero participation     p = 0 reproduces `immediate` exactly

Run:  python -m multi_household.experiments.ev_factorial
Writes: reports/multi_household/ev_factorial.json
        figures/multi_household/ev_factorial.png
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import itertools

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_household.config import CLEAN_HOUSES
from multi_household.data.preprocess import prepare_house
from multi_household.aggregator.ev_coordinator import advisory_ev_schedule
from multi_household.experiments.rollout import REPORTS
from multi_household.experiments.metrics import FIGS

# Deterministic EV-household ordering; the first N are equipped at a given
# penetration, so the 3-house set is a subset of the 5-house set, and the
# 5-house set is exactly the reference configuration.
EV_ORDER = [5, 7, 9, 13, 18, 2, 4, 8, 10, 15]

N_EV = [3, 5, 10]
POWER_W = [3600.0, 7000.0]
DUR_STEPS = [12, 24]                        # 2 h and 4 h at 10-min resolution
ARRIVALS = {"clustered": (21, 24),          # 21:00-23:50 (the reference)
            "dispersed": (17, 24)}          # 17:00-23:50
LOAD_SEEDS = [1001, 1002, 1003, 1004, 1005]
ACCEPT_SEEDS = [41, 42, 43]
NIGHT_PROB = 0.70
ACCEPT_P = 0.85
STRATEGIES = ["random", "stagger", "edf"]
MAX_DELAY_STEPS = 48                        # 8 h comfort cap


def load_base(n_steps):
    """Aggregate demand WITHOUT any synthetic EV, plus the shared timestamps."""
    base = None
    ts = None
    for h in CLEAN_HOUSES:
        d = prepare_house(h, inject_ev=False)["test_df"].iloc[:n_steps]
        v = d["aggregate_w"].to_numpy(dtype=float)
        base = v if base is None else base + v
        if ts is None:
            ts = pd.DatetimeIndex(pd.to_datetime(d["time"].to_numpy()))
    return base, ts


def gen_ev(ts, houses, power_w, dur, arrival, load_seed):
    """Synthetic charging events under the same generative rule as the
    reference set: each night charges with probability 0.70, plug-in uniform
    on the arrival window at 10-min granularity, constant power for `dur`."""
    T = len(ts)
    lo, hi = arrival
    ev = {h: np.zeros(T, dtype=np.float32) for h in houses}
    dates = sorted({t.date() for t in ts})
    for h in houses:
        rng = np.random.default_rng(load_seed * 1000 + h)
        for d in dates:
            if rng.random() > NIGHT_PROB:
                continue
            hour = lo + int(rng.random() * (hi - lo))
            minute = int(rng.random() * 6) * 10
            plug = pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=minute)
            i = int(ts.searchsorted(plug))
            if i >= T:
                continue
            ev[h][i:min(i + dur, T)] = power_w
    return ev


def served_for(base, ev, orig_app, shift_app):
    """served = base + natural EV - removed natural blocks + relocated blocks."""
    out = base.copy()
    for h, arr in ev.items():
        out = out + arr - orig_app[h] + shift_app[h]
    return out


def run_scenario(base, ts, houses, power_w, dur, arrival, load_seed, acc_seed):
    ev = gen_ev(ts, houses, power_w, dur, arrival, load_seed)
    immediate = base + sum(ev.values())
    imm_peak = float(immediate.max())
    imm_p95 = float(np.percentile(immediate, 95))
    row = {"immediate": {"peak": round(imm_peak, 1), "p95": round(imm_p95, 1)}}
    n_blocks = 0

    for strat in STRATEGIES:
        oa, sa, dec = advisory_ev_schedule(ev, ts, accept_rate=ACCEPT_P,
                                           seed=acc_seed, strategy=strat)
        s = served_for(base, ev, oa, sa)
        n_blocks = len(dec)

        # ---- invariants -------------------------------------------------
        assert abs(s.sum() - immediate.sum()) <= 1e-6 * immediate.sum(), \
            "energy not conserved (%s)" % strat
        for d in dec:
            if d["accepted"]:
                delay = d["new_start_idx"] - d["start_idx"]
                assert delay >= 0, "%s: charging before arrival" % strat
                assert delay <= MAX_DELAY_STEPS, \
                    "%s: start delay %d > 8 h" % (strat, delay)

        row[strat] = {
            "peak": round(float(s.max()), 1),
            "p95": round(float(np.percentile(s, 95)), 1),
            "peak_red_pct": round(100 * (imm_peak - float(s.max())) / imm_peak, 3),
            "p95_red_pct": round(
                100 * (imm_p95 - float(np.percentile(s, 95))) / imm_p95, 3),
        }

    # zero-participation invariance, checked on the promoted method
    oa0, sa0, _ = advisory_ev_schedule(ev, ts, accept_rate=0.0,
                                       seed=acc_seed, strategy="edf")
    assert np.allclose(served_for(base, ev, oa0, sa0), immediate), \
        "p=0 did not reproduce `immediate`"

    row["n_blocks"] = n_blocks
    return row


def _stat(rows, sel, strat, key):
    v = [r[strat][key] for r in rows if sel(r)]
    return (round(float(np.mean(v)), 2),
            round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0,
            round(float(np.min(v)), 2), round(float(np.max(v)), 2))


def summarise(rows):
    out = {"n_scenarios": len(rows), "accept_p": ACCEPT_P,
           "strategies": {}, "by_arrival": {}, "by_factor": {}}

    for strat in STRATEGIES:
        pm, ps, plo, phi = _stat(rows, lambda r: True, strat, "peak_red_pct")
        qm, qs, qlo, qhi = _stat(rows, lambda r: True, strat, "p95_red_pct")
        out["strategies"][strat] = {
            "peak_red_pct": {"mean": pm, "sd": ps, "min": plo, "max": phi},
            "p95_red_pct": {"mean": qm, "sd": qs, "min": qlo, "max": qhi},
            "n_peak_worse": sum(1 for r in rows if r[strat]["peak_red_pct"] < 0),
            "n_peak_unchanged": sum(1 for r in rows
                                    if abs(r[strat]["peak_red_pct"]) < 1e-9),
            "n_p95_worse": sum(1 for r in rows if r[strat]["p95_red_pct"] < 0),
        }

    for arr in ARRIVALS:
        def sel(r, a=arr):
            return r["arrival"] == a
        out["by_arrival"][arr] = {
            s: {"peak_red_pct_mean": _stat(rows, sel, s, "peak_red_pct")[0],
                "p95_red_pct_mean": _stat(rows, sel, s, "p95_red_pct")[0],
                "n_peak_worse": sum(1 for r in rows
                                    if sel(r) and r[s]["peak_red_pct"] < 0),
                "n": sum(1 for r in rows if sel(r))}
            for s in STRATEGIES}

    for fname, key, vals in (("n_ev", "n_ev", N_EV),
                             ("power_kw", "power_kw", [p / 1000 for p in POWER_W]),
                             ("dur_h", "dur_h", [d / 6 for d in DUR_STEPS])):
        out["by_factor"][fname] = {}
        for v in vals:
            def sel(r, k=key, vv=v):
                return r[k] == vv
            out["by_factor"][fname][str(v)] = {
                s: {"peak_red_pct_mean": _stat(rows, sel, s, "peak_red_pct")[0],
                    "p95_red_pct_mean": _stat(rows, sel, s, "p95_red_pct")[0],
                    "n_peak_worse": sum(1 for r in rows
                                        if sel(r) and r[s]["peak_red_pct"] < 0)}
                for s in STRATEGIES}
    return out


def plot(rows, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = {"random": "#8A95A1", "stagger": "#2A7F6F", "edf": "#1F3A5F"}
    marks = {"clustered": "o", "dispersed": "^"}
    for s in STRATEGIES:
        for arr in ARRIVALS:
            xs = [r[s]["peak_red_pct"] for r in rows if r["arrival"] == arr]
            ys = [r[s]["p95_red_pct"] for r in rows if r["arrival"] == arr]
            ax.scatter(xs, ys, s=15, alpha=0.55, color=colors[s],
                       marker=marks[arr], linewidths=0,
                       label="%s / %s" % (s, arr))
    ax.axvline(0, color="#9E4A32", lw=1.2, ls=(0, (5, 3)))
    # Annotate in axes coordinates near the top of the line: at the bottom it
    # ran under the legend and was clipped.
    ax.annotate("left of this line = peak got worse",
                xy=(0, 0.965), xycoords=("data", "axes fraction"),
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, color="#9E4A32",
                style="italic")
    ax.set_xlabel("Peak reduction vs immediate charging (%)")
    ax.set_ylabel("P95 reduction vs immediate charging (%)")
    ax.set_title("EV factorial: %d paired scenarios "
                 "(3 penetrations x 2 powers x 2 durations x 2 arrival spreads)"
                 % len(rows), fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.95,
              borderpad=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    n_steps = args.days * 144

    print("[1/3] Base demand without EV (%d houses) ..." % len(CLEAN_HOUSES))
    base, ts = load_base(n_steps)
    print("      T=%d   natural peak %.2f kW" % (len(base), base.max() / 1000))

    settings = list(itertools.product(N_EV, POWER_W, DUR_STEPS, sorted(ARRIVALS)))
    total = len(settings) * len(LOAD_SEEDS) * len(ACCEPT_SEEDS)
    print("[2/3] %d settings x %d load seeds x %d accept seeds = %d paired "
          "scenarios ..." % (len(settings), len(LOAD_SEEDS),
                             len(ACCEPT_SEEDS), total))

    rows = []
    for (n_ev, pw, dur, arr_name) in settings:
        houses = EV_ORDER[:n_ev]
        for ls in LOAD_SEEDS:
            for acs in ACCEPT_SEEDS:
                r = run_scenario(base, ts, houses, pw, dur,
                                 ARRIVALS[arr_name], ls, acs)
                r.update({"n_ev": n_ev, "power_kw": pw / 1000,
                          "dur_h": dur / 6, "arrival": arr_name,
                          "load_seed": ls, "accept_seed": acs})
                rows.append(r)
        print("      done: %2d EV | %.1f kW | %.0f h | %s"
              % (n_ev, pw / 1000, dur / 6, arr_name))

    summary = summarise(rows)
    out = {"note": ("paired within-scenario comparison against `immediate`; "
                    "placement calls the production advisory_ev_schedule; "
                    "no appliance controller and no learned forecaster"),
           "factors": {"n_ev": N_EV, "power_w": POWER_W,
                       "dur_steps": DUR_STEPS, "arrivals": ARRIVALS,
                       "load_seeds": LOAD_SEEDS, "accept_seeds": ACCEPT_SEEDS,
                       "night_prob": NIGHT_PROB, "accept_p": ACCEPT_P},
           "summary": summary, "rows": rows}

    p = REPORTS / "ev_factorial.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[3/3] saved %s" % p)

    for s in STRATEGIES:
        d = summary["strategies"][s]
        print("      %-8s P95 %+6.2f%% (sd %4.2f)   peak %+6.2f%%   "
              "peak worse in %d/%d"
              % (s, d["p95_red_pct"]["mean"], d["p95_red_pct"]["sd"],
                 d["peak_red_pct"]["mean"], d["n_peak_worse"], len(rows)))

    fig_path = FIGS / "ev_factorial.png"
    plot(rows, fig_path)
    print("      saved %s" % fig_path)


if __name__ == "__main__":
    main()
