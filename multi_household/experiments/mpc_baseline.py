"""MPC perfect-foresight bound — the controller-ladder upper line.

Multi-house extension of the conference single-house MPC (same buffer LP,
lifted to the 16-house aggregate, full-window perfect foresight):

    served[t] = demand[t] − defer[t] + release[t]
    0 ≤ defer[t] ≤ deferrable_w[t]      (actual deferrable watts that step,
                                         summed over houses, incl. the EVs)
    buffer ≥ 0, buffer[T−1] = 0          (energy conserved, all debt repaid)
    repayment: every deferred Wh is released within MAX_DEFER_STEPS (8 h,
               the most generous comfort cap → a valid relaxation)
    minimize M  s.t.  served[t] ≤ M      (pure min-peak)

Honest label: this is a *perfect-foresight LP relaxation* — blocks may be
split and repayment uses the loosest cap, so no realizable controller can
beat it. Our rule-based system is measured against this bound:

    No-DR | Rule (ours, 85% accept) | Rule (100%) | MPC bound

Run:  python -m multi_household.experiments.mpc_baseline --days 14
Writes: reports/multi_household/mpc_ladder.json
"""
from __future__ import annotations
import sys, argparse, json
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from multi_household.config import CLEAN_HOUSES
from multi_household.experiments.rollout import compute_all_forecasts, REPORTS

MAX_DEFER_STEPS = 48        # 8 h — the loosest comfort cap (EV); relaxation


def build_series(houses_data: dict):
    """Aggregate demand + per-step total deferrable watts (incl. EV)."""
    T = min(len(houses_data[h]["test_df"]) for h in houses_data)
    demand = np.zeros(T)
    deferrable = np.zeros(T)
    for h, hd in houses_data.items():
        df = hd["test_df"]
        demand += df["aggregate_w"].values[:T].astype(float)
        for col in hd["deferable_cols"]:
            deferrable += np.clip(df[col].values[:T].astype(float), 0, None)
    # deferrable can't exceed demand at any step (data glitches)
    deferrable = np.minimum(deferrable, demand)
    return demand, deferrable


def solve_mpc_bound(demand: np.ndarray, deferrable: np.ndarray,
                    max_defer_steps: int = MAX_DEFER_STEPS):
    """Full-window min-peak LP. Returns (served, peak_kW, diagnostics).

    O(T) sparse formulation: running totals cumD/cumR are STATE VARIABLES tied
    by recurrence equalities (a dense cumulative-triangle matrix at T≈2000 is
    O(T²) nonzeros and kills the solver).
    Variable layout: [defer(T), release(T), cumD(T), cumR(T), M]  → 4T+1 vars.
    """
    T = len(demand)
    iD, iR, icD, icR, iM = 0, T, 2 * T, 3 * T, 4 * T
    nv = 4 * T + 1
    K = max_defer_steps

    c = np.zeros(nv); c[iM] = 1.0                     # minimize M

    def coo(rows, cols, vals, shape):
        return sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsc()

    # -- equalities: cumD[0]=defer[0]; cumD[t]−cumD[t−1]−defer[t]=0 (same for R)
    er, ec, ev, beq = [], [], [], []
    r = 0
    for (base, cum) in ((iD, icD), (iR, icR)):
        er += [r, r]; ec += [cum, base]; ev += [1.0, -1.0]; beq.append(0.0); r += 1
        for t in range(1, T):
            er += [r, r, r]
            ec += [cum + t, cum + t - 1, base + t]
            ev += [1.0, -1.0, -1.0]
            beq.append(0.0); r += 1
    # terminal: cumD[T−1] − cumR[T−1] = 0 (all debt repaid)
    er += [r, r]; ec += [icD + T - 1, icR + T - 1]; ev += [1.0, -1.0]
    beq.append(0.0); r += 1
    A_eq, b_eq = coo(er, ec, ev, (r, nv)), np.array(beq)

    # -- inequalities
    ur, uc, uv, bub = [], [], [], []
    r = 0
    for t in range(T):        # peak: −defer[t] + release[t] − M ≤ −demand[t]
        ur += [r, r, r]; uc += [iD + t, iR + t, iM]; uv += [-1.0, 1.0, -1.0]
        bub.append(-demand[t]); r += 1
    for t in range(T):        # buffer ≥ 0: cumR[t] − cumD[t] ≤ 0
        ur += [r, r]; uc += [icR + t, icD + t]; uv += [1.0, -1.0]
        bub.append(0.0); r += 1
    for t in range(K, T):     # repayment: cumD[t−K] − cumR[t] ≤ 0
        ur += [r, r]; uc += [icD + t - K, icR + t]; uv += [1.0, -1.0]
        bub.append(0.0); r += 1
    A_ub, b_ub = coo(ur, uc, uv, (r, nv)), np.array(bub)

    bounds = ([(0.0, float(d)) for d in deferrable]        # defer
              + [(0.0, None)] * T                           # release
              + [(0.0, None)] * T + [(0.0, None)] * T       # cumD, cumR
              + [(0.0, None)])                              # M

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    defer, release = res.x[iD:iD + T], res.x[iR:iR + T]
    served = demand - defer + release
    diag = {
        "energy_in_wh":  float(defer.sum() / 6.0),
        "energy_out_wh": float(release.sum() / 6.0),
        "lp_status":     res.message,
    }
    return served, float(res.x[iM]) / 1000.0, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    print(f"[1/3] Loading data ({args.days} d) ...")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=args.days * 144)
    demand, deferrable = build_series(hd)
    print(f"      T={len(demand)} slots; deferrable share "
          f"{100 * deferrable.sum() / demand.sum():.1f}% of energy")

    print(f"[2/3] Solving full-window min-peak LP (perfect foresight) ...")
    served, peak_kw, diag = solve_mpc_bound(demand, deferrable)
    p95_kw = float(np.percentile(served, 95)) / 1000.0
    base_peak = float(demand.max()) / 1000.0
    base_p95 = float(np.percentile(demand, 95)) / 1000.0
    econs = abs(diag["energy_in_wh"] - diag["energy_out_wh"]) < 1e-3 * max(
        diag["energy_in_wh"], 1.0)

    # our rule numbers from the saved headline + multiseed (if present)
    rule85 = {"peak_kw": None, "p95_kw": None}
    try:
        c = np.load(REPORTS / "rollout_coordinated.npz", allow_pickle=True)
        agg = c["served"].sum(0) / 1000.0
        rule85 = {"peak_kw": round(float(agg.max()), 2),
                  "p95_kw": round(float(np.percentile(agg, 95)), 2)}
    except Exception:
        pass
    rule100 = None
    try:
        ms = json.loads((REPORTS / "multiseed_results.json").read_text())
        r = ms["coordinated"]["accept=1.00"]
        rule100 = {"peak_kw": r["peak_kw_mean"], "p95_kw": r["p95_kw_mean"]}
    except Exception:
        pass

    ladder = {
        "note": "MPC = perfect-foresight LP relaxation (blocks splittable, "
                "8h repayment) — no realizable controller can beat it.",
        "no_dr":        {"peak_kw": round(base_peak, 2), "p95_kw": round(base_p95, 2)},
        "rule_85":      rule85,
        "rule_100":     rule100,
        "mpc_bound":    {"peak_kw": round(peak_kw, 2), "p95_kw": round(p95_kw, 2)},
        "energy_conserved": bool(econs),
        "diag": {k: round(v, 1) if isinstance(v, float) else v
                 for k, v in diag.items()},
    }

    print(f"[3/3] Controller ladder (same window, same loads):")
    print(f"      {'No-DR':<22} peak {base_peak:6.2f} kW   P95 {base_p95:6.2f} kW")
    if rule85["peak_kw"]:
        print(f"      {'Rule (ours, 85%)':<22} peak {rule85['peak_kw']:6.2f} kW   "
              f"P95 {rule85['p95_kw']:6.2f} kW")
    if rule100:
        print(f"      {'Rule (100% accept)':<22} peak {rule100['peak_kw']:6.2f} kW   "
              f"P95 {rule100['p95_kw']:6.2f} kW")
    print(f"      {'MPC bound (perfect)':<22} peak {peak_kw:6.2f} kW   "
          f"P95 {p95_kw:6.2f} kW")
    print(f"      energy conserved: {econs}")

    out = REPORTS / "mpc_ladder.json"
    out.write_text(json.dumps(ladder, indent=2), encoding="utf-8")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
