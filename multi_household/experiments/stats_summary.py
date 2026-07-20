"""Statistical summary for the paper: 95% CIs + bootstrap difference CIs.

Reads the raw per-seed values from multiseed_results.json (peaks / p95s lists)
and produces, per accept level:
  • mean, sample std (ddof=1), and t-based 95% CI for peak and P95;
  • bootstrap (10k resamples) 95% CI for the REDUCTION vs the deterministic
    No-DR baseline (baseline has no seed variance, so the difference
    distribution is just the seed distribution shifted).
Also folds in the training-seed and cohort sensitivity rows from
sensitivity_suite.json when present.

Run:  python -m multi_household.experiments.stats_summary
Writes: reports/multi_household/stats_summary.json
"""
from __future__ import annotations
import sys, json
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
from scipy import stats as sstats

from multi_household.experiments.rollout import REPORTS

N_BOOT = 10_000
RNG = np.random.default_rng(20260719)


def t_ci(x: np.ndarray) -> tuple[float, float, float, float]:
    """mean, sample std, and t-based 95% CI half-width bounds."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and s > 0:
        h = float(sstats.t.ppf(0.975, n - 1) * s / np.sqrt(n))
    else:
        h = 0.0
    return m, s, m - h, m + h


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ms = json.loads((REPORTS / "multiseed_results.json").read_text())
    base_peak = ms["baseline"]["peak_kw"]
    base_p95 = ms["baseline"]["p95_kw"]
    n_seeds = len(ms["seeds"])

    out = {"n_seeds": n_seeds, "seeds": ms["seeds"],
           "baseline": ms["baseline"], "accept_levels": {}}
    print(f"seeds n={n_seeds}; baseline peak {base_peak} / P95 {base_p95} kW")
    for key, row in ms["coordinated"].items():
        peaks = np.array(row["peaks"], float)
        p95s = np.array(row.get("p95s", []), float)
        pm, psd, plo, phi = t_ci(peaks)
        blo, bhi = boot_ci(base_peak - peaks)
        entry = {
            "peak_kw": {"mean": round(pm, 2), "std": round(psd, 3),
                        "ci95": [round(plo, 2), round(phi, 2)]},
            "peak_reduction_kw_boot_ci95": [round(blo, 2), round(bhi, 2)],
        }
        if len(p95s):
            qm, qsd, qlo, qhi = t_ci(p95s)
            qblo, qbhi = boot_ci(base_p95 - p95s)
            entry["p95_kw"] = {"mean": round(qm, 2), "std": round(qsd, 3),
                               "ci95": [round(qlo, 2), round(qhi, 2)]}
            entry["p95_reduction_kw_boot_ci95"] = [round(qblo, 2), round(qbhi, 2)]
        out["accept_levels"][key] = entry
        print(f"  {key}: peak {pm:.2f}±{psd:.2f} CI[{plo:.2f},{phi:.2f}] | "
              f"Δpeak boot CI[{blo:.2f},{bhi:.2f}] kW"
              + (f" | P95 {entry['p95_kw']['mean']:.2f} "
                 f"CI{entry['p95_kw']['ci95']}" if len(p95s) else ""))

    sp = REPORTS / "sensitivity_suite.json"
    if sp.exists():
        out["sensitivity"] = json.loads(sp.read_text(encoding="utf-8"))
        print("sensitivity_suite.json folded in")

    dst = REPORTS / "stats_summary.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
