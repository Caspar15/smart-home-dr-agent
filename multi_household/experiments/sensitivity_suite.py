"""Cohort + training-seed sensitivity suite (paper appendix tables).

Three questions reviewers will ask, answered on the SAME window / seed /
frozen threshold (17.7 kW — calibrated once on the primary cohort and reused
unchanged, per protocol):

1. cohort_16_with_H3   — does excluding solar-affected House 3 drive the
                         result? (15 + H3)
2. cohort_17_all_nonsolar — does the H12/H19 (low-flexibility) exclusion
                         inflate DR potential? (15 + H12 + H19)
3. training seeds      — does forecaster training randomness move the DR
                         numbers? (retrain all 15 with 2 extra base seeds;
                         the headline models are base seed 42)

Any cohort whose test region fails the v2 quality bar (alignment/contiguity
guards in compute_all_forecasts) is reported as FAILED with the reason —
honestly, instead of silently fabricating features across the gap.

Run:  python -m multi_household.experiments.sensitivity_suite
Writes: reports/multi_household/sensitivity_suite.json
"""
from __future__ import annotations
import sys, json, random, subprocess, time
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from pathlib import Path
import numpy as np
import torch

from multi_household.config import CLEAN_HOUSES
from multi_household.forecasting.per_house_lstm import train_one_house, MODEL_DIR
from multi_household.experiments.rollout import compute_all_forecasts, rollout, REPORTS

SEED = 42
ACCEPT = 0.85
DAYS = 14
REPRO = Path(__file__).resolve().parents[2]


def _reseed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _stats(r):
    agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0) / 1000.0
    return float(agg.max()), float(np.percentile(agg, 95))


def _model_is_current(h: int) -> bool:
    """A model is usable only if its checkpoint provenance matches the CURRENT
    pipeline (cache version + split). Mere file existence is not enough —
    stale pre-v2 checkpoints (different features/architecture era) linger in
    models/ and either fail to load or silently carry old-pipeline training."""
    p = MODEL_DIR / f"house_{h:02d}.pt"
    if not p.exists():
        return False
    try:
        import torch as _t
        from multi_household.config import CACHE_VERSION, SPLIT_AT
        ck = _t.load(p, map_location="cpu", weights_only=False)
        return (ck.get("cache_version") == CACHE_VERSION
                and ck.get("split_at") == SPLIT_AT)
    except Exception:
        return False


def run_cohort(tag: str, houses: list[int]) -> dict:
    print(f"\n=== cohort: {tag} ({len(houses)} houses) ===", flush=True)
    # (re)train any house whose model is missing OR from a stale pipeline
    for h in houses:
        if not _model_is_current(h):
            print(f"  training forecaster H{h} (missing/stale) ...", flush=True)
            train_one_house(h, lookback=48, epochs=30, train_seed=SEED,
                            verbose=False)
    try:
        hd = compute_all_forecasts(houses, n_test_steps=DAYS * 144)
    except RuntimeError as e:
        print(f"  ✗ FAILED quality bar: {e}")
        return {"tag": tag, "houses": houses, "status": "failed",
                "reason": str(e)}
    _reseed()
    rb = rollout(hd, mode="baseline", user_accept=ACCEPT, verbose=False)
    _reseed()
    rc = rollout(hd, mode="coordinated", user_accept=ACCEPT, verbose=False,
                 ev_smart=True)
    bp, b95 = _stats(rb)
    cp, c95 = _stats(rc)
    out = {"tag": tag, "houses": houses, "status": "ok",
           "baseline": {"peak_kw": round(bp, 2), "p95_kw": round(b95, 2)},
           "coordinated": {"peak_kw": round(cp, 2), "p95_kw": round(c95, 2)},
           "peak_red_pct": round(100 * (bp - cp) / bp, 2),
           "p95_red_pct": round(100 * (b95 - c95) / b95, 2)}
    print(f"  base {bp:.2f}/{b95:.2f} → coord {cp:.2f}/{c95:.2f} kW "
          f"(peak −{out['peak_red_pct']}%, P95 −{out['p95_red_pct']}%)")
    return out


def run_training_seed(base_seed: int) -> dict:
    """Retrain all 15 forecasters with a different base seed in an isolated
    model dir, then evaluate the standard coordinated rollout."""
    tag = f"train_seed_{base_seed}"
    mdir = REPRO / "multi_household" / f"models_ts{base_seed}"
    print(f"\n=== {tag} (models → {mdir.name}) ===", flush=True)
    env = dict(**__import__("os").environ, MH_MODEL_DIR=str(mdir),
               PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m",
                        "multi_household.experiments.train_all",
                        "--epochs", "30", "--lookback", "48",
                        "--train-seed", str(base_seed)],
                       env=env, cwd=REPRO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-800:]); print(r.stderr[-800:])
        return {"tag": tag, "status": "failed", "reason": "training failed"}
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)
    # evaluate in a SUBPROCESS so MH_MODEL_DIR binds at import time
    code = (
        "import random, numpy as np, torch, json;"
        "from multi_household.config import CLEAN_HOUSES;"
        "from multi_household.experiments.rollout import compute_all_forecasts, rollout;"
        f"hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps={DAYS*144});"
        f"random.seed({SEED}); np.random.seed({SEED}); torch.manual_seed({SEED});"
        f"r = rollout(hd, mode='coordinated', user_accept={ACCEPT}, verbose=False, ev_smart=True);"
        "agg = np.stack([r['served_w'][h] for h in r['houses']]).sum(0)/1000.0;"
        "print('RESULT', round(float(agg.max()),2), round(float(np.percentile(agg,95)),2))"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPRO,
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    if r.returncode != 0 or not line:
        print(r.stdout[-800:]); print(r.stderr[-800:])
        return {"tag": tag, "status": "failed", "reason": "rollout failed"}
    peak, p95 = (float(x) for x in line[0].split()[1:3])
    print(f"  coordinated peak {peak:.2f} kW  P95 {p95:.2f} kW")
    return {"tag": tag, "status": "ok",
            "coordinated": {"peak_kw": peak, "p95_kw": p95}}


def main():
    out = {"seed": SEED, "accept": ACCEPT, "days": DAYS, "cohorts": [], "training_seeds": []}

    out["cohorts"].append(run_cohort("cohort_15_primary", CLEAN_HOUSES))
    out["cohorts"].append(run_cohort("cohort_16_with_H3",
                                     sorted(CLEAN_HOUSES + [3])))
    out["cohorts"].append(run_cohort("cohort_17_all_nonsolar",
                                     sorted(CLEAN_HOUSES + [12, 19])))

    # headline models = base seed 42 (row given for completeness); add 2 more
    out["training_seeds"].append({"tag": "train_seed_42 (headline)",
                                  "status": "ok",
                                  "coordinated": {"peak_kw": None,
                                                  "p95_kw": None,
                                                  "note": "see metrics_summary"}})
    for s in (7, 123):
        out["training_seeds"].append(run_training_seed(s))

    p = REPORTS / "sensitivity_suite.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
