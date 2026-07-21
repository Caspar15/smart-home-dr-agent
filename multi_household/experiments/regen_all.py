"""ONE command → every formal artifact, from the current checkout, plus a
SHA256 hash manifest. This is the reproducibility contract for the paper:
all numbers in reports/ and figures/ must come from a single invocation of
this script on a clean commit — no mixed-batch artifacts, ever again.

Order (later steps read earlier steps' outputs):
    1. rollout --mode all + metrics        (headline npz/json/figures)
    2. mechanism_decomposition             (factorial)
    3. multiseed (10 seeds)                (error bars; raw peaks/p95s)
    4. ablations                           (forecast / accept / closed-loop)
    5. fairness_sweep                      (budget sweep)
    6. mpc_baseline                        (perfect-foresight bound)
    7. threshold_sweep                     (trigger sensitivity)
    8. ev_strategy_ladder                  (random / EDF / stagger, paired)
    9. sensitivity_suite + stats_summary   (cohorts, training seeds, CIs)
   10. llm_eval                            (needs Ollama; skipped if down)
   11. seasons (train + eval x4)           (--skip-seasons to omit)
   12. artifact_manifest.json              (SHA256 of every report/figure +
                                            git commit + dirty flag)

Run:  python -m multi_household.experiments.regen_all [--skip-seasons] [--skip-llm]
"""
from __future__ import annotations
import sys, json, time, hashlib, subprocess, argparse, urllib.request
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from pathlib import Path

REPRO = Path(__file__).resolve().parents[2]
REPORTS = REPRO / "reports" / "multi_household"
FIGS = REPRO / "figures" / "multi_household"

STEPS = [
    ("rollout",        ["-m", "multi_household.experiments.rollout",
                        "--days", "14", "--mode", "all", "--user-accept", "0.85"]),
    ("metrics",        ["-m", "multi_household.experiments.metrics"]),
    ("decomposition",  ["-m", "multi_household.experiments.mechanism_decomposition"]),
    ("multiseed",      ["-m", "multi_household.experiments.multiseed", "--days", "14",
                        "--seeds", "41", "42", "43", "44", "45",
                        "46", "47", "48", "49", "50"]),
    ("ablations",      ["-m", "multi_household.experiments.ablations", "--days", "14"]),
    ("fairness_sweep", ["-m", "multi_household.experiments.fairness_sweep", "--days", "14"]),
    ("mpc_baseline",   ["-m", "multi_household.experiments.mpc_baseline", "--days", "14"]),
    ("threshold_sweep",["-m", "multi_household.experiments.threshold_sweep", "--days", "14"]),
    ("ev_ladder",      ["-m", "multi_household.experiments.ev_strategy_ladder", "--days", "14"]),
    ("sensitivity",    ["-m", "multi_household.experiments.sensitivity_suite"]),
    ("stats",          ["-m", "multi_household.experiments.stats_summary"]),
]


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3):
            return True
    except Exception:
        return False


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPRO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seasons", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    dirty = bool(_git("status", "--porcelain"))
    commit = _git("rev-parse", "HEAD")
    print(f"regen_all @ {commit[:10]}{' (DIRTY TREE — not paper-grade)' if dirty else ''}")

    t_all = time.time()
    results = {}
    steps = list(STEPS)
    if not args.skip_llm:
        if _ollama_up():
            steps.append(("llm_eval", ["-m", "multi_household.experiments.llm_eval"]))
        else:
            print("  (Ollama down — llm_eval skipped; rerun it separately)")
            results["llm_eval"] = "skipped: ollama down"
    if not args.skip_seasons:
        steps.append(("seasons", ["-m", "multi_household.experiments.season_windows",
                                  "--epochs", "30", "--lookback", "48"]))

    for name, argv in steps:
        t0 = time.time()
        print(f"\n=== [{name}] ===", flush=True)
        r = subprocess.run([sys.executable, *argv], cwd=REPRO)
        results[name] = ("ok" if r.returncode == 0 else f"FAILED rc={r.returncode}")
        print(f"=== [{name}] {results[name]} ({time.time()-t0:.0f}s) ===")
        if r.returncode != 0:
            print("stopping — fix the failing step, then rerun.")
            break

    # SHA256 manifest: outputs (deterministic vs volatile) + INPUTS
    # (model checkpoints, data caches) so a checker can verify the whole
    # provenance chain, not just the results.
    # Idempotency contract: rerunning on the same commit must reproduce every
    # hash in `outputs` bit-for-bit (all sampling is seeded); `volatile`
    # entries (wall-clock timestamps, per-call latencies, LLM generations)
    # are hashed for the record but expected to differ between runs.
    VOLATILE = ("run_manifest.json", "llm_eval_")
    manifest = {"commit": commit, "dirty_tree": dirty,
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_s": round(time.time() - t_all, 1),
                "steps": results,
                "inputs": {}, "outputs": {}, "volatile": {}}

    def _h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for root in (REPRO / "multi_household" / "models",
                 REPRO / "multi_household" / "cache"):
        if root.exists():
            for p in sorted(root.glob("*")):
                if p.is_file():
                    rel = str(p.relative_to(REPRO)).replace("\\", "/")
                    manifest["inputs"][rel] = _h(p)
    for root in (REPORTS, FIGS):
        for p in sorted(root.rglob("*")):
            if p.is_file() and "legacy" not in p.parts[-2]:
                rel = str(p.relative_to(REPRO)).replace("\\", "/")
                bucket = ("volatile" if any(v in p.name for v in VOLATILE)
                          else "outputs")
                manifest[bucket][rel] = _h(p)
    dst = REPORTS / "artifact_manifest.json"
    dst.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nsaved {dst}  (inputs {len(manifest['inputs'])}, "
          f"outputs {len(manifest['outputs'])}, "
          f"volatile {len(manifest['volatile'])})")
    ok = all(v == "ok" for k, v in results.items() if not str(v).startswith("skipped"))
    print("ALL OK" if ok else "SOME STEPS FAILED — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
