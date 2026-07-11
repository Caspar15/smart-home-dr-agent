"""Cross-season robustness — replicate the headline on autumn + winter windows.

Reviewer question this answers: "your test window is ~14 days in early summer —
what about other seasons?" REFIT has two more clean 14-day test windows (scan
2026-07-07, all 16 houses):

    autumn: test 2014-10-01 .. 10-15  (min-house coverage 100%, max gap 0)
    winter: test 2015-02-14 .. 02-28  (min-house coverage 100%, max gap 0)

Method per window (same system, same hyper-parameters, same seed):
  1. train the 16 per-house CNN-LSTMs on the ~60 days BEFORE the test window
     (fixed-timestamp split MH_SPLIT_AT — the clean test region stays
     cross-house aligned even though training months have per-house gaps);
  2. roll out baseline vs coordinated (advisory EV, accept 0.85, seed 42);
  3. report peak / P95 / energy per season next to the spring headline.

Seasonal models live in their own folders (multi_household/models_<tag>/) and
never touch the headline forecasters or the headline rollout outputs.

Run:  python -m multi_household.experiments.season_windows            # both
      python -m multi_household.experiments.season_windows --tag autumn
Writes: reports/multi_household/season/<tag>.json + season_summary.json
"""
from __future__ import annotations
import sys, os, json, argparse, subprocess, time
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from pathlib import Path

REPRO = Path(__file__).resolve().parents[2]          # reproduction/
SEASON_DIR = REPRO / "reports" / "multi_household" / "season"

WINDOWS = {
    # tag: (clean_window_start, clean_window_end, split_at)
    # NOON-ANCHORED test windows: EVs plug in 21:00-23:50, so a window that
    # ends at midnight leaves the FINAL night's EVs with no room to stagger
    # inside the horizon (a finite-window artifact that showed up as a fake
    # 39 kW tail peak). Ending the window at 12:00 keeps every overnight
    # trough fully inside the horizon — every night is treatable.
    "autumn": ("2014-08-02", "2014-10-15 12:00", "2014-10-01 12:00"),
    "winter": ("2014-12-16", "2015-02-28 12:00", "2015-02-14 12:00"),
    # true spring — completes all four seasons (the headline test window
    # Jun-30..Jul-14 is SUMMER; its training months are spring)
    "spring": ("2015-02-06", "2015-04-21 12:00", "2015-04-07 12:00"),
}
SEED = 42
ACCEPT = 0.85
DAYS = 14


def _env_for(tag: str) -> dict:
    w = WINDOWS[tag]
    env = os.environ.copy()
    env["MH_CLEAN_WINDOW"] = f"{w[0]},{w[1]}"
    env["MH_SPLIT_AT"] = w[2]
    env["MH_MODEL_DIR"] = str(REPRO / "multi_household" / f"models_{tag}")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------- runner ----
def runner(tag: str) -> None:
    """Executed IN A SUBPROCESS with the window env set (config binds the
    window at import time, so this must not run in the orchestrator process)."""
    import random
    import numpy as np
    import torch
    from multi_household.config import CLEAN_HOUSES, CLEAN_WINDOW, SPLIT_AT
    from multi_household.experiments.rollout import compute_all_forecasts, rollout

    print(f"[runner:{tag}] window={CLEAN_WINDOW} split_at={SPLIT_AT}")
    hd = compute_all_forecasts(CLEAN_HOUSES, n_test_steps=DAYS * 144)

    def agg(r):
        s = np.stack([r["served_w"][h] for h in r["houses"]]).sum(0) / 1000.0
        d = np.stack([r["demand_w"][h] for h in r["houses"]]).sum(0) / 1000.0
        return s, d

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rb = rollout(hd, mode="baseline", user_accept=ACCEPT, verbose=False)
    sb, db = agg(rb)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rc = rollout(hd, mode="coordinated", user_accept=ACCEPT, verbose=False,
                 ev_smart=True, ev_seed=SEED)
    sc, dc = agg(rc)

    out = {
        "tag": tag,
        "window": list(CLEAN_WINDOW), "split_at": SPLIT_AT,
        "test_slots": int(len(sc)),
        "baseline": {"peak_kw": round(float(sb.max()), 2),
                     "p95_kw": round(float(np.percentile(sb, 95)), 2)},
        "coordinated": {"peak_kw": round(float(sc.max()), 2),
                        "p95_kw": round(float(np.percentile(sc, 95)), 2)},
        "peak_red_pct": round(100 * (sb.max() - sc.max()) / sb.max(), 1),
        "p95_red_pct": round(100 * (np.percentile(sb, 95) - np.percentile(sc, 95))
                             / np.percentile(sb, 95), 1),
        "energy_drift_pct": round(100 * abs(sc.sum() - dc.sum()) / dc.sum(), 3),
    }
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    (SEASON_DIR / f"{tag}.json").write_text(json.dumps(out, indent=2),
                                            encoding="utf-8")
    print(f"[runner:{tag}] peak {out['baseline']['peak_kw']} -> "
          f"{out['coordinated']['peak_kw']} kW ({out['peak_red_pct']}%)  "
          f"P95 {out['baseline']['p95_kw']} -> {out['coordinated']['p95_kw']} "
          f"({out['p95_red_pct']}%)  drift {out['energy_drift_pct']}%")


# ----------------------------------------------------------- orchestrator ----
def orchestrate(tags: list[str], epochs: int, lookback: int) -> None:
    for tag in tags:
        env = _env_for(tag)
        print(f"\n=== [{tag}] 1/2 train 16 forecasters "
              f"(window {WINDOWS[tag][0]}..{WINDOWS[tag][1]}) ===")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m",
                            "multi_household.experiments.train_all",
                            "--epochs", str(epochs), "--lookback", str(lookback)],
                           env=env, cwd=REPRO)
        if r.returncode != 0:
            print(f"[{tag}] training FAILED"); continue
        print(f"[{tag}] trained in {time.time()-t0:.0f}s")

        print(f"=== [{tag}] 2/2 rollout on the {DAYS}-day test window ===")
        r = subprocess.run([sys.executable, "-m",
                            "multi_household.experiments.season_windows",
                            "--runner", "--tag", tag], env=env, cwd=REPRO)
        if r.returncode != 0:
            print(f"[{tag}] runner FAILED")

    # summary table incl. the spring headline
    rows = []
    headline = {"tag": "summer (headline)", "window": ["2014-04-30", "2014-07-14"],
                "baseline": {"peak_kw": 40.50, "p95_kw": 27.99},
                "coordinated": {"peak_kw": 32.74, "p95_kw": 19.99},
                "peak_red_pct": 19.2, "p95_red_pct": 28.6}
    rows.append(headline)
    for tag in WINDOWS:
        p = SEASON_DIR / f"{tag}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    (SEASON_DIR / "season_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== Cross-season summary (same system, same params, seed 42) ===")
    print(f"{'season':<20}{'No-DR peak':>11}{'coord peak':>11}{'peak%':>8}"
          f"{'P95%':>8}")
    for r in rows:
        print(f"{r['tag']:<20}{r['baseline']['peak_kw']:>11.2f}"
              f"{r['coordinated']['peak_kw']:>11.2f}"
              f"{r['peak_red_pct']:>+8.1f}{r['p95_red_pct']:>+8.1f}")
    print(f"saved {SEASON_DIR / 'season_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(WINDOWS), default=None,
                    help="one window only (default: both)")
    ap.add_argument("--runner", action="store_true",
                    help="internal: run the in-window evaluation (env preset)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lookback", type=int, default=48)
    args = ap.parse_args()

    if args.runner:
        runner(args.tag)
    else:
        orchestrate([args.tag] if args.tag else list(WINDOWS),
                    args.epochs, args.lookback)


if __name__ == "__main__":
    main()
