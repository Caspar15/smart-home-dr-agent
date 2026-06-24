"""Full pipeline runner — runs rollout (3 modes) + metrics + daily summary
for one sample day. Convenience script after train_all has finished.

Run:  python -m multi_household.experiments.run_full --days 60
"""
from __future__ import annotations
import argparse, sys, io, subprocess
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--accept", type=float, default=0.85)
    ap.add_argument("--skip-summary", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print(f" Multi-household DR Advisory — full run ({args.days} days)")
    print("=" * 70)

    print("\n[STEP 1] rollout — 3 modes")
    rc = subprocess.run([sys.executable, "-m", "multi_household.experiments.rollout",
                         "--days", str(args.days),
                         "--mode", "all",
                         "--user-accept", str(args.accept)],
                        check=False)
    if rc.returncode != 0:
        print("rollout failed"); return

    print("\n[STEP 2] metrics + figures")
    rc = subprocess.run([sys.executable, "-m", "multi_household.experiments.metrics"],
                        check=False)
    if rc.returncode != 0:
        print("metrics failed"); return

    if not args.skip_summary:
        print("\n[STEP 3] sample daily summary (House 9, day with recommendations)")
        rc = subprocess.run([sys.executable, "-m",
                             "multi_household.experiments.daily_summary",
                             "--house", "9", "--day", "11"],
                            check=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
