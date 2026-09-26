"""Re-score an archived LLM run against the current one, same scorer for both.

Why this exists
---------------
The manuscript needs to say something about how stable the interface metrics
are over time. The obvious way to do that — quote the number each run's own
harness printed — is wrong, because the scorer changed: an earlier version
ignored numeric literals carried inside string facts, so a message that
correctly named `washing_machine_2` was scored as inventing the number 2.
Comparing two figures produced by two different scorers measures the scorer.

This loads an archived run out of git, re-scores it and the current run with
today's `llm_contract` checks, and reports both. It also counts how many of
the generated messages actually differ, which is the thing the server version
can be blamed for.

Run:    python -m multi_household.experiments.llm_version_compare
        [--baseline-ref HEAD] [--model llama3.1:8b]
Writes: reports/multi_household/llm_version_compare.json
"""
from __future__ import annotations
import sys, json, argparse, subprocess

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.experiments.rollout import REPORTS
from multi_household.experiments.llm_contract import check_row

CHECKS = ("schema_ok", "numbers_grounded", "citations_grounded",
          "citation_coverage", "units_ok", "currency_scale_ok",
          "time_explicit", "wording_is_proposal")


def score(rows: list) -> dict:
    res = [check_row(r) for r in rows]
    out = {"n_messages": len(res), "per_check": {}}
    for k in CHECKS:
        vals = [r[k] for r in res if r.get(k) is not None]
        if vals:
            out["per_check"][k] = {"passed": sum(vals), "applicable": len(vals),
                                   "rate": round(sum(vals) / len(vals), 4)}
    allp = sum(all(v for v in r.values() if v is not None) for r in res)
    out["all_checks_passed"] = allp
    out["all_checks_pass_rate"] = round(allp / len(res), 4) if res else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--baseline-ref", default="HEAD",
                    help="git ref holding the archived run to compare against")
    args = ap.parse_args()

    safe = args.model.replace(":", "_").replace("/", "_")
    rel = f"reports/multi_household/llm_eval_{safe}.json"
    cur_path = REPORTS / f"llm_eval_{safe}.json"
    if not cur_path.exists():
        print(f"missing {cur_path} — run llm_eval first")
        return
    current = json.loads(cur_path.read_text(encoding="utf-8"))

    repro = REPORTS.parents[1]
    raw = subprocess.run(["git", "show", f"{args.baseline_ref}:{rel}"],
                         cwd=repro, capture_output=True, encoding="utf-8")
    if raw.returncode != 0 or not raw.stdout.strip():
        print(f"no archived run at {args.baseline_ref}:{rel} — nothing to compare")
        return
    baseline = json.loads(raw.stdout)

    base_rows, cur_rows = baseline.get("rows") or [], current.get("rows") or []
    n = min(len(base_rows), len(cur_rows))
    differing = sum(1 for a, b in zip(base_rows, cur_rows)
                    if a.get("message_zh") != b.get("message_zh"))

    b_scored, c_scored = score(base_rows), score(cur_rows)
    moved = {}
    for k in CHECKS:
        bk, ck = b_scored["per_check"].get(k), c_scored["per_check"].get(k)
        if bk and ck:
            moved[k] = round(ck["rate"] - bk["rate"], 4)

    out = {
        "note": ("both runs re-scored with the CURRENT llm_contract checks, so "
                 "the comparison reflects the generated text and not a change "
                 "of scorer; figures each run's own harness printed are not "
                 "comparable across scorer versions"),
        "baseline_ref": args.baseline_ref,
        "model": args.model,
        "baseline_server": baseline.get("server_env"),
        "current_server": current.get("server_env"),
        "messages_compared": n,
        "messages_with_different_text": differing,
        "messages_with_different_text_pct": round(100 * differing / n, 1) if n else None,
        "baseline": b_scored,
        "current": c_scored,
        "rate_change_current_minus_baseline": moved,
    }
    dst = REPORTS / "llm_version_compare.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model={args.model}   baseline={args.baseline_ref}")
    print(f"  messages compared        {n}")
    print(f"  differing message text   {differing} ({out['messages_with_different_text_pct']}%)")
    print(f"  {'check':<22}{'archived':>10}{'current':>10}{'delta':>9}")
    for k in CHECKS:
        bk, ck = b_scored["per_check"].get(k), c_scored["per_check"].get(k)
        if bk and ck:
            print(f"  {k:<22}{bk['rate']:>10}{ck['rate']:>10}{moved[k]:>+9}")
    print(f"  {'ALL CHECKS':<22}{b_scored['all_checks_pass_rate']:>10}"
          f"{c_scored['all_checks_pass_rate']:>10}"
          f"{c_scored['all_checks_pass_rate'] - b_scored['all_checks_pass_rate']:>+9.4f}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
