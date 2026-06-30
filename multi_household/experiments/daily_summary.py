"""Build a daily summary for one household using the LLM advisor.

Run:  python -m multi_household.experiments.daily_summary --house 7 --day 1

Loads the saved coordinated rollout, extracts that house+day, computes
the day's facts (cost, recommendations, peak events), then calls Ollama
(local Llama 3.1, see advisor.MODEL) with schema-constrained output.
Saves Markdown + JSON.
"""
from __future__ import annotations
import sys, io, argparse, json, time
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
import pandas as pd

from multi_household.config import PEAK_HOURS_LOCAL, OFFPEAK_HOURS
from multi_household.experiments.metrics import (
    REPORTS, tou_price_vec, cost_gbp,
)
from multi_household.llm.advisor import call_ollama_summary
from multi_household.data.preprocess import prepare_house
from multi_household.forecasting.per_house_lstm import (
    load_forecaster, predict_24h_recursive,
)


def compute_daily_facts(house_id: int, day_index: int) -> dict:
    """Build the facts JSON for one house × one day from saved rollout."""
    data = np.load(REPORTS / "rollout_coordinated.npz", allow_pickle=True)
    houses = data["houses"]
    if house_id not in houses:
        raise ValueError(f"house {house_id} not in rollout")
    idx = int(np.where(houses == house_id)[0][0])

    served   = data["served"][idx]
    demand   = data["demand"][idx]
    ts       = pd.to_datetime(data["timestamps"].astype(str))
    hours    = ts.hour.values

    # Slice one day
    steps_per_day = 144
    s = day_index * steps_per_day
    e = s + steps_per_day
    if e > len(served):
        raise ValueError("day_index past end of rollout")

    d_served, d_demand, d_hours = served[s:e], demand[s:e], hours[s:e]
    d_ts = ts[s:e]

    cost_baseline = cost_gbp(d_demand, d_hours)
    cost_after    = cost_gbp(d_served, d_hours)

    # Recommendations for this house + day
    all_recs = json.loads((REPORTS / "rollout_coordinated_recs.json").read_text(encoding="utf-8"))
    day_recs = [r for r in all_recs
                if r["house_id"] == house_id
                and s <= r["timestep"] < e]

    # Peak-window mean
    is_peak = (d_hours >= PEAK_HOURS_LOCAL[0]) & (d_hours < PEAK_HOURS_LOCAL[1])
    pw_dem  = float(d_demand[is_peak].mean()) if is_peak.any() else 0.0
    pw_ser  = float(d_served[is_peak].mean()) if is_peak.any() else 0.0

    facts = {
        "household_id":             int(house_id),
        "date":                     str(d_ts[0].date()),
        "horizon_steps":            int(len(d_served)),
        "controller":               "rule-based + appliance-aware (multi-house)",
        "cost_baseline_gbp":        round(cost_baseline, 3),
        "cost_after_gbp":           round(cost_after, 3),
        "cost_saving_gbp":          round(cost_baseline - cost_after, 3),
        "cost_saving_pct":          round(100*(cost_baseline-cost_after)/cost_baseline
                                          if cost_baseline > 0 else 0, 2),
        "recommendations_total":    len(day_recs),
        "recommendations_accepted": sum(1 for r in day_recs if r.get("accepted")),
        "peak_window_mean_demand_w": round(pw_dem, 1),
        "peak_window_mean_served_w": round(pw_ser, 1),
        "peak_window_reduction_pct": round(100*(pw_dem-pw_ser)/pw_dem if pw_dem>0 else 0, 2),
    }

    # --- REAL 24-hour-ahead forecast (recursive, replaces fake "tomorrow") --
    try:
        full = prepare_house(house_id)
        full_test = full["test_df"]
        fcols     = full["feature_cols"]
        model, ckpt = load_forecaster(house_id)
        lookback = ckpt["lookback"]
        # use the last `lookback` steps of THIS day as the seed window
        end_idx_in_test = e   # absolute index in test_df
        seed_start = max(0, end_idx_in_test - lookback)
        seed_window = full_test[fcols].values[seed_start:end_idx_in_test]
        if len(seed_window) == lookback:
            # find lag column indices in the feature list
            lag_cols = [i for i, c in enumerate(fcols)
                        if c.startswith("aggregate_w_lag")]
            # Use first few (lag1, lag2, lag3) for recursion
            preds = predict_24h_recursive(
                model, ckpt, seed_window.astype("float32"),
                lag_col_indices=lag_cols[:3], horizon_steps=144,
            )
            peak_idx = int(preds.argmax())
            facts["tomorrow_forecast"] = {
                "source":          "LSTM recursive multi-step",
                "horizon_steps":   144,
                "peak_hour_idx":   peak_idx,
                "peak_value_w":    round(float(preds[peak_idx]), 1),
                "mean_w":          round(float(preds.mean()), 1),
                "min_w":           round(float(preds.min()), 1),
                "note":            "recursive — error compounds; treat as guidance",
            }
        else:
            facts["tomorrow_forecast"] = {
                "source": "insufficient history",
                "note":   "not enough lookback to forecast 24h ahead",
            }
    except Exception as ex:
        facts["tomorrow_forecast"] = {
            "source": "error",
            "note":   f"forecast failed: {ex.__class__.__name__}",
        }
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", type=int, default=7)
    ap.add_argument("--day",   type=int, default=1)
    ap.add_argument("--no-ollama", action="store_true",
                    help="skip the LLM call (offline test)")
    args = ap.parse_args()

    print(f"[1/3] Computing daily facts for House {args.house}, day {args.day} ...")
    facts = compute_daily_facts(args.house, args.day)
    print(json.dumps(facts, ensure_ascii=False, indent=2))

    if args.no_ollama:
        return

    from multi_household.llm.advisor import MODEL as LLM_MODEL
    print(f"\n[2/3] Calling Ollama ({LLM_MODEL}) for daily summary ...")
    t0 = time.time()
    summary = call_ollama_summary(facts)
    print(f"      elapsed {time.time()-t0:.1f}s")

    if summary is None:
        print("      !! Ollama unreachable or returned non-JSON. Skipping.")
        return

    print(f"      keys: {list(summary.keys())}")

    print(f"\n[3/3] Rendered summary")
    print("=" * 60)
    print(f"# 戶 {facts['household_id']} 每日能源回顧 ({facts['date']})")
    print(f"")
    print(f"## 今日數字")
    print(f"- 電費 (DR 前): £{facts['cost_baseline_gbp']:.2f}")
    print(f"- 電費 (DR 後): £{facts['cost_after_gbp']:.2f}")
    print(f"- 節省: £{facts['cost_saving_gbp']:.2f} ({facts['cost_saving_pct']}%)")
    print(f"- 收到推薦: {facts['recommendations_total']} 次")
    print(f"- 接受推薦: {facts['recommendations_accepted']} 次")
    print(f"- 尖峰時段平均負載降幅: {facts['peak_window_reduction_pct']}%")
    print(f"")
    if "summary_zh" in summary:
        print(f"## LLM 摘要")
        print(summary["summary_zh"])
        print(f"")
    if "highlight_zh" in summary:
        print(f"## LLM highlight")
        print(summary["highlight_zh"])
        print(f"")
    if "next_day_advice_zh" in summary:
        print(f"## LLM 明日建議")
        print(summary["next_day_advice_zh"])

    # Save
    out_dir = REPORTS / "daily"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"house_{args.house:02d}_day_{args.day:02d}.json"
    out_json.write_text(json.dumps({"facts": facts, "summary": summary},
                                    ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
