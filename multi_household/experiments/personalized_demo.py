"""End-to-end demo of the personalized LLM advisor — v3 (interactive).

Run:  python -m multi_household.experiments.personalized_demo --house 7 --day 15

After all panels display, the script waits for user input (1/2/3):
   1 = Accept → simulate deferral, show savings
   2 = Reject → simulate normal run, show extra cost
   3 = Modify → prompt for new defer-to time
Add --auto to skip the prompt (useful for scripted runs).

User decisions are appended to reports/multi_household/user_choices.json
so future demos can show "you have overridden the system N times".
"""
from __future__ import annotations
import sys, io, argparse, json, re
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import numpy as np
import pandas as pd

from multi_household.config import (
    PEAK_HOURS_LOCAL, OFFPEAK_HOURS,
    PEAK_PRICE_GBP, MID_PRICE_GBP, OFFPEAK_PRICE_GBP,
)
from multi_household.experiments.metrics import REPORTS
from multi_household.experiments.daily_summary import compute_daily_facts
from multi_household.llm.advisor import (
    call_ollama_personalized, validate_units,
)


def _read_user_choices(house_id: int) -> list[dict]:
    """Read all REAL user overrides from user_choices.json for this house."""
    path = REPORTS / "user_choices.json"
    if not path.exists(): return []
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
        return [e for e in log if e.get("house") == house_id]
    except Exception:
        return []


def build_user_history(house_id: int, current_day: int) -> dict:
    """Combine SIMULATED rollout history + REAL user overrides.

    The rollout-side history (from rollout_coordinated_recs.json) tells us
    what would have happened under the synthetic 85% acceptance. The
    user_choices.json captures real overrides done in this demo. We merge
    both — real overrides take precedence — and ALSO compute a per-pattern
    rejection map that the agent can use to suppress future recommendations.
    """
    recs_path = REPORTS / "rollout_coordinated_recs.json"
    house_recs = []
    if recs_path.exists():
        all_recs = json.loads(recs_path.read_text(encoding="utf-8"))
        house_recs = [r for r in all_recs if r["house_id"] == house_id
                      and r["timestep"] < current_day * 144]

    n_total = len(house_recs)
    n_accepted = sum(1 for r in house_recs if r.get("accepted"))

    # Per-appliance accept stats from SIMULATED history
    from collections import defaultdict
    apl_total = defaultdict(int)
    apl_accept = defaultdict(int)
    for r in house_recs:
        apl = r["appliance"].replace("appliance_", "").replace("_w", "")
        apl_total[apl] += 1
        if r.get("accepted"): apl_accept[apl] += 1
    per_appliance = {a: {"total": apl_total[a],
                          "accepted": apl_accept[a],
                          "rate": round(apl_accept[a]/apl_total[a], 3)
                                  if apl_total[a] else 0.0}
                      for a in apl_total}

    recent = []
    for r in house_recs[-5:]:
        apl = r["appliance"].replace("appliance_", "").replace("_w", "")
        recent.append({"accepted":  bool(r.get("accepted")),
                        "appliance": apl, "hour": r.get("hour"),
                        "source": "sim"})

    # --- REAL user overrides (the closed loop) -----------------------------
    real = _read_user_choices(house_id)
    n_real_accept   = sum(1 for e in real if e.get("user_choice") == 1)
    n_real_reject   = sum(1 for e in real if e.get("user_choice") == 2)
    n_real_modify   = sum(1 for e in real if e.get("user_choice") == 3)

    # Pattern rejection count: (appliance, hour_bucket) → reject count
    # hour_bucket: peak (17-22) / mid (6-17, 22-24) / off (0-6)
    def _bucket(h):
        if h is None or h < 0: return "unknown"
        if 0 <= h < 6:  return "off-peak"
        if 17 <= h < 22: return "peak"
        return "mid"
    pattern_reject: dict[str, int] = {}
    pattern_total:  dict[str, int] = {}
    for e in real:
        apl = e.get("rec_appliance", "")
        cons = e.get("consequence", {})
        # The hour comes from the rec; we don't store it in the choice
        # but we know it's the rec_step's hour. For simplicity here use
        # the bucket of the rec's hour if we have it.
        # We'll use a coarse pattern: appliance × user_choice
        pat = apl
        pattern_total[pat] = pattern_total.get(pat, 0) + 1
        if e.get("user_choice") == 2:
            pattern_reject[pat] = pattern_reject.get(pat, 0) + 1

    real_recent = []
    for e in real[-5:]:
        cons = e.get("consequence", {})
        real_recent.append({
            "user_choice":  e.get("user_choice"),
            "appliance":    e.get("rec_appliance"),
            "action":       cons.get("action"),
            "source":       "real",
        })

    return {
        "n_recommendations":  n_total,
        "n_accepted":         n_accepted,
        "accept_rate":        round(n_accepted / n_total if n_total else 0, 3),
        "per_appliance":      per_appliance,
        "recent_decisions":   recent,
        # REAL closed-loop data:
        "real_overrides_total":   len(real),
        "real_accepted":          n_real_accept,
        "real_rejected":          n_real_reject,
        "real_modified":          n_real_modify,
        "real_reject_by_appliance": pattern_reject,
        "real_total_by_appliance":  pattern_total,
        "real_recent_overrides":    real_recent,
    }


def pick_sample_recommendation(house_id: int, current_day: int) -> dict | None:
    """Find ONE recommendation from this day for this house — that's what
    we'll wrap in the advisory."""
    recs_path = REPORTS / "rollout_coordinated_recs.json"
    if not recs_path.exists(): return None
    all_recs = json.loads(recs_path.read_text(encoding="utf-8"))
    day_recs = [r for r in all_recs
                if r["house_id"] == house_id
                and current_day*144 <= r["timestep"] < (current_day+1)*144]
    if not day_recs: return None
    # pick the one with the highest expected saving
    return max(day_recs, key=lambda r: r.get("saving_gbp", 0))


def extract_numbers(text: str) -> list[str]:
    """Pull out number-looking tokens from a string for validation."""
    # Includes percentages, decimals, GBP amounts
    return re.findall(r"\d+(?:[.,]\d+)?(?:%|次|GBP|£|W)?", text)


def collect_all_known_numbers(facts: dict, hist: dict, rec: dict) -> set[str]:
    """Numbers the LLM is allowed to cite."""
    known = set()
    def add(v):
        if isinstance(v, (int, float)):
            known.add(str(v))
            if isinstance(v, float):
                known.add(f"{v:.0f}")
                known.add(f"{v:.1f}")
                known.add(f"{v:.2f}")
                known.add(f"{v:.3f}")
        elif isinstance(v, str):
            known.add(v.lower())

    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        else:
            add(o)
    walk(facts); walk(hist); walk(rec or {})
    return {x for x in known if x and x != "none"}


def render_panel(facts: dict, hist: dict, rec: dict | None,
                 llm: dict | None) -> str:
    out = []
    def hr(s=""): out.append(s)
    def box(title, lines):
        bar = "═" * 60
        hr(f"╔══ {title} ".ljust(62, "═") + "╗")
        for L in lines: hr(f"║ {L:<58} ║")
        hr("╚" + bar + "╝")
        hr()

    h_id = facts.get("household_id", "?")
    date = facts.get("date", "?")
    tf   = facts.get("tomorrow_forecast", {})
    hr(f"════════════════════════════════════════════════════════════════")
    hr(f"  住戶 {h_id} 能源建議  ({date})")
    hr(f"════════════════════════════════════════════════════════════════")
    hr()

    # --- Panel 1: today numbers ---------------------------------------
    p1 = [
        f"今日電費 (DR 前):         £{facts['cost_baseline_gbp']:.3f}",
        f"今日電費 (DR 後):         £{facts['cost_after_gbp']:.3f}",
        f"已節省:                   £{facts['cost_saving_gbp']:.3f}  ({facts['cost_saving_pct']}%)",
        f"今日已收到推薦:           {facts['recommendations_total']} 次",
        f"已接受:                   {facts['recommendations_accepted']} 次",
        f"尖峰時段平均負載 (DR 前): {facts['peak_window_mean_demand_w']:.1f} W",
        f"尖峰時段平均負載 (DR 後): {facts['peak_window_mean_served_w']:.1f} W",
        f"尖峰時段降幅:             {facts['peak_window_reduction_pct']}%",
    ]
    box("📊 今日數字 (寫死,來自 rollout)", p1)

    # --- Panel 2: tomorrow forecast (with honesty label) ---------------
    if "peak_value_w" in tf:
        peak_h_idx = tf["peak_hour_idx"]
        peak_h = peak_h_idx // 6
        peak_m = (peak_h_idx % 6) * 10
        p2 = [
            f"預測來源:                 {tf['source']}",
            f"預估明日尖峰時刻:         +{peak_h:02d}:{peak_m:02d}",
            f"預估尖峰負載:             {tf['peak_value_w']:.1f} W",
            f"預估平均:                 {tf['mean_w']:.1f} W",
            f"預估最低:                 {tf['min_w']:.1f} W",
            f"⚠️  {tf.get('note', '')}",
        ]
        box("🔮 明日 24h 預測 (LSTM recursive,誤差會累積)", p2)

    # --- Panel 3: this recommendation (hardcoded from agent) ----------
    if rec is not None:
        apl_pretty = rec["appliance"].replace("appliance_", "").replace("_w", "")
        p3 = [
            f"觸發時刻:                 {rec['hour']:02d}:00 (step {rec['timestep']})",
            f"目標設備:                 {apl_pretty}",
            f"建議動作:                 延後到下個離峰時段",
            f"預估省電費:               £{rec.get('saving_gbp', 0):.3f}",
            f"觸發原因:                 {rec.get('headline', '')[:55]}",
        ]
        box("💡 本次建議 (寫死,來自 agent)", p3)

    # --- Panel 4: user history -----------------------------------------
    p4 = [
        f"模擬接受率 (rollout):     {hist['n_accepted']}/{hist['n_recommendations']}"
        f" = {hist['accept_rate']*100:.1f}%",
    ]
    if hist.get("per_appliance"):
        p4.append("各設備模擬接受率:")
        for apl, st in list(hist["per_appliance"].items())[:5]:
            p4.append(f"  {apl:<22} {st['accepted']}/{st['total']} ({st['rate']*100:.0f}%)")
    p4.append("")
    p4.append(f"★ 您實際 overrides (closed loop):")
    p4.append(f"  總計: {hist.get('real_overrides_total', 0)} 次   "
              f"接受 {hist.get('real_accepted', 0)} / "
              f"拒絕 {hist.get('real_rejected', 0)} / "
              f"修改 {hist.get('real_modified', 0)}")
    if hist.get("real_reject_by_appliance"):
        p4.append("  您拒絕過的設備:")
        for apl, n_rej in hist["real_reject_by_appliance"].items():
            n_tot = hist["real_total_by_appliance"].get(apl, 0)
            p4.append(f"    {apl:<20} {n_rej}/{n_tot} 次拒絕")
    box("📈 您的紀錄 (寫死,模擬 + 真實 overrides)", p4)

    # --- Panel 4.5: closed-loop warning (NEW) --------------------------
    if rec is not None:
        apl_pretty = rec["appliance"].replace("appliance_", "").replace("_w", "")
        n_rej = hist.get("real_reject_by_appliance", {}).get(apl_pretty, 0)
        n_tot = hist.get("real_total_by_appliance", {}).get(apl_pretty, 0)
        if n_rej >= 2:
            warn = [
                f"⚠️  您過去 {n_tot} 次收到 {apl_pretty} 的建議,",
                f"   拒絕了 {n_rej} 次 ({n_rej/n_tot*100:.0f}% 拒絕率)",
                f"",
                f"系統會把這個納入考量:",
                f"  • 下次 rollout 會降低 {apl_pretty} 在此時段的觸發頻率",
                f"  • LLM 預測接受率會自動下調",
            ]
            box("🔔 學習提示 — 此設備您常拒絕", warn)

    # --- Panel 5: LLM personalized touch -------------------------------
    if llm is not None:
        p5 = []
        if llm.get("greeting_zh"):       p5.append(llm["greeting_zh"])
        if llm.get("personal_note_zh"):  p5.append(llm["personal_note_zh"])
        box("🤖 個人化補充 (LLM 生成,有驗證)", p5)

        # Predicted accept probability
        p = llm.get("predicted_accept_prob", 0)
        n_full = int(p * 30)
        bar = "█" * n_full + "░" * (30 - n_full)
        p6 = [f"預測接受機率:  {bar}  {p*100:.0f}%",
              f"推理:          {llm.get('rationale_zh', '')}"]
        box("🎯 接受機率預測 (LLM 估計)", p6)

    # --- Panel 6: action -----------------------------------------------
    p7 = [
        "[ 1 ] 接受建議  — 系統會排程延後並執行",
        "[ 2 ] 拒絕      — 設備正常運轉,系統會學習這次拒絕",
        "[ 3 ] 修改時間  — 您指定一個延後到的時段",
    ]
    box("✅ 您的選擇", p7)

    return "\n".join(out)


def validate_llm(llm: dict, known: set[str]) -> dict:
    """Cross-check LLM's free-form text against known facts.

    Returns dict with:
      unit_issues:      kWh / MWh hallucinations
      cited:            list[str] — what LLM said it cited
      cited_unverified: list[str] — citations that don't appear in known facts
      numbers_in_text:  numbers found in LLM text
      numbers_unknown:  numbers in LLM text that aren't in known facts
    """
    unit_issues = []
    for k in ("greeting_zh", "personal_note_zh", "rationale_zh"):
        if k in llm:
            unit_issues.extend(validate_units(llm[k], "Wh"))

    cited = [c for c in (llm.get("fact_citations") or []) if c]
    known_lower = {k.lower() for k in known}
    cited_unverified = [c for c in cited
                        if c.lower() not in known_lower
                        and not any(c.lower() in k.lower() for k in known_lower)]

    text_blob = " ".join(str(llm.get(k, "")) for k in
                          ("greeting_zh", "personal_note_zh", "rationale_zh"))
    numbers = extract_numbers(text_blob)
    nums_unknown = []
    for n in numbers:
        # Strip suffixes for matching
        bare = re.sub(r"[%次GBP£W]", "", n)
        if bare in known or n in known: continue
        # Number ≤ 3 is too generic, skip
        try:
            if float(bare.replace(",", ".")) < 5: continue
        except ValueError:
            pass
        nums_unknown.append(n)

    return {
        "unit_issues":      unit_issues,
        "cited":            cited,
        "cited_unverified": cited_unverified,
        "numbers_in_text":  numbers,
        "numbers_unknown":  nums_unknown,
        "all_ok":           (not unit_issues
                              and not cited_unverified
                              and not nums_unknown),
    }


def consequence_of_action(choice: int, modified_hour: int | None,
                          rec: dict, facts: dict) -> dict:
    """Compute and explain what the chosen action would do.

    This is HARDCODED math from the facts. No LLM. The user can verify
    every line.
    """
    saving = float(rec.get("saving_gbp", 0.0)) if rec else 0.0
    apl = (rec.get("appliance", "") if rec else "").replace("appliance_", "").replace("_w", "")
    trigger_hour = rec.get("hour", -1) if rec else -1
    out = {"choice": choice, "appliance": apl}

    if choice == 1:                          # Accept
        out["action"]      = "deferred"
        out["new_time_zh"] = "下一個離峰時段 (預設 22:00 後)"
        out["impact_gbp"]  = -saving
        out["explanation"] = (f"系統會把 {apl} 排程到下一個離峰時段。"
                              f"預估電費省 £{saving:.3f}。設備會在離峰時段自動執行。")
    elif choice == 2:                        # Reject
        out["action"]      = "rejected"
        out["new_time_zh"] = f"{trigger_hour:02d}:00 (照常運轉)"
        out["impact_gbp"]  = +saving        # lost saving
        out["explanation"] = (f"{apl} 維持在 {trigger_hour:02d}:00 啟動。"
                              f"放棄省下的 £{saving:.3f}。系統會記錄這次拒絕,"
                              f"未來推薦類似情境的頻率會降低。")
    elif choice == 3:                        # Modify
        if modified_hour is None or not (0 <= modified_hour <= 23):
            out["action"]      = "modify_invalid"
            out["explanation"] = "未提供有效時段 (0-23)。本次推薦視為拒絕。"
            out["impact_gbp"]  = +saving
        else:
            # estimate partial saving — peak hours get 0.30, off-peak 0.08,
            # mid 0.15. (Same as base_tou_price.)
            from multi_household.aggregator.price_broadcast import base_tou_price
            p_orig = base_tou_price(trigger_hour)
            p_new  = base_tou_price(modified_hour)
            # use the recommendation's expected_cycle_wh if available
            # otherwise approximate from saving / (p_orig - p_off)
            cycle_wh = (saving / max(p_orig - 0.08, 1e-6)) * 1000.0
            partial_saving = (p_orig - p_new) * cycle_wh / 1000.0
            out["action"]      = "modified"
            out["new_time_zh"] = f"{modified_hour:02d}:00 (使用者指定)"
            out["impact_gbp"]  = -partial_saving
            out["explanation"] = (
                f"{apl} 排到 {modified_hour:02d}:00。"
                f"原觸發時段 {trigger_hour:02d}:00 電價 £{p_orig:.2f}/kWh,"
                f"指定時段 £{p_new:.2f}/kWh,"
                f"預估省 £{partial_saving:.3f}。")
    return out


def log_user_choice(house_id: int, day: int, choice_dict: dict) -> Path:
    """Append the user's decision to a persistent log."""
    log_path = REPORTS / "user_choices.json"
    log = []
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append({
        "house": house_id, "day": day,
        "timestamp_logged": str(pd.Timestamp.now()),
        **choice_dict,
    })
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", type=int, default=7)
    ap.add_argument("--day",   type=int, default=15)
    ap.add_argument("--auto",  action="store_true",
                    help="skip interactive prompt (always treat as accept)")
    args = ap.parse_args()

    print(f"[1/4] Build daily facts (incl. REAL 24h forecast) ...")
    facts = compute_daily_facts(args.house, args.day)
    tf = facts.get("tomorrow_forecast", {})
    print(f"      tomorrow source: {tf.get('source')}")

    print(f"\n[2/4] Build user history ...")
    hist = build_user_history(args.house, args.day)
    print(f"      lifetime: {hist['n_accepted']}/{hist['n_recommendations']} "
          f"accepted ({hist['accept_rate']*100:.1f}%)")

    print(f"\n[3/4] Pick sample recommendation + call Ollama ...")
    rec = pick_sample_recommendation(args.house, args.day)
    if rec is None:
        print(f"      ⚠️  no recommendations for house {args.house} day {args.day}")
        print(f"          (system did not detect a peak event needing this house)")
    llm = None
    if rec is not None:
        llm = call_ollama_personalized(facts, hist)
        if llm is None:
            print(f"      !! Ollama unreachable")
        else:
            print(f"      ✅ Ollama responded with keys: {list(llm.keys())}")

    print(f"\n[4/4] Render panel + validate\n")
    panel = render_panel(facts, hist, rec, llm)
    print(panel)

    if llm is not None:
        known = collect_all_known_numbers(facts, hist, rec)
        val = validate_llm(llm, known)
        print()
        print("╔══ 🔍 驗證報告 (LLM 有沒有亂編?) " + "═" * 28 + "╗")
        if val["all_ok"]:
            print("║  ✅ 全部通過                                                ║")
        if val["unit_issues"]:
            print("║  ❌ 單位 hallucination:                                     ║")
            for i in val["unit_issues"]:
                print(f"║     - {i:<54} ║")
        print(f"║  LLM 宣稱引用的事實: {len(val['cited'])} 個                                 ║")
        for c in val["cited"][:5]:
            mark = "✓" if c.lower() in {k.lower() for k in known} or any(c.lower() in k.lower() for k in known) else "✗"
            print(f"║     [{mark}] {c:<50} ║")
        if val["numbers_unknown"]:
            print(f"║  ⚠️  文字裡的數字找不到對應事實: {val['numbers_unknown'][:5]}".ljust(63) + " ║")
        print("╚" + "═" * 60 + "╝")

    # ---- INTERACTIVE PROMPT --------------------------------------------
    user_choice = None
    if rec is not None:
        if args.auto:
            print(f"\n[--auto] 預設選擇: 1 (接受)")
            user_choice = 1
            modified_hour = None
        else:
            print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  請輸入您的選擇 (1/2/3),按 Enter 確認:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            try:
                raw = input("  >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = ""
            modified_hour = None
            if raw == "1":
                user_choice = 1
            elif raw == "2":
                user_choice = 2
            elif raw == "3":
                user_choice = 3
                try:
                    h = input("  延後到幾點 (0-23)?  >>> ").strip()
                    modified_hour = int(h)
                except (EOFError, ValueError, KeyboardInterrupt):
                    modified_hour = None
            else:
                print(f"\n  ⚠️  未輸入有效選項,跳過 (視為拒絕)")
                user_choice = 2

        if user_choice is not None:
            cons = consequence_of_action(user_choice, modified_hour, rec, facts)
            print(f"\n╔══ 📋 您的選擇 + 後果 (寫死計算,無 LLM) " + "═" * 21 + "╗")
            print(f"║  選擇:    {user_choice}                                              ║")
            print(f"║  動作:    {cons.get('action', ''):<48s} ║")
            print(f"║  目標設備: {cons.get('appliance', ''):<48s} ║")
            print(f"║  新時段:   {cons.get('new_time_zh', '')[:48]:<48s} ║")
            sign = "節省" if cons.get('impact_gbp', 0) < 0 else "多花"
            print(f"║  金額影響: {sign} £{abs(cons.get('impact_gbp', 0)):.3f}                                  ║")
            print(f"║                                                              ║")
            # explanation may be long; wrap
            expl = cons.get('explanation', '')
            for i in range(0, len(expl), 56):
                print(f"║  {expl[i:i+56]:<58s} ║")
            print(f"╚" + "═" * 60 + "╝")

            log_path = log_user_choice(args.house, args.day, {
                "rec_step":       rec["timestep"],
                "rec_appliance":  cons.get("appliance"),
                "user_choice":    user_choice,
                "modified_hour":  modified_hour,
                "consequence":    cons,
                "llm_predicted_accept_prob": (llm or {}).get("predicted_accept_prob"),
            })
            print(f"\n💾 user decision logged → {log_path.name}")

            # Show LLM's prediction vs actual choice
            if llm is not None:
                pred = llm.get("predicted_accept_prob", 0.5)
                actual = 1.0 if user_choice == 1 else 0.0
                error = abs(pred - actual)
                if user_choice == 3:
                    actual = 0.5
                    error = abs(pred - 0.5)
                tag = "✅ 預測正確" if error < 0.3 else "❌ 預測偏離"
                print(f"   LLM 預測接受率 {pred*100:.0f}% vs 實際 {actual*100:.0f}%   {tag}")

    out_dir = REPORTS / "personalized"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"house_{args.house:02d}_day_{args.day:02d}.json"
    out_json.write_text(json.dumps({
        "facts": facts, "history": hist, "recommendation": rec,
        "llm_advice": llm,
        "user_choice": user_choice,
        "consequence": consequence_of_action(user_choice, modified_hour if user_choice == 3 else None, rec, facts)
                       if user_choice is not None and rec is not None else None,
        "validation": validate_llm(llm, collect_all_known_numbers(facts, hist, rec))
                       if llm else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 full record → {out_json}")


if __name__ == "__main__":
    main()
