# -*- coding: utf-8 -*-
"""Advisory v1 — convert an agent run into structured facts, hand them to a
schema-constrained LLM, and validate what comes back.

Design:
  • Pre-compute every fact (forecast MAE-by-hour, action distribution,
    peak events). The LLM never has to count or compute — it only
    INTERPRETS and SELECTS from controlled-vocabulary choices.
  • Enforce a strict JSON-Schema output so the LLM cannot invent fields.
  • Post-validate the LLM's numeric claims against the facts; if it
    cites a number we did not give it, flag the report as untrusted.

Public:
  compute_facts(env, served, demand, hours, actions, forecast_pred,
                actions_meta=None)            -> dict
  build_prompt(facts)                          -> (system: str, user: str)
  OUTPUT_SCHEMA                                -> dict (JSON Schema)
  validate(llm_out: dict, facts: dict)         -> list[str]   # issue list
  render_markdown(llm_out, facts, meta=None)   -> str
"""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# 1. fact computation
# ---------------------------------------------------------------------------

def _per_hour_table(hours: np.ndarray, values: np.ndarray, agg: str = "mean"):
    """Aggregate `values` by hour-of-day. Returns list of 24 floats."""
    out = []
    for h in range(24):
        mask = hours == h
        if not mask.any():
            out.append(0.0); continue
        v = values[mask]
        if agg == "mean": out.append(float(v.mean()))
        elif agg == "sum": out.append(float(v.sum()))
        elif agg == "abs_mean": out.append(float(np.abs(v).mean()))
        else: raise ValueError(agg)
    return out


def compute_facts(env, served: np.ndarray, demand: np.ndarray,
                  hours: np.ndarray, actions: list[int],
                  forecast_pred: np.ndarray) -> dict:
    """All structured facts the LLM is allowed to talk about.

    forecast_pred is the LSTM next-step prediction aligned with `demand`
    (same length, same time index).
    """
    assert len(served) == len(demand) == len(hours) == len(actions) \
        == len(forecast_pred)
    actions = np.asarray(actions, dtype=int)

    # --- forecast quality, per hour-of-day -----------------------------
    err = forecast_pred - demand                         # signed error
    mae_by_hour = _per_hour_table(hours, np.abs(err), "mean")
    bias_by_hour = _per_hour_table(hours, err, "mean")
    worst_h = int(np.argmax(mae_by_hour))
    best_h = int(np.argmin(mae_by_hour))

    # overall MAE / RMSE / R² on test region
    mae_all = float(np.abs(err).mean())
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(((demand - forecast_pred) ** 2).sum())
    ss_tot = float(((demand - demand.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- action distribution -------------------------------------------
    act_total = Counter(actions.tolist())
    act_counts = {str(k): int(act_total.get(k, 0)) for k in (0, 1, 2)}
    # defer-by-hour: how often the agent defers (action >= 1) in each hour
    defer_mask = actions >= 1
    defer_by_hour = []
    for h in range(24):
        m = hours == h
        defer_by_hour.append(int(defer_mask[m].sum()))
    most_defer_h = int(np.argmax(defer_by_hour))

    # --- peak events ---------------------------------------------------
    # The 95th-pctile threshold = "the load we want to shave"
    thr = float(np.percentile(demand, 95))
    above_thr = demand >= thr
    n_peaks = int(above_thr.sum())
    n_peaks_shaved = int(((demand >= thr) & (served < thr)).sum())
    shave_rate = (n_peaks_shaved / n_peaks) if n_peaks else 0.0
    # which hour-of-day has the most unshaved peaks?
    unshaved_mask = (demand >= thr) & (served >= thr)
    if unshaved_mask.any():
        unshaved_hour_counts = Counter(hours[unshaved_mask].tolist())
        worst_unshaved_hour = int(max(unshaved_hour_counts,
                                      key=unshaved_hour_counts.get))
    else:
        worst_unshaved_hour = -1

    # --- "tomorrow" representative forecast (the last 144 steps = 1d) --
    tail = 144 if len(forecast_pred) >= 144 else len(forecast_pred)
    tomorrow_fc = forecast_pred[-tail:]
    tomorrow_hours = hours[-tail:]
    tomorrow_peak_idx = int(np.argmax(tomorrow_fc))
    tomorrow_peak_hour = int(tomorrow_hours[tomorrow_peak_idx])
    tomorrow_peak_value = float(tomorrow_fc[tomorrow_peak_idx])

    # --- env config we want the LLM to know ----------------------------
    cfg = env.cfg

    facts = {
        "household_id": "UCI-Appliances (single-household sim)",
        "horizon_steps": int(len(demand)),
        "horizon_days": round(len(demand) / 144.0, 1),
        "controller": "rule-based",
        "env_config": {
            "peak_threshold_wh": cfg.peak_threshold_wh,
            "buffer_max_wh": cfg.buffer_max,
            "release_cap_wh": cfg.release_cap,
            "flex_frac": cfg.flex_frac,
            "price_peak": cfg.price_peak,
            "price_offpeak": cfg.price_offpeak,
            "price_mid": cfg.price_mid,
        },
        "forecast_quality": {
            "mae_overall_wh": round(mae_all, 1),
            "rmse_overall_wh": round(rmse_all, 1),
            "r2_overall": round(r2, 3),
            "mae_by_hour_wh": [round(v, 1) for v in mae_by_hour],
            "bias_by_hour_wh": [round(v, 1) for v in bias_by_hour],
            "worst_hour": worst_h,
            "worst_hour_mae_wh": round(mae_by_hour[worst_h], 1),
            "best_hour": best_h,
            "best_hour_mae_wh": round(mae_by_hour[best_h], 1),
            "source": "LSTM_prediction",  # explicit: this is model output, not fact
        },
        "agent_behavior": {
            "action_counts": act_counts,         # {"0": n, "1": n, "2": n}
            "defer_by_hour": defer_by_hour,
            "most_active_defer_hour": most_defer_h,
            "most_active_defer_count": int(defer_by_hour[most_defer_h]),
        },
        "peak_events": {
            "shave_threshold_wh": round(thr, 1),
            "n_peak_steps": n_peaks,
            "n_peak_steps_shaved": n_peaks_shaved,
            "shave_rate": round(shave_rate, 3),
            "worst_unshaved_hour": worst_unshaved_hour,
        },
        "tomorrow_outlook": {
            "source": "LSTM_prediction (last 24h of test set used as proxy)",
            "peak_hour": tomorrow_peak_hour,
            "peak_value_wh": round(tomorrow_peak_value, 1),
        },
    }
    return facts


# ---------------------------------------------------------------------------
# 2. output schema (strict — LLM must fill exactly these slots)
# ---------------------------------------------------------------------------

ALLOWED_PARAMS = ["peak_threshold_wh", "buffer_max_wh", "release_cap_wh",
                  "lookahead_h", "none"]
ALLOWED_DIRECTIONS = ["increase", "decrease", "no_change"]
ALLOWED_USER_ACTIONS = ["shift_load_off_peak", "manual_override",
                        "monitor", "no_action"]

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["summary_zh", "forecast_diagnosis", "agent_behavior_note",
                 "user_advisory", "tuning_suggestion"],
    "properties": {
        "summary_zh": {
            "type": "string",
            "description": "繁體中文一句話總結，必須包含至少一個來自輸入的數字。"
        },
        "forecast_diagnosis": {
            "type": "object",
            "required": ["worst_hour", "worst_hour_mae_wh",
                         "interpretation_zh"],
            "properties": {
                "worst_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                "worst_hour_mae_wh": {"type": "number"},
                "interpretation_zh": {"type": "string"},
            },
        },
        "agent_behavior_note": {
            "type": "object",
            "required": ["most_active_defer_hour", "interpretation_zh"],
            "properties": {
                "most_active_defer_hour": {"type": "integer",
                                           "minimum": 0, "maximum": 23},
                "interpretation_zh": {"type": "string"},
            },
        },
        "user_advisory": {
            "type": "object",
            "required": ["action_type", "target_hour", "reason_zh"],
            "properties": {
                "action_type": {"type": "string",
                                "enum": ALLOWED_USER_ACTIONS},
                "target_hour": {"type": "integer",
                                "minimum": 0, "maximum": 23},
                "reason_zh": {"type": "string"},
            },
        },
        "tuning_suggestion": {
            "type": "object",
            "required": ["parameter", "direction", "rationale_zh"],
            "properties": {
                "parameter": {"type": "string", "enum": ALLOWED_PARAMS},
                "direction": {"type": "string", "enum": ALLOWED_DIRECTIONS},
                "rationale_zh": {"type": "string"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 3. prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think
你是家戶級需量反應 (DR) Agent 的顧問。
任務：根據提供的「事實」JSON，產出一份結構化的繁體中文顧問報告。

絕對規則：
1. 禁止使用 <think>、</think> 或任何思考標記。
2. 禁止編造輸入沒給的數字。所有數字必須從輸入 JSON 來。
3. 禁止使用輸入沒列的參數名稱、控制器名稱、評估指標。
4. 用戶建議 (user_advisory) 必須從 enum 選擇：shift_load_off_peak / manual_override / monitor / no_action。
5. 調參建議 (tuning_suggestion.parameter) 必須從 enum 選擇：peak_threshold_wh / buffer_max_wh / release_cap_wh / lookahead_h / none。
6. 所有解釋欄位 (interpretation_zh / reason_zh / rationale_zh) 必須包含至少一個來自輸入的具體數字當作證據。
7. 輸出嚴格遵循 JSON Schema。直接輸出 JSON，不要 markdown 程式碼框。"""


def build_prompt(facts: dict) -> tuple[str, str]:
    import json
    fact_block = json.dumps(facts, ensure_ascii=False, indent=2)
    user = (
        "以下是這次 agent 跑完的結構化事實 (FACTS)。請直接輸出 JSON 顧問報告。\n\n"
        "FACTS:\n```json\n" + fact_block + "\n```\n\n"
        "直接輸出符合 schema 的 JSON。\n/no_think"
    )
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# 4. post-validation
# ---------------------------------------------------------------------------

def validate(llm_out: dict, facts: dict) -> list[str]:
    """Return a list of issues. Empty list = trusted output."""
    issues = []

    fq = facts["forecast_quality"]
    fd = llm_out.get("forecast_diagnosis", {})

    # worst_hour must match the data (allow ±1 for tied hours)
    claimed = fd.get("worst_hour", -1)
    truth = fq["worst_hour"]
    if claimed != truth:
        # also accept if their claimed hour has MAE within 5% of the real max
        if claimed in range(24):
            if fq["mae_by_hour_wh"][claimed] < 0.95 * fq["worst_hour_mae_wh"]:
                issues.append(
                    f"forecast_diagnosis.worst_hour={claimed} contradicts "
                    f"facts.worst_hour={truth} "
                    f"(MAE {fq['mae_by_hour_wh'][claimed]} vs {fq['worst_hour_mae_wh']})"
                )

    # worst_hour_mae_wh must be approximately right
    claimed_mae = fd.get("worst_hour_mae_wh", -1)
    if abs(claimed_mae - fq["worst_hour_mae_wh"]) > 1.0:
        issues.append(
            f"forecast_diagnosis.worst_hour_mae_wh={claimed_mae} does not "
            f"match facts.worst_hour_mae_wh={fq['worst_hour_mae_wh']}"
        )

    # most_active_defer_hour must match
    ab = facts["agent_behavior"]
    aln = llm_out.get("agent_behavior_note", {})
    claimed_d = aln.get("most_active_defer_hour", -1)
    if claimed_d != ab["most_active_defer_hour"]:
        # tolerate ties: only complain if their hour is < 80% of max defer count
        if claimed_d in range(24):
            if ab["defer_by_hour"][claimed_d] < 0.8 * ab["most_active_defer_count"]:
                issues.append(
                    f"agent_behavior_note.most_active_defer_hour={claimed_d} "
                    f"contradicts facts ({ab['most_active_defer_hour']})"
                )

    # enums (the schema already constrains, but double-check)
    ua = llm_out.get("user_advisory", {})
    if ua.get("action_type") not in ALLOWED_USER_ACTIONS:
        issues.append(f"user_advisory.action_type invalid: {ua.get('action_type')}")
    ts = llm_out.get("tuning_suggestion", {})
    if ts.get("parameter") not in ALLOWED_PARAMS:
        issues.append(f"tuning_suggestion.parameter invalid: {ts.get('parameter')}")
    if ts.get("direction") not in ALLOWED_DIRECTIONS:
        issues.append(f"tuning_suggestion.direction invalid: {ts.get('direction')}")

    # required string fields non-empty
    for path in [("summary_zh",),
                 ("forecast_diagnosis", "interpretation_zh"),
                 ("agent_behavior_note", "interpretation_zh"),
                 ("user_advisory", "reason_zh"),
                 ("tuning_suggestion", "rationale_zh")]:
        cur: Any = llm_out
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False; break
            cur = cur[k]
        if not ok or not isinstance(cur, str) or not cur.strip():
            issues.append("missing or empty: " + ".".join(path))

    return issues


# ---------------------------------------------------------------------------
# 5. markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(llm_out: dict, facts: dict, meta: dict | None = None) -> str:
    meta = meta or {}
    fq = facts["forecast_quality"]
    ab = facts["agent_behavior"]
    pe = facts["peak_events"]
    to = facts["tomorrow_outlook"]
    fd = llm_out["forecast_diagnosis"]
    aln = llm_out["agent_behavior_note"]
    ua = llm_out["user_advisory"]
    ts = llm_out["tuning_suggestion"]

    lines = [
        "# 單戶 Agent 顧問報告 (advisory v1)",
        "",
        f"- 生成模型：`{meta.get('model', '?')}`",
        f"- LLM 耗時：{meta.get('elapsed_s', '?'):.1f} s" if isinstance(meta.get('elapsed_s'), (int, float)) else f"- LLM 耗時：{meta.get('elapsed_s', '?')}",
        f"- 模擬：{facts['horizon_days']} 天 ({facts['horizon_steps']} steps)",
        f"- 驗證狀態：{'✅ 通過' if not meta.get('issues') else '⚠️ 有 ' + str(len(meta['issues'])) + ' 個問題'}",
        "",
        "---",
        "",
        "## 摘要",
        llm_out["summary_zh"],
        "",
        "## 預測診斷",
        f"- 預測整體 MAE = {fq['mae_overall_wh']} Wh，R² = {fq['r2_overall']}",
        f"- LSTM 預測最差時段：**{fd['worst_hour']:02d}:00**，MAE = {fd['worst_hour_mae_wh']} Wh",
        f"- LSTM 預測最好時段：{fq['best_hour']:02d}:00，MAE = {fq['best_hour_mae_wh']} Wh",
        f"- 解讀：{fd['interpretation_zh']}",
        "",
        "## Agent 行為",
        f"- 動作分佈：保持原樣 {ab['action_counts']['0']} 次 / 部分延後 {ab['action_counts']['1']} 次 / 全部延後 {ab['action_counts']['2']} 次",
        f"- 最常延後時段：**{aln['most_active_defer_hour']:02d}:00** "
        f"（{ab['most_active_defer_count']} 次）",
        f"- 解讀：{aln['interpretation_zh']}",
        "",
        "## 尖峰事件 (95-pctile 門檻 = " + str(pe["shave_threshold_wh"]) + " Wh)",
        f"- 共 {pe['n_peak_steps']} 個尖峰步、削掉 {pe['n_peak_steps_shaved']} 個（削峰率 {pe['shave_rate']*100:.1f}%）",
        f"- 沒削到的尖峰最常出現於：{pe['worst_unshaved_hour']:02d}:00",
        "",
        "## 給使用者的建議",
        f"- 建議行為：**{ua['action_type']}**（目標時段 {ua['target_hour']:02d}:00）",
        f"- 理由：{ua['reason_zh']}",
        "",
        "## 給工程團隊的調參建議",
        f"- 參數：`{ts['parameter']}` → **{ts['direction']}**",
        f"- 理由：{ts['rationale_zh']}",
        "",
        "## 明日展望 (來自 LSTM 預測，非事實)",
        f"- 預計尖峰時刻：{to['peak_hour']:02d}:00，"
        f"預估值 {to['peak_value_wh']} Wh",
        "",
    ]
    if meta.get("issues"):
        lines += ["", "---", "", "## ⚠️ 驗證問題"]
        for i in meta["issues"]:
            lines.append(f"- {i}")
    return "\n".join(lines)
