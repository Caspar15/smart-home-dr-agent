# -*- coding: utf-8 -*-
"""Phase-3 prelude: hand the single-household agent's run metrics to a local
LLM (Ollama / qwen3:4b) and have it write a Chinese summary report.

Run:  python -m experiments.llm_report
Output: reports/agent_report.md
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# Force UTF-8 stdout/stderr on Windows so 中文 doesn't come out as mojibake.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent.env import HouseholdEnv, EnvConfig
from src.agent.rule_based import decide as rule_decide
from src.agent.mpc import MPCController
from experiments.run_agent import rollout, metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


# ---------------------------------------------------------------------------
# 1. run the agent and collect metrics for all three controllers
# ---------------------------------------------------------------------------

def run_all_three() -> dict:
    env = HouseholdEnv(EnvConfig())
    s_b, d_b, p_b, h_b, lb = rollout(env, None)
    s_r, d_r, p_r, h_r, lr = rollout(env, lambda obs, e: rule_decide(obs, e))
    mpc = MPCController()
    t0 = time.time()
    s_m, d_m, p_m, h_m, lm = rollout(env, lambda obs, e: mpc.decide(obs, e))
    mpc_t = time.time() - t0

    return {
        "household_id": "UCI-Appliances (single-household sim)",
        "horizon_steps": int(len(d_b)),
        "horizon_days": round(len(d_b) / 144.0, 1),  # 144 = 10-min steps per day
        "price_schedule": {
            "peak": env.cfg.price_peak,
            "offpeak": env.cfg.price_offpeak,
            "mid": env.cfg.price_mid,
            "peak_hours": [17, 22],       # hard-coded in env._price_at
            "offpeak_hours": [0, 7],
        },
        "rule_based": metrics(s_r, d_r, p_r, h_r, lr),
        "mpc": metrics(s_m, d_m, p_m, h_m, lm),
        "mpc_solve_time_s": round(mpc_t, 1),
    }


# ---------------------------------------------------------------------------
# 2. talk to Ollama
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think
你是能源管理系統的技術助理。你的工作：把家戶級需量反應 (DR) 實驗的數值結果，寫成一份繁體中文技術報告。

絕對禁止：
- 禁止使用 <think> 或 </think> 標籤
- 禁止內部思考或自言自語
- 禁止任何形式的「我先想一下」、「先列出要點」之類的草稿過程
- 禁止客套話、結語、自我介紹

直接輸出最終報告。

格式（必須完全照抄這四個小節標題）：
## 摘要
（3-4 句話，包含關鍵數字）

## 成本與峰值表現
（條列出 rule-based 與 MPC 的電費節省、尖峰時段降幅、95 百分位削峰、PAR、能量守恆數字）

## 行為解讀
（解釋為什麼 MPC 在 95 百分位削峰上比 rule-based 明顯好，但電費節省差距小。關鍵：lookahead）

## 下一步建議
（2-3 點具體建議）"""

USER_TEMPLATE = """/no_think
實驗設定
- 家戶：{household_id}
- 模擬：{horizon_days} 天 ({horizon_steps} 個 10 分鐘步)
- ToU 電價：峰時 {peak_price}/kWh ({peak_start}–{peak_end} 時)，谷時 {offpeak_price}/kWh ({offpeak_start}–{offpeak_end} 時)，其餘 {mid_price}/kWh

Rule-based 控制器 vs 無 DR baseline
- 電費節省 {rule_cost_pct:.1f}%
- 尖峰時段 (17–22) 平均負載降 {rule_peakwin_pct:.1f}%
- 95 百分位削峰 {rule_p95_pct:.1f}%
- PAR {rule_PAR_b:.2f} -> {rule_PAR_a:.2f}
- 需求 {rule_demand:.0f} Wh，供給 {rule_served:.0f} Wh，未交付 {rule_left:.0f} Wh

MPC 控制器 (LP receding horizon, perfect foresight) vs 無 DR baseline
- 求解時間 {mpc_t:.1f} s
- 電費節省 {mpc_cost_pct:.1f}%
- 尖峰時段平均負載降 {mpc_peakwin_pct:.1f}%
- 95 百分位削峰 {mpc_p95_pct:.1f}%
- PAR {mpc_PAR_b:.2f} -> {mpc_PAR_a:.2f}
- 需求 {mpc_demand:.0f} Wh，供給 {mpc_served:.0f} Wh，未交付 {mpc_left:.0f} Wh

直接輸出四節報告。"""


def call_ollama(system: str, user: str, model: str = MODEL,
                temperature: float = 0.1, num_predict: int = 3500) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,         # top-level (Ollama 0.5+)
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": 0.9,
            "think": False,     # belt-and-braces
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read().decode("utf-8")
    dt = time.time() - t0
    obj = json.loads(raw)
    return {"content": obj["message"]["content"], "elapsed_s": dt, "raw": obj}


# ---------------------------------------------------------------------------
# 3. glue
# ---------------------------------------------------------------------------

def build_user_msg(data: dict) -> str:
    p = data["price_schedule"]
    r = data["rule_based"]
    m = data["mpc"]
    return USER_TEMPLATE.format(
        household_id=data["household_id"],
        horizon_days=data["horizon_days"],
        horizon_steps=data["horizon_steps"],
        peak_price=p["peak"], offpeak_price=p["offpeak"], mid_price=p["mid"],
        peak_start=p["peak_hours"][0], peak_end=p["peak_hours"][1],
        offpeak_start=p["offpeak_hours"][0], offpeak_end=p["offpeak_hours"][1],
        rule_cost_pct=r["cost_saving_pct"],
        rule_peakwin_pct=r["peakwin_reduction_pct"],
        rule_p95_pct=r["p95_reduction_pct"],
        rule_PAR_b=r["PAR_baseline"], rule_PAR_a=r["PAR_agent"],
        rule_demand=r["energy_demand"], rule_served=r["energy_served"],
        rule_left=r["undelivered_buffer"],
        mpc_t=data["mpc_solve_time_s"],
        mpc_cost_pct=m["cost_saving_pct"],
        mpc_peakwin_pct=m["peakwin_reduction_pct"],
        mpc_p95_pct=m["p95_reduction_pct"],
        mpc_PAR_b=m["PAR_baseline"], mpc_PAR_a=m["PAR_agent"],
        mpc_demand=m["energy_demand"], mpc_served=m["energy_served"],
        mpc_left=m["undelivered_buffer"],
    )


def main():
    print(f"\n[1/3] running 3-way agent comparison ...")
    data = run_all_three()
    raw_path = REPORTS / "agent_metrics.json"
    raw_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"      metrics -> {raw_path}")

    print(f"\n[2/3] calling Ollama ({MODEL}) ...")
    user_msg = build_user_msg(data)
    try:
        result = call_ollama(SYSTEM_PROMPT, user_msg)
    except urllib.error.URLError as e:
        print(f"  !! Ollama not reachable ({e}). Is `ollama serve` running?")
        return

    report = result["content"]
    # strip leftover <think>...</think> if model emitted one despite /no_think.
    # use rsplit because the model sometimes quotes the </think> token inside
    # its thinking content — we want the LAST occurrence (the real closer).
    if "</think>" in report:
        report = report.rsplit("</think>", 1)[1].lstrip()
    elif "<think>" in report:
        # unclosed think block — model ran out before emitting the report.
        # try to salvage anything after the last `## 摘要`-style header
        for marker in ("## 摘要", "**摘要**", "摘要："):
            if marker in report:
                report = report[report.index(marker):]
                break
        else:
            report = ("[警告] 模型一直在 <think> 模式裡沒輸出報告。"
                      "原始輸出見 raw 欄。\n\n---\n\n" + report)

    print(f"      LLM elapsed: {result['elapsed_s']:.1f}s, "
          f"output {len(report)} chars")

    out_path = REPORTS / "agent_report.md"
    header = (
        "# 單戶 Agent 報告書 (LLM-generated, 中文)\n\n"
        f"- 生成模型：Ollama / `{MODEL}`\n"
        f"- LLM 耗時：{result['elapsed_s']:.1f} s\n"
        f"- 模擬天數：{data['horizon_days']} 天 "
        f"({data['horizon_steps']} steps)\n"
        f"- 對應 metrics：`reports/agent_metrics.json`\n\n"
        "---\n\n"
    )
    out_path.write_text(header + report, encoding="utf-8")

    print(f"\n[3/3] report -> {out_path}\n")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
