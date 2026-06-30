"""LLM-mediated user-facing recommendation.

Three modes:
  • template       — pure string templates, no LLM call. Used per timestep
                     inside the rollout.
  • ollama_summary — schema-constrained Ollama call for ONE polished daily
                     summary per house after the rollout finishes.
  • personalized   — Ollama call that takes the user's recent accept/reject
                     history as context. Produces tailored advice and
                     predicts likely acceptance.

A unit-string validator catches hallucinated kWh vs Wh confusions in any
free-form text field — see `validate_units()`.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import time
import urllib.request, urllib.error

from multi_household.agent.appliance_controller import AgentDecision
from multi_household.aggregator.price_broadcast import Broadcast


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:8b"


@dataclass
class Recommendation:
    """The user-facing message for one timestep."""
    house_id: int
    timestep: int
    hour: int
    headline: str          # one short sentence
    body: str              # the longer recommendation
    saving_gbp: float
    appliance: str
    accepted: bool


def _appliance_pretty(col: str) -> str:
    n = col.replace("appliance_", "").replace("_w", "").replace("_", " ")
    return n.title()


# --- template-based mode ----------------------------------------------------

def template_recommendation(house_id: int, step: int,
                            decision: AgentDecision,
                            broadcast: Broadcast) -> Recommendation | None:
    """Produce a recommendation iff the agent made a relevant decision.
    Returns None for routine no-ops or auto-releases that don't surface
    to the user.
    """
    if decision.action == "no_op" and not decision.rationale.get("user_accepted") is False:
        return None
    if decision.action == "release" and decision.rationale.get("reason") == "auto off-peak drain":
        return None      # silent — handled in background

    h = broadcast.hour
    p_now = broadcast.p_now_gbp_kwh
    p_off = broadcast.p_off_gbp_kwh

    if decision.action == "defer":
        apl = _appliance_pretty(decision.target_appliance or "appliance")
        time_word = "Tonight" if h >= 17 else ("This morning" if h < 12 else "This afternoon")
        headline = (f"Peak alert: defer {apl} to save £{decision.expected_saving_gbp:.2f}")
        body = (
            f"{time_word} {h:02d}:00 is a peak event "
            f"(grid forecast {broadcast.aggregate_forecast_w/1000:.1f} kW, "
            f"{broadcast.overage_ratio*100:.0f}% over capacity). "
            f"Price is now £{p_now:.2f}/kWh, off-peak {p_off:.2f}. "
            f"Suggest: hold {apl} until the next off-peak window."
        )
        return Recommendation(
            house_id=house_id, timestep=step, hour=h,
            headline=headline, body=body,
            saving_gbp=decision.expected_saving_gbp,
            appliance=decision.target_appliance or "",
            accepted=True,
        )

    if decision.action == "no_op" and decision.rationale.get("user_accepted") is False:
        apl = _appliance_pretty(decision.target_appliance or "appliance")
        return Recommendation(
            house_id=house_id, timestep=step, hour=h,
            headline=f"Suggestion declined: continue using {apl}",
            body=(f"You declined the suggestion to defer {apl}. "
                  f"Estimated additional cost £{decision.expected_saving_gbp:.2f}."),
            saving_gbp=0.0,
            appliance=decision.target_appliance or "",
            accepted=False,
        )

    if decision.action == "release" and decision.rationale.get("reason") == "comfort force release":
        apl = _appliance_pretty(decision.target_appliance or "appliance")
        return Recommendation(
            house_id=house_id, timestep=step, hour=h,
            headline=f"Releasing held {apl}",
            body=(f"{apl} cycle held for too long, releasing now to keep things on time."),
            saving_gbp=0.0,
            appliance=decision.target_appliance or "",
            accepted=True,
        )
    return None


# --- ollama mode (daily summary) --------------------------------------------

SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["summary_zh", "highlight_zh", "next_day_advice_zh"],
    "properties": {
        "summary_zh":         {"type": "string"},
        "highlight_zh":       {"type": "string"},
        "next_day_advice_zh": {"type": "string"},
    },
}

PERSONALIZED_SCHEMA = {
    "type": "object",
    "required": ["greeting_zh", "personal_note_zh",
                  "predicted_accept_prob", "rationale_zh",
                  "fact_citations"],
    "properties": {
        "greeting_zh":           {"type": "string"},
        "personal_note_zh":      {"type": "string"},
        "predicted_accept_prob": {"type": "number",
                                  "minimum": 0.0, "maximum": 1.0},
        "rationale_zh":          {"type": "string"},
        "fact_citations":        {"type": "array",
                                  "items": {"type": "string"}},
    },
}


# --- unit validation --------------------------------------------------------

BAD_UNITS = (
    ("kWh", "Wh"),
    ("MWh", "Wh"),
    ("kW",  "W"),
    ("MW",  "W"),
)


def validate_units(text: str, expected_unit: str = "Wh") -> list[str]:
    """Return list of issues — e.g. 'kWh found, expected Wh'."""
    issues = []
    low = text.lower()
    if expected_unit == "Wh":
        if "kwh" in low: issues.append("hallucinated unit 'kWh' (expected Wh)")
        if "mwh" in low: issues.append("hallucinated unit 'MWh' (expected Wh)")
    if expected_unit == "W":
        if "kw" in low and "kwh" not in low:
            issues.append("hallucinated unit 'kW' (expected W)")
    return issues

SYSTEM_PROMPT_SUMMARY = """
你是住戶能源顧問。任務:根據一天的「事實」JSON,寫一份簡短的繁體中文回顧。
規則:
1. 禁止使用 <think>/思考標記。
2. 禁止編造輸入沒給的數字。
3. summary_zh:1-2 句話總結今天省了多少電費、削掉幾個尖峰。
4. highlight_zh:1 句話指出最值得注意的點。
5. next_day_advice_zh:1 句話給明天的一般性建議。
6. 所有電量請用 Wh 單位,不要使用 kWh 或 MWh。
7. 嚴格遵守 JSON Schema。直接輸出 JSON。"""


SYSTEM_PROMPT_PERSONALIZED = """
你是個人化的住戶能源顧問。系統已經算好建議內容跟所有數字,你只負責
「**用 user 看得懂的話包裝**」+「**預測 user 會不會接受**」。

★ 輸入會包含兩種歷史:
  • 模擬紀錄 (rollout 跑出來的): n_recommendations / n_accepted / per_appliance
  • **真實 overrides** (closed loop): real_rejected / real_reject_by_appliance / real_recent_overrides

★ 真實 overrides **遠比模擬紀錄重要**。如果 user 對某設備真實拒絕過,即使模擬
  接受率高,你也要把預測接受率調低。

絕對規則:
1. 禁止使用 <think>/思考標記。
2. 禁止編造數字。你**只能引用**輸入 JSON 裡有的數字。
3. **不能重複寫死的建議內容**。系統已經會顯示「延後 X 設備、省 £Y、新時段 Z」
   給住戶看。你的工作是**補充上下文跟語氣**,不是重複事實。
4. `greeting_zh`:1 句問候,反映住戶接受傾向。
5. `personal_note_zh`:1-2 句個人化補充。如果有 real_overrides 包含拒絕,
   要明確提到「我注意到您上次拒絕了 X」這種承認。
6. `predicted_accept_prob`:0 到 1 的浮點數。**優先看 real_overrides**:
   - 如果 real_rejected > 0 對該設備,接受率必須 < 0.6
   - 如果 real_rejected ≥ real_total/2 對該設備,接受率必須 < 0.3
   - 沒 real_overrides 就回到模擬紀錄
7. `rationale_zh`:1-2 句,解釋為什麼這樣預測。**必須引用 real_overrides 數字**
   (例如:「您過去拒絕了這個設備 3 次,所以降低預測」)。
8. `fact_citations`:列出你在 personal_note + rationale 裡引用的**數字字串**。
9. 所有電量單位用 Wh,絕對不准用 kWh / MWh。
10. 嚴格遵守 JSON Schema。直接輸出 JSON,不要 markdown code block。"""


def call_ollama_personalized(facts: dict,
                              user_history: dict,
                              model: str = MODEL,
                              timeout_s: int = 120) -> dict | None:
    """Personalized recommendation using accept/reject history as context.

    user_history must include:
        n_recommendations:  total seen
        n_accepted:         total accepted
        accept_rate:        n_accepted / n_recommendations
        recent_decisions:   list of {accepted, appliance, hour} (last 5)
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_PERSONALIZED},
            {"role": "user",
             "content": ("住戶最近紀錄:\n```json\n"
                         + json.dumps(user_history, ensure_ascii=False, indent=2)
                         + "\n```\n\n當前推薦事實:\n```json\n"
                         + json.dumps(facts, ensure_ascii=False, indent=2)
                         + "\n```\n直接輸出 JSON。")},
        ],
        "stream": False,
        "format": PERSONALIZED_SCHEMA,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 700},
    }
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        content = obj["message"]["content"]
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1].lstrip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            a, b = content.find("{"), content.rfind("}")
            if a >= 0 and b > a:
                parsed = json.loads(content[a:b+1])
            else:
                return None
        # Validate units in all text fields (must match PERSONALIZED_SCHEMA keys)
        issues = []
        for k in ("greeting_zh", "personal_note_zh", "rationale_zh"):
            if k in parsed:
                issues.extend(validate_units(parsed[k], "Wh"))
        parsed["_unit_validation_issues"] = issues
        return parsed
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    except Exception:
        return None


def call_ollama_summary(facts: dict,
                        model: str = MODEL,
                        timeout_s: int = 120) -> dict | None:
    """Optional: call Ollama for a daily summary. Returns parsed dict or None
    on failure (silently swallows errors so missing Ollama doesn't break the
    rollout)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user",
             "content": "今日事實:\n```json\n" + json.dumps(facts, ensure_ascii=False, indent=2)
                        + "\n```\n直接輸出 JSON。"},
        ],
        "stream": False,
        "format": SUMMARY_SCHEMA,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 600},
    }
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        content = obj["message"]["content"]
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1].lstrip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            a, b = content.find("{"), content.rfind("}")
            if a >= 0 and b > a:
                parsed = json.loads(content[a:b+1])
            else:
                return None
        # Validate units in all text fields
        issues = []
        for k in ("summary_zh", "highlight_zh", "next_day_advice_zh"):
            if k in parsed:
                issues.extend(validate_units(parsed[k], "Wh"))
        parsed["_unit_validation_issues"] = issues
        return parsed
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    except Exception:
        return None
    return None
