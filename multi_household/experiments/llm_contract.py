"""Targeted output contract for the advisory messages — post-hoc, no LLM calls.

Why this exists
---------------
The manuscript's interface section describes a richer contract than
`llm_eval.py` measures: besides schema, number grounding and units, it claims
checks on citation COVERAGE (every number in the message is declared in
`cited_numbers`), on EV messages carrying an explicit recommended time, and on
the message reading as a request for consent rather than a notification that
the change already happened. Those three were never implemented, so the
manuscript reported figures no artifact supports.

All three are computable from what `llm_eval.py` already saves per row
(`message_zh`, `cited_numbers_raw`, `facts`, `event_type`), so this runs as
pure post-processing — no new generation, and the numbers refer to exactly the
messages that were evaluated.

Honest scope
------------
Checks 1-4 are mechanical: they compare numeric literals and unit strings, and
either hold or do not. Checks 5-6 are keyword heuristics over Traditional
Chinese surface forms. A pass means "satisfied the implemented check", never
"semantically correct" — a message can cite every number correctly and still
attach it to the wrong concept. The manuscript must say so.

Run:    python -m multi_household.experiments.llm_contract [--model llama3.1:8b]
Writes: reports/multi_household/llm_contract.json
"""
from __future__ import annotations
import sys, json, re, argparse

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.experiments.rollout import REPORTS

_NUM = re.compile(r"\d+(?:\.\d+)?")

# A rendered clock time, checked as one token rather than two numbers.
_CLOCK = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})")

# An explicit clock time: "23:00", "23 時 50 分", "晚上 9 點 30 分".
# "晚上 9 點" (hour only, no minutes) deliberately does NOT match — the
# manuscript's contract asks for the minute to be stated.
_EXPLICIT_TIME = re.compile(
    r"(\d{1,2}\s*[:：]\s*\d{2})"                       # 23:00
    r"|(\d{1,2}\s*[點点時时]\s*\d{1,2}\s*分)"           # 9 點 30 分
    r"|(\d{1,2}\s*[點点時时]\s*整)"                     # 9 點整
)

# Wording that asserts the change already happened. The message is shown to a
# user who has NOT decided yet, so any of these is a contract violation.
_COMPLETED = [
    "已調整", "已经调整", "已經調整", "已排程", "已安排", "已接受",
    "已為您", "已为您", "已將", "已将", "已完成", "已延後", "已延后",
    "已更改", "已變更", "已改為", "已改为", "系統已", "系统已",
]

# Wording that asks for consent. At least one must be present.
_ASKING = [
    "是否", "可以嗎", "可以吗", "願意", "愿意", "同意", "要不要",
    "好嗎", "好吗", "建議", "建议", "請問", "请问", "是否同意",
]


def _hour_variants(h) -> set[str]:
    """Renderings of an hour fact that are conversions, not inventions.

    An hour fact of 21 legitimately appears as 21, 09 (12-hour clock), or 9.
    `llm_eval` makes the same allowance; without it the ungrounded-number rate
    inflates roughly sevenfold.
    """
    out: set[str] = set()
    try:
        hi = int(h)
    except (TypeError, ValueError):
        return out
    out.add(str(hi))
    out.add(f"{hi:02d}")
    h12 = hi % 12 or 12
    out.add(str(h12))
    out.add(f"{h12:02d}")
    return out


def _allowed_numbers(facts: dict) -> set[str]:
    """Every literal a faithful message may legitimately contain.

    Deliberately permissive about RENDERING and strict about VALUE:
      * rounding variants, because GBP 0.451 may be written 0.45;
      * the ":00" minutes token of an explicit clock time;
      * 12-hour-clock and zero-padded forms of an hour fact;
      * numbers carried inside string facts, e.g. washing_machine_2.
    A wrong magnitude is not a rendering, so it still fails — that is what
    `_currency_ok` is for.
    """
    out: set[str] = set()
    for k, v in (facts or {}).items():
        if isinstance(v, str):
            out |= set(_NUM.findall(v))
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        for s in (f"{f:g}", f"{f:.0f}", f"{f:.1f}", f"{f:.2f}", f"{f:.3f}"):
            out.add(s)
            if s.endswith(".0"):
                out.add(s[:-2])
        if f.is_integer():
            out.add(str(int(f)))
            out.add(f"{int(f):02d}")
        if k == "hour":
            out |= _hour_variants(v)
            out.add("00")                     # the minutes token of "21:00"
    return out



# Money in the message must carry the same magnitude as the fact. The model
# renders saving_gbp 0.021 as "2.1 英鎊" often enough to matter: for an
# advisory message that is a hundredfold overstatement of what the user gains,
# and no grounding or unit check catches it, because 2.1 is a real number and
# the currency word is correct.
_GBP = re.compile(r"£\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:英鎊|鎊)")
_PENCE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:p\b|便士|辨士)")


def _currency_ok(msg: str, facts: dict) -> bool | None:
    """None when the event carries no money fact, else whether it is faithful."""
    if "saving_gbp" not in facts:
        return None
    try:
        gbp = float(facts["saving_gbp"])
    except (TypeError, ValueError):
        return None
    claims = []
    for m in _GBP.finditer(msg):
        claims.append(float(m.group(1) or m.group(2)))
    for m in _PENCE.finditer(msg):
        claims.append(float(m.group(1)) / 100.0)
    if not claims:
        return True                      # states no amount; nothing to get wrong
    tol = max(0.005, abs(gbp) * 0.02)
    return all(abs(c - gbp) <= tol for c in claims)


def check_row(row: dict) -> dict:
    """Apply the checks to one evaluated message."""
    msg = row.get("message_zh")
    res = {
        "schema_ok": bool(row.get("schema_ok")),
        "numbers_grounded": None,
        "citations_grounded": None,
        "citation_coverage": None,
        "units_ok": None,
        "currency_scale_ok": None,
        "time_explicit": None,
        "wording_is_proposal": None,
    }
    if not row.get("schema_ok") or not isinstance(msg, str):
        return res                      # nothing to check against

    facts0 = row.get("facts") or {}
    allowed0 = _allowed_numbers(facts0)
    # Recomputed here rather than taken from llm_eval's stored field: that
    # field was produced by a version that ignored numeric literals inside
    # string facts, so it over-reported ungrounded numbers.
    in_msg0 = set(_NUM.findall(msg))
    res["numbers_grounded"] = in_msg0.issubset(allowed0)
    declared0: set[str] = set()
    for c in row.get("cited_numbers_raw") or []:
        declared0 |= set(_NUM.findall(str(c)))
    # also recomputed locally, for the same reason as numbers_grounded
    res["citations_grounded"] = declared0.issubset(allowed0)
    res["units_ok"] = not row.get("unit_issues")
    res["currency_scale_ok"] = _currency_ok(msg, facts0)

    # Coverage: is every number in the prose also declared in cited_numbers?
    # Two things this must NOT do:
    #  - flag the house id. It addresses the reader, it is not quoted evidence.
    #  - split a clock time. "23:00" is one rendering of hour=23, not the two
    #    numbers 23 and 00; a naive scan flags every correct EV message.
    # Clock times are therefore checked as times (hour must be backed by the
    # hour fact, minutes must be :00 unless a fact supplies them) and then
    # removed before the remaining loose numbers are checked.
    facts = row.get("facts") or {}
    exempt = {str(facts.get("house_id"))} if facts.get("house_id") is not None else set()
    allowed = _allowed_numbers(facts)
    declared: set[str] = set()
    for c in row.get("cited_numbers_raw") or []:
        declared |= set(_NUM.findall(str(c)))

    ok = True
    for m in _CLOCK.finditer(msg):
        hh, mm = m.group(1), m.group(2)
        if hh.lstrip("0") not in {a.lstrip("0") for a in allowed | declared}:
            ok = False                       # hour not backed by any fact
        if mm != "00" and mm not in (allowed | declared):
            ok = False                       # invented a minute
    rest = _CLOCK.sub(" ", msg)
    if not (set(_NUM.findall(rest)) - exempt).issubset(declared):
        ok = False
    res["citation_coverage"] = ok

    # explicit recommended time — only meaningful for EV reschedules
    if row.get("event_type") == "ev_advisory":
        res["time_explicit"] = bool(_EXPLICIT_TIME.search(msg))

    res["wording_is_proposal"] = (
        not any(w in msg for w in _COMPLETED) and any(w in msg for w in _ASKING))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()

    safe = args.model.replace(":", "_").replace("/", "_")
    src = REPORTS / f"llm_eval_{safe}.json"
    if not src.exists():
        print(f"missing {src} — run llm_eval first")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    if not rows:
        print("no rows to analyse")
        return

    checked = [dict(r, _contract=check_row(r)) for r in rows]
    names = ["schema_ok", "numbers_grounded", "citations_grounded",
             "citation_coverage", "units_ok", "currency_scale_ok",
             "time_explicit", "wording_is_proposal"]

    per_check = {}
    for n in names:
        vals = [c["_contract"][n] for c in checked if c["_contract"][n] is not None]
        per_check[n] = {
            "applicable": len(vals),
            "passed": sum(vals),
            "rate": round(sum(vals) / len(vals), 4) if vals else None,
        }

    # joint: a message passes only if every applicable check passes
    joint = sum(all(v for v in c["_contract"].values() if v is not None)
                for c in checked)

    out = {
        "model": data.get("summary", {}).get("model", args.model),
        "source": src.name,
        "n_messages": len(checked),
        "note": ("post-hoc contract over the messages llm_eval already "
                 "generated; no new LLM calls. schema/grounding/coverage/units "
                 "are mechanical; time_explicit and wording_is_proposal are "
                 "keyword heuristics over Traditional Chinese surface forms. "
                 "A pass means the implemented check held, not that the "
                 "message is semantically correct."),
        "per_check": per_check,
        "all_checks_passed": joint,
        "all_checks_pass_rate": round(joint / len(checked), 4) if checked else None,
        "template_fallback_note": (
            "the fixed template satisfies this contract by construction: it "
            "emits only system-computed numbers, never writes kWh/MWh, states "
            "the recommended time explicitly and is phrased as a question. "
            "It is a design property, not a measurement, and it is why the "
            "LLM is presented as an interface option rather than a "
            "contribution."),
        "rows": [{"i": c.get("i"), "event_type": c.get("event_type"),
                  "house": c.get("house"), "contract": c["_contract"],
                  "message_zh": c.get("message_zh")} for c in checked],
    }
    dst = REPORTS / "llm_contract.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model={out['model']}  messages={out['n_messages']}")
    for n in names:
        p = per_check[n]
        if p["applicable"]:
            print(f"  {n:<22} {p['passed']:>4}/{p['applicable']:<4} = {p['rate']}")
    print(f"  {'ALL CHECKS':<22} {joint:>4}/{len(checked):<4} = "
          f"{out['all_checks_pass_rate']}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
