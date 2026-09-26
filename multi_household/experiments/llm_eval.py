"""LLM interface characterization — numbers for the paper's LLM section.

Positioning (per AGENT_DESIGN decision A): the LLM is an INTERFACE-layer
application, not a contribution claim. This harness quantifies exactly what
the paper may state about it, on the real decision events of the headline
window:

  • JSON schema success rate      (parse + required keys, first attempt)
  • fact-citation precision       (numbers in the message that exist in facts)
  • hallucinated-number rate      (numbers in the message NOT in facts)
  • unit-error rate               (kWh/MWh where Wh expected — validate_units)
  • latency per call              (wall clock, local Ollama)

Input events = the decision-level events (new_defer / declined_defer /
ev_advisory) from rollout_coordinated_recs.json — i.e. the actual 102 user
decision points, not synthetic prompts.

Run:  python -m multi_household.experiments.llm_eval [--model llama3.1:8b]
      (requires a local Ollama server)
Writes: reports/multi_household/llm_eval_<model>.json
"""
from __future__ import annotations
import sys, json, time, re, argparse, urllib.request
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from multi_household.experiments.rollout import REPORTS
from multi_household.llm.advisor import OLLAMA_URL, validate_units

DECISION_TYPES = ("new_defer", "declined_defer", "ev_advisory")

SCHEMA = {
    "type": "object",
    "required": ["message_zh", "cited_numbers"],
    "properties": {
        "message_zh":    {"type": "string"},
        "cited_numbers": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM = """你是住戶能源顧問。任務:把「事實」JSON 包裝成 1-2 句給住戶的繁體中文**建議訊息**
(徵詢住戶是否接受這項調整;住戶尚未決定,不要寫成已接受/已拒絕的通知)。
規則:
1. 禁止編造輸入沒有的數字;你只能引用事實 JSON 裡的數字。
2. cited_numbers 列出你在訊息中用到的每一個數字(字串)。
3. 能量單位一律 Wh,金額一律英鎊(£),不得使用 kWh/MWh。
4. 直接輸出 JSON,不要 markdown。"""

_NUM = re.compile(r"\d+(?:\.\d+)?")


def _fact_numbers(facts: dict) -> set[str]:
    """All numeric literals present in the facts (with rounding variants).

    Hour facts additionally allow their 12-hour-clock rendering ("hour": 21 →
    「晚上 9 點」) and zero-padded form — a first version flagged those as
    hallucinations, which inflated the rate ~7× (34% → ~5%): the model was
    CORRECTLY converting the hour, not inventing numbers."""
    out: set[str] = set()
    for k, v in facts.items():
        if isinstance(v, str):
            # An index carried by a string fact is still a fact. Without this,
            # "appliance": "washing_machine_2" made every message that named
            # the appliance correctly look like it invented the number 2 —
            # 7 of 20 flagged messages were this false positive.
            out |= set(_NUM.findall(v))
            continue
        if isinstance(v, (int, float)):
            f = float(v)
            for s in (f"{f:g}", f"{f:.0f}", f"{f:.1f}", f"{f:.2f}", f"{f:.3f}"):
                out.add(s)
                if s.endswith(".0"):
                    out.add(s[:-2])
            if k == "hour":
                h = int(v)
                out.add(str(h % 12 or 12))       # 12-hour clock
                out.add(f"{h:02d}")              # zero-padded (22:00 → "22")
                out.add("00")                    # the ":00" minutes token
    return out


def _event_facts(r: dict) -> dict:
    """PRE-DECISION facts only. The user's eventual accept/reject outcome is
    deliberately EXCLUDED: in deployment the recommendation message is written
    BEFORE the user decides, so an eval whose facts contain the outcome would
    characterize post-hoc notifications, not the advisory interface the paper
    claims. (A first version included `accepted` — scope now corrected.)"""
    base = {"house_id": r["house_id"], "hour": r["hour"]}
    if r["event_type"] == "ev_advisory":
        # length in minutes appears in the body text ("... N min EV charge")
        m = re.search(r"(\d+) min", r.get("body", ""))
        base.update({"ev_charge_min": int(m.group(1)) if m else 240,
                     "kind": "ev_reschedule"})
    else:
        base.update({"appliance": r["appliance"].replace("appliance_", "")
                     .replace("_w", ""),
                     "saving_gbp": r["saving_gbp"],
                     "kind": "appliance_defer"})
    return base


def _call(model: str, facts: dict, timeout_s: int = 120) -> tuple[dict | None, float, str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "事實:\n```json\n"
             + json.dumps(facts, ensure_ascii=False) + "\n```\n直接輸出 JSON。"},
        ],
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.2, "seed": 20260719, "num_predict": 300},
    }
    t0 = time.time()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        content = obj["message"]["content"]
        try:
            return json.loads(content), dt, content
        except json.JSONDecodeError:
            return None, dt, content
    except Exception as e:                       # noqa: BLE001
        return None, time.time() - t0, f"<error: {e}>"


def _server_env(model: str) -> dict:
    """Which Ollama build and which model blob produced this run.

    The tag alone is not a specification. Two runs of this harness, both
    `llama3.1:8b`, both seed 20260719, gave grounded-message rates of 0.9839
    (July) and 0.9435 (September) while the blob digest was provably
    unchanged — so the difference came from the server, not the weights.
    Record both so a future reader can tell the cases apart.
    """
    env: dict = {"model_tag": model}
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5) as r:
            env["ollama_version"] = json.load(r).get("version")
    except Exception as exc:
        env["ollama_version"] = f"<unavailable: {exc}>"
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model:
                    env["model_digest"] = m.get("digest")
                    env["model_modified_at"] = m.get("modified_at")
                    env["quantization"] = (m.get("details") or {}).get("quantization_level")
                    env["parameter_size"] = (m.get("details") or {}).get("parameter_size")
                    break
    except Exception as exc:
        env["model_digest"] = f"<unavailable: {exc}>"
    return env


def _evaluate(model: str, events: list) -> tuple[dict, list]:
    """One full pass over `events`. Returns (summary, rows)."""
    rows = []
    n_ok = n_cited_ok = n_halluc_msg = n_unit_bad = 0
    n_transport_fail = n_schema_fail = n_retried = 0
    latencies = []
    for i, r in enumerate(events):
        facts = _event_facts(r)
        parsed, dt, raw = _call(model, facts)
        if raw.startswith("<error:"):
            # One documented retry on transport error/timeout (environmental,
            # e.g. GPU cold start) — retries are counted and disclosed.
            n_retried += 1
            parsed, dt, raw = _call(model, facts)
        latencies.append(dt)
        transport_ok = not raw.startswith("<error:")
        ok = (parsed is not None and isinstance(parsed.get("message_zh"), str)
              and isinstance(parsed.get("cited_numbers"), list))
        if not transport_ok:
            n_transport_fail += 1
        elif not ok:
            n_schema_fail += 1
        # facts + raw citations are stored per row so a third party can
        # re-verify every metric from this JSON alone.
        row = {"i": i, "event_type": r["event_type"], "house": r["house_id"],
               "facts": facts,
               "latency_s": round(dt, 2), "schema_ok": bool(ok)}
        if ok:
            n_ok += 1
            allowed = _fact_numbers(facts)
            msg_nums = set(_NUM.findall(parsed["message_zh"]))
            halluc = sorted(n for n in msg_nums if n not in allowed)
            cited = set()
            for c in parsed["cited_numbers"]:
                cited |= set(_NUM.findall(str(c)))
            cited_bad = sorted(n for n in cited if n not in allowed)
            units = validate_units(parsed["message_zh"], "Wh")
            row.update({"cited_numbers_raw": parsed["cited_numbers"],
                        "hallucinated_numbers": halluc,
                        "cited_not_in_facts": cited_bad,
                        "unit_issues": units,
                        "message_zh": parsed["message_zh"]})
            if not cited_bad:
                n_cited_ok += 1
            if halluc:
                n_halluc_msg += 1
            if units:
                n_unit_bad += 1
        else:
            row["raw"] = raw[:200]
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(events)} done "
                  f"(schema {n_ok}/{i+1}, halluc msgs {n_halluc_msg})", flush=True)

    lat = sorted(latencies)
    n = len(events)
    # Metric names are MESSAGE-LEVEL rates (share of messages), not
    # per-number precision — named accordingly. Transport failures (timeouts —
    # environmental) are separated from schema violations (model behaviour):
    # conflating them once reported "schema 91.1%" when every completed call
    # was schema-valid and all failures were 122 s timeouts.
    n_completed = n - n_transport_fail
    summary = {
        "model": model,
        "scope": ("pre-decision recommendation messages (facts exclude the "
                  "user's eventual accept/reject outcome)"),
        "n_events": n,
        "call_completion_rate": round(n_completed / n, 4) if n else None,
        "n_transport_failures": n_transport_fail,
        "n_retried_once": n_retried,
        "schema_valid_rate_of_completed": round(
            (n_completed - n_schema_fail) / n_completed, 4) if n_completed else None,
        "msg_all_citations_grounded_rate": round(n_cited_ok / n_ok, 4) if n_ok else None,
        "msg_with_ungrounded_number_rate": round(n_halluc_msg / n_ok, 4) if n_ok else None,
        "msg_with_unit_error_rate": round(n_unit_bad / n_ok, 4) if n_ok else None,
        "latency_s": {"mean": round(sum(lat) / n, 2) if n else None,
                      "p50": round(lat[n // 2], 2) if n else None,
                      "p95": round(lat[int(n * 0.95)] if n else 0, 2)},
        "note": ("interface characterization only — the LLM makes no control "
                 "decisions; grid results are attributed to the coordination "
                 "mechanism (see mechanism_decomposition.json)"),
    }
    return summary, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of events (debug)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the whole evaluation N times. Three repeats "
                         "in one session came back bit-identical, so at a fixed "
                         "seed the harness IS deterministic against a given "
                         "server. It is not stable across sessions: the same "
                         "model digest and seed gave 0.9839 in July and 0.9435 "
                         "in September. Repeats establish which of the two you "
                         "are looking at; `server_env` in the output records "
                         "the build and blob that produced it.")
    args = ap.parse_args()

    recs = json.loads((REPORTS / "rollout_coordinated_recs.json")
                      .read_text(encoding="utf-8"))
    events = [r for r in recs if r.get("event_type") in DECISION_TYPES]
    if args.limit:
        events = events[:args.limit]
    print(f"model={args.model}  events={len(events)}  repeats={args.repeats}")

    runs = []
    for k in range(args.repeats):
        if args.repeats > 1:
            print(f"--- repeat {k + 1}/{args.repeats}")
        summary, rows = _evaluate(args.model, events)
        runs.append({"summary": summary, "rows": rows})

    out = {"summary": runs[-1]["summary"], "rows": runs[-1]["rows"],
           "server_env": _server_env(args.model)}
    if args.repeats > 1:
        # Report the spread, not just the last pass — a single figure would
        # look more reproducible than the interface actually is.
        keys = ("call_completion_rate", "schema_valid_rate_of_completed",
                "msg_all_citations_grounded_rate",
                "msg_with_ungrounded_number_rate", "msg_with_unit_error_rate")
        spread = {}
        for key in keys:
            vals = [r["summary"][key] for r in runs
                    if r["summary"].get(key) is not None]
            if vals:
                spread[key] = {
                    "n_runs": len(vals), "values": vals,
                    "mean": round(sum(vals) / len(vals), 4),
                    "min": min(vals), "max": max(vals),
                }
        out["repeatability"] = {
            "n_repeats": args.repeats,
            "note": ("same model, same decoding options and seed; the spread "
                     "below is run-to-run variance of the local server, not "
                     "sampling over events"),
            "per_metric": spread,
        }
        out["summary"] = dict(out["summary"], n_repeats=args.repeats)
        out["all_runs"] = [r["summary"] for r in runs]

    safe = args.model.replace(":", "_").replace("/", "_")
    dst = REPORTS / f"llm_eval_{safe}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(out.get("repeatability", out["summary"]),
                     ensure_ascii=False, indent=2))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
