# -*- coding: utf-8 -*-
"""Run the rule-based single-household agent, compute structured facts,
hand them to qwen3:4b with a strict JSON schema, validate, and render.

Run:   python -m experiments.llm_advisory
Out:   reports/agent_facts.json
       reports/agent_advisory.json
       reports/agent_advisory.md
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

from src.agent.env import HouseholdEnv, EnvConfig
from src.agent.rule_based import decide as rule_decide
from src.agent.advisory import (
    compute_facts, build_prompt, OUTPUT_SCHEMA, validate, render_markdown
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


# ---------------------------------------------------------------------------
# 1. agent rollout that ALSO tracks actions and forecast values
# ---------------------------------------------------------------------------

def rollout_with_trace(env, policy):
    obs = env.reset()
    served, demand, hours, actions, fc_pred = [], [], [], [], []
    done = False
    while not done:
        t = env.t                                # capture BEFORE step advances
        a = policy(obs, env)
        # forecast that was used for THIS step's decision
        fc_pred.append(float(env.forecast[min(t + 1, env.n - 1)]))
        obs, _, done, _, info = env.step(a)
        actions.append(int(a))
        served.append(info["served"])
        demand.append(info["demand"])
        hours.append(info["hour"])
    return (np.array(served), np.array(demand), np.array(hours),
            actions, np.array(fc_pred))


# ---------------------------------------------------------------------------
# 2. Ollama with schema-constrained output
# ---------------------------------------------------------------------------

def call_ollama_json(system: str, user: str, schema: dict,
                     model: str = MODEL,
                     temperature: float = 0.1,
                     num_predict: int = 2000) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": schema,           # Ollama 0.5+ structured output
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": 0.9,
            "think": False,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=420) as resp:
        raw = resp.read().decode("utf-8")
    dt = time.time() - t0
    obj = json.loads(raw)
    content = obj["message"]["content"]

    # robustness: strip any <think>...</think> the model may still emit
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].lstrip()

    # parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        # try to find the outermost { ... } block
        a, b = content.find("{"), content.rfind("}")
        if a >= 0 and b > a:
            parsed = json.loads(content[a:b + 1])
        else:
            raise RuntimeError(
                f"LLM did not return valid JSON. Raw:\n{content[:500]}") from e

    return {"parsed": parsed, "elapsed_s": dt, "raw_content": content}


# ---------------------------------------------------------------------------
# 3. glue
# ---------------------------------------------------------------------------

def main():
    print("[1/4] rolling out rule-based agent ...")
    env = HouseholdEnv(EnvConfig())
    served, demand, hours, actions, fc_pred = rollout_with_trace(
        env, lambda obs, e: rule_decide(obs, e))
    print(f"      {len(actions)} steps, "
          f"actions: 0={actions.count(0)} 1={actions.count(1)} 2={actions.count(2)}")

    print("\n[2/4] computing structured facts ...")
    facts = compute_facts(env, served, demand, hours, actions, fc_pred)
    facts_path = REPORTS / "agent_facts.json"
    facts_path.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      facts -> {facts_path}")
    fq = facts["forecast_quality"]
    print(f"      forecast: MAE={fq['mae_overall_wh']} Wh, R²={fq['r2_overall']}, "
          f"worst hour {fq['worst_hour']:02d}")
    pe = facts["peak_events"]
    print(f"      peaks: {pe['n_peak_steps']} total, "
          f"{pe['n_peak_steps_shaved']} shaved ({pe['shave_rate']*100:.1f}%)")

    print(f"\n[3/4] calling Ollama ({MODEL}) with schema-constrained output ...")
    system, user = build_prompt(facts)
    try:
        result = call_ollama_json(system, user, OUTPUT_SCHEMA)
    except urllib.error.URLError as e:
        print(f"  !! Ollama not reachable: {e}")
        return
    except Exception as e:
        print(f"  !! LLM call failed: {e}")
        return

    print(f"      elapsed {result['elapsed_s']:.1f}s, "
          f"raw chars {len(result['raw_content'])}")

    parsed = result["parsed"]
    issues = validate(parsed, facts)
    if issues:
        print(f"      ⚠️  {len(issues)} validation issue(s):")
        for i in issues:
            print(f"        - {i}")
    else:
        print("      ✅ validation passed")

    print("\n[4/4] rendering markdown ...")
    md = render_markdown(parsed, facts,
                         meta={"model": MODEL,
                               "elapsed_s": result["elapsed_s"],
                               "issues": issues})

    (REPORTS / "agent_advisory.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "agent_advisory.md").write_text(md, encoding="utf-8")
    print(f"      json -> {REPORTS / 'agent_advisory.json'}")
    print(f"      md   -> {REPORTS / 'agent_advisory.md'}")

    print("\n" + "=" * 60)
    print(md)


if __name__ == "__main__":
    main()
