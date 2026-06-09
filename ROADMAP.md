# Roadmap — Smart-Home Multi-Agent Energy Management

Strategy: **build and validate ONE complete single-household agent first,
add an LLM advisory layer on top, then scale to multi-household, and
finally federated learning + global coordination.** (Build the single
unit before the system.)

Maps to the 0508 architecture: `Local Household Intelligence → Multiple
Household AI Agents Layer → Global Coordination + Federated Learning`.

---

## Phase 1 — Forecasting Baseline ✅ DONE

Reproduce Durrani et al. (2025) on the UCI Appliances Energy dataset.

- [x] Data preprocessing + feature engineering (lags, cyclical time, comfort indices)
- [x] 8 models: LR / RF / SVR / kNN / LSTM + Persistence / Seasonal-Naive / ETS
- [x] 7 DR strategies (rule-based load transforms, closed-form formulas)
- [x] Evaluation: MAE / RMSE / R² across 8 × 7 cells
- [x] **Key finding**: the paper's R²=0.94 comes from coarser evaluation
      granularity; honest 10-min R² caps ~0.6 (see `docs/archive/REPORT.md`)
- [x] **Improved CNN-LSTM (v2)**: cyclical feats + 4-seed ensemble →
      raw R²=0.64, hourly-eval ~0.91
- [x] E1–E6 paper experiment suite (granularity / split / features / arch)

**Code:** `src/data`, `src/forecasting`, `src/dr`, `src/evaluation`, `src/viz`

---

## Phase 2 — Single-Household Agent ✅ DONE

Turn "prediction" into "decision" on ONE household. Close the loop:
`forecast → state → decision → DR action → reward`.

Because the dataset is a static log (no price, no actions, no feedback),
we first build a **simulator** (`src/agent/env.py`) whose "physics" is
the DR transforms we already have.

- [x] **Step 1a — Environment** (`src/agent/env.py`): Gym-style
      `reset/step`, synthetic ToU price, deferrable-load dynamics,
      discrete 3 actions
- [x] **State** (`src/agent/state.py`): demand + precomputed LSTM
      forecast + price + cyclical time + buffer level
- [x] **Reward** (`src/agent/reward.py`): `-(cost + w1·peak + w2·comfort + w3·switching)`
- [x] **Step 1b — Rule-based controller** (`src/agent/rule_based.py`):
      defer on high price + forecast peak, auto-release off-peak
- [x] **Step 1c — MPC baseline** (`src/agent/mpc.py`): LP receding-horizon
      (perfect-foresight upper bound). Beats the rule on peak-shaving:
      95th-pctile −13.3% vs −10.3%, cost −2.8% vs −2.6%
- [x] **Evaluation** (`experiments/run_agent.py`): 3-way baseline / rule / MPC
- [ ] **Step 1d — RL agent** (`src/agent/rl_agent.py`): PPO/DQN in the
      env, must beat MPC to be worth it  ◀ optional, deferred

**Forecast cache:** `src/agent/forecast.py` → `cache/forecast_full.npz`
**Load curve:** `figures/figF_agent_loadcurve.png`

**Honest caveats:** the price and the action→load effect are *simulated*
(the dataset has neither); single household only (no coordination yet);
comfort is a proxy (deviation from original load).

---

## Phase 3 prelude — LLM Advisory Layer ✅ DONE (v1)

Add a natural-language analyst on top of the deterministic controller.
The LLM is NOT in the real-time control loop — it reads structured facts
from a finished run and produces a Chinese advisory report.

- [x] **v0 metrics report** (`experiments/llm_report.py`): metrics →
      qwen3:4b → Chinese summary. Quick demo path.
- [x] **v1 advisory** (`experiments/llm_advisory.py` + `src/agent/advisory.py`):
      structured facts (forecast MAE per hour, action distribution, peak
      events, tomorrow outlook) → Ollama with JSON-Schema-constrained
      output → post-validator → markdown.
- [x] Hallucination control:
      A-class (input) — pre-computed numbers, LLM can't invent.
      B-class (reasoning) — schema enums force controlled vocabulary.
      C-class (output) — post-validator checks claimed numbers vs facts.
- [ ] **Next polish** — extend `validate()` with domain rules (e.g.
      shave_rate < 50% blocks `threshold → increase`); compare qwen3:4b
      vs 8b on advisory quality.

**Outputs:** `reports/agent_facts.json`, `reports/agent_advisory.{json,md}`

---

## Phase 3 — Multi-Household Coordination  ◀ NEXT

- [ ] **Step 2a — Multi-household env**: replicate the validated
      single-household env to N households (each its own LSTM forecast +
      buffer + controller). Add community-level peak, Jain-fairness, and
      synchronous-rebound diagnostics.
- [ ] **Step 2b — Data**: ingest Low Carbon London (5,567 UK homes,
      half-hourly, includes 1,100-household **real dToU tariff trial** —
      lets us drop the synthetic price assumption).
- [ ] **Step 2c — Coordination mechanism**: compare 2-3 options —
      central price-signal aggregator (greedy) vs. game-theoretic /
      consensus vs. learned coordinator.
- [ ] **Step 2d — LLM community advisor**: aggregate-level advisory
      (which hours, which cohorts, which interventions).

---

## Phase 4 — Federated + Global  (later)

- [ ] **Step 3a — Federated learning**: train per-household LSTMs
      federated across cohorts; preserve privacy.
- [ ] **Step 3b — Global coordination layer**: grid-scenario evaluation;
      Workflow vs Multi-Agent vs Hybrid; LLM as system-level explainer.

---

## Status snapshot

| Layer (0508 architecture) | Status |
|---|---|
| Data | ✅ done |
| LSTM Forecast | ✅ done (CNN-LSTM v2 ensemble) |
| DR Simulation | ✅ done (rule-based env physics + 7 strategies) |
| Single-household decision core | ✅ done (rule-based + MPC; RL deferred) |
| LLM advisory (single household) | ✅ done (v1, schema-constrained + validated) |
| Multi-household agent layer | ☐ next |
| Global coordination + Federated | ☐ later |

## This week (2026-06-09)

- ISASD 2026 paper polished — final title decided by advisor:
  *AI-Agent-Driven Demand Response Forecasting for Smart Homes with a
  CNN-LSTM Framework* (12 pages, Springer LLNCS, compiles cleanly).
- Added LLM advisory layer (qwen3:4b + JSON Schema + post-validator).
- Reorganised reproduction repo (slides out of root, stale docs to
  archive, .gitignore updated).
