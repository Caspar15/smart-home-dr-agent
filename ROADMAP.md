# Roadmap — Smart-Home Multi-Agent Energy Management

Strategy: **build and validate ONE complete single-household agent first, then
scale to multi-agent, then add federated learning + global coordination.**
(Build the single unit before the system.)

Maps to the 0508 architecture: `Local Household Intelligence → Multiple Household
AI Agents Layer → Global Coordination + Federated Learning`.

---

## Phase 1 — Forecasting Baseline ✅ DONE

Reproduce Durrani et al. (2025) on the UCI Appliances Energy dataset.

- [x] Data preprocessing + feature engineering (lags, cyclical time, comfort indices)
- [x] 8 models: LR / RF / SVR / kNN / LSTM + Persistence / Seasonal-Naive / ETS
- [x] 7 DR strategies (rule-based load transforms)
- [x] Evaluation: MAE / RMSE / R² across 8 × 7 cells
- [x] **Key finding**: the paper's R²=0.94 comes from coarser evaluation
      granularity; honest 10-min R² caps ~0.6 (see docs/literature.md)
- [x] **Improved LSTM (v2)**: CNN-LSTM + cyclical feats + ensemble → raw R²=0.64,
      hourly-eval ~0.91

**Code:** `src/data`, `src/forecasting`, `src/dr`, `src/evaluation`, `src/viz`

---

## Phase 2 — Single-Household Agent  ◀ Step 1 DONE (rule-based loop validated)

Turn "prediction" into "decision" on ONE household. Close the loop:
`forecast → state → decision → DR action → reward`.

Because the dataset is a static log (no price, no actions, no feedback), we
first build a **simulator** (`src/agent/env.py`) whose "physics" is the DR
transforms we already have.

- [x] **Step 1a — Environment** (`src/agent/env.py`): Gym-style `reset/step`,
      synthetic ToU price, deferrable-load dynamics, discrete 3 actions ✅
- [x] **State** (`src/agent/state.py`): demand + precomputed LSTM forecast +
      price + cyclical time + buffer level ✅
- [x] **Reward** (`src/agent/reward.py`): `-(cost + w1·peak + w2·comfort + w3·switching)` ✅
- [x] **Step 1b — Rule-based controller** (`src/agent/rule_based.py`): defer on
      high price + forecast peak, auto-release off-peak. **Loop validated.** ✅
- [x] **Evaluation** (`experiments/run_agent.py`): baseline vs agent — cost
      −2.6%, peak-window −5.6%, 95th-pctile load −10.3%, energy conserved ✅
- [ ] **Step 1c — RL agent** (`src/agent/rl_agent.py`): PPO/DQN in the env,
      compare to rule-based  ◀ next

**Forecast cache:** `src/agent/forecast.py` → `cache/forecast_full.npz`
**Load curve:** `figures/figF_agent_loadcurve.png`

**Honest caveats to state in reports:** the price and the action→load effect
are *simulated* (the dataset has neither); single household only (no
coordination yet); comfort is a proxy (deviation from original load).

---

## Phase 3 — Multi-Agent + Federated  (later)

- [ ] **Step 2 — Multi-agent**: replicate the validated single agent to N
      households; add coordination to avoid synchronized off-peak rebound (new peaks)
- [ ] **Step 3 — Federated + global**: federated learning across households;
      global grid-coordination layer; Workflow vs Multi-Agent vs Hybrid; grid-scenario eval

---

## Status snapshot

| Layer (0508 architecture) | Status |
|---|---|
| Data | ✅ done |
| LSTM Forecast | ✅ done |
| DR Simulation | ✅ rule-based (env physics) |
| State / Decision Core / Objective | ◑ rule-based done; RL next |
| Multiple-agent layer | ☐ Phase 3 |
| Global coordination + Federated | ☐ Phase 3 |
