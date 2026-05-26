# Single-Household Agent — Design (Phase 2)

How to set up and implement the single-household DR agent, mapped to the
architecture diagram: `Data → LSTM Forecast → State → Decision Core → Objective`.

---

## 0. The key constraint (read this first)

Our dataset is a **static historical log** — it records what happened, but has:
- **no electricity price**
- **no controllable actions**
- **no "what-if" feedback**

So we cannot learn control directly from the logs. We must build a
**simulation environment** (`src/agent/env.py`) where:
- historical load = the "natural demand"
- we add a **synthetic price** (Time-of-Use)
- the agent's **action reshapes the load** — reusing `src/dr/strategies.py`
- we compute **cost / peak / comfort** as feedback

> The DR Simulation block we already built **is** the environment's physics engine.

---

## 1. State  (`src/agent/state.py`)

The observation the agent sees each step:

| Field | Source |
|---|---|
| current_load | dataset row |
| forecast_next (peak risk) | **LSTM forecast** (Phase 1 output) |
| price_now | synthetic ToU schedule |
| hour_sin, hour_cos, is_weekend | time features |
| deferred_so_far | comfort budget (how much load already pushed forward) |
| (optional) battery_soc | if storage is modeled |

---

## 2. Action  (Decision Core, `src/agent/rule_based.py` → `rl_agent.py`)

Start **discrete** and simple, expand later:

- `0 = normal` (do nothing)
- `1 = defer` (push flexible load to a later off-peak slot)
- `2 = clip` (curtail the peak by X%)

Later: continuous action = "fraction of flexible load to shift".

Two implementations (in order):
1. **Rule-based / heuristic** — `if price high & forecast peak → defer/clip`.
   Deterministic, interpretable. **Build and validate this first.**
2. **RL agent** — PPO (continuous) or DQN (discrete), trained in the env, then
   compared against the rule-based baseline.

---

## 3. Objective / Reward  (`src/agent/reward.py`)

```
R_t = -( Cost_t + w1·Peak_t + w2·Discomfort_t + w3·Switching_t )
```
- **Cost** = load × price
- **Peak** = penalty for exceeding a load threshold
- **Discomfort** = deviation from the original (un-shifted) consumption
- **Switching** = penalty for frequent on/off toggling

(The global **Coordination** bonus is added in the multi-agent phase.)

---

## 4. Environment loop  (`src/agent/env.py`)

Gym-style:
```
obs = env.reset()
while not done:
    action = controller.decide(obs, env)     # rule-based or RL
    obs, reward, done, _, info = env.step(action)
```
`step()` = apply action → reshape load (DR transform) → compute reward → advance.

---

## 5. Implementation milestones

| Step | Build | Done when |
|---|---|---|
| 1a | `env.py` (reset/step + synthetic price) | one episode runs end-to-end |
| 1b | `state.py` + `reward.py` | state vector + reward return sensible values |
| 1c | `rule_based.py` | controller reduces cost & peak vs no-DR baseline |
| 1d | `rl_agent.py` (PPO/DQN) | RL ≥ rule-based on cost/peak (or explainably not) |
| 1e | evaluation | report cost saving %, peak reduction %, PAR, comfort |

---

## 6. Evaluation metrics (system level — 0508 slide 11)

- **Forecasting**: MAE / RMSE / R² (already have)
- **System**: Peak reduction %, PAR (peak-to-average ratio), load variance
- **Economics**: cost saving %
- **Human**: comfort index, fairness

---

## 7. Honesty checklist (state in every report)

- Price and action→load effect are **simulated** (dataset has neither).
- **Single household** only — no cross-household coordination yet.
- **Comfort** is a proxy (deviation from original load), not a measured value.
