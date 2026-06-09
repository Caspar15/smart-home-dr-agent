# Single-Household Agent — Design

How the single-household demand-response pipeline is set up and
implemented, mapped to the architecture:
`Data → LSTM Forecast → State → Decision Core → Objective → LLM Advisory`.

---

## 0. The key constraint (read this first)

Our dataset is a **static historical log** — it records what happened, but has:
- **no electricity price**
- **no controllable actions**
- **no "what-if" feedback**

So we cannot learn control directly from the logs. We built a
**simulation environment** (`src/agent/env.py`) where:
- historical load = the "natural demand"
- we add a **synthetic Time-of-Use price**
- the agent's **action reshapes the load** via a deferrable-load buffer
- we compute **cost / peak / comfort** as feedback

The DR-simulation block from Phase 1 (`src/dr/strategies.py`) is the
inspiration for the env's physics, though the env uses a simpler
buffer model (defer flexible fraction φ into buffer; release off-peak).

---

## 1. State  (`src/agent/state.py`)

The observation the agent sees each step (all normalised to ~[0,1]):

| Field | Source |
|---|---|
| demand_now | dataset row |
| forecast_next | **LSTM forecast** (Phase 1 output, cached) |
| price_now | synthetic ToU schedule |
| hour_sin, hour_cos | cyclic time features |
| buffer_level | how much load already deferred |

---

## 2. Action — Decision Core

Discrete 3-way action:
- `0 = serve normally`
- `1 = defer half of the flexible fraction`
- `2 = defer all of the flexible fraction`

The flexible fraction `φ` of each step's load is what the agent can shift;
the rest is non-deferrable. Deferred energy enters a buffer capped at
`buffer_max`, and is automatically released during off-peak hours up
to `release_cap` per step.

### 2.1 Rule-based controller  (`src/agent/rule_based.py`)

```
if buffer_full: return 0
if price ≥ peak_price AND forecast_next > peak_threshold: return 2
if price ≥ mid_price  AND forecast_next > peak_threshold: return 1
return 0
```

Deterministic and interpretable. Built first to validate the loop.

### 2.2 MPC controller  (`src/agent/mpc.py`)

Linear-program receding horizon, solved with `scipy.optimize.linprog`.
Horizon H = 18 steps (3 h); decision variables: defer / release / peak
excess per step. Uses **perfect foresight** of demand → this is the
**upper bound** of what planning-with-forecast can achieve. Solves in
~3 s on the full test window.

| | cost saving | peak-win avg | 95-pctile shave |
|---|---|---|---|
| rule-based | −2.6% | −5.6% | −10.3% |
| MPC (perfect foresight) | −2.8% | −4.5% | **−13.3%** |

MPC's advantage is concentrated on extreme peaks (the 95th-pctile shave),
not on average — exactly what a look-ahead policy should buy.

### 2.3 RL agent (deferred)

PPO/DQN against the same env. Will only be worth building if it can
beat MPC; otherwise MPC is the principled benchmark.

---

## 3. Reward  (`src/agent/reward.py`)

```
R_t = -( w_cost · cost_t
       + w_peak · peak_excess_t
       + w_comfort · |served - demand|_t
       + w_switch · switching_t )
```

- **cost** = served × price
- **peak_excess** = max(0, served − peak_threshold)
- **comfort** = deviation from the original (un-shifted) load
- **switch** = penalty for action toggling

Used by the env to compute step rewards. Rule-based and MPC don't need
the reward (they minimise their own cost directly), but it's there for
the RL path.

---

## 4. Environment loop  (`src/agent/env.py`)

Gym-style:
```python
obs = env.reset()
while not done:
    action = controller.decide(obs, env)   # rule_based, mpc, or RL
    obs, reward, done, _, info = env.step(action)
```

`step()` = release-from-buffer → defer-flexible-load → serve → compute
reward → advance. `info` exposes `served, demand, price, hour, released,
deferred, buffer`.

---

## 5. LLM Advisory Layer  (`src/agent/advisory.py`)

Sits **outside** the real-time control loop. After a finished agent run,
the advisory module:

1. **`compute_facts()`** — turns the run's traces into structured facts:
   - forecast quality: MAE / RMSE / R² overall + per hour-of-day
   - agent behaviour: action counts + defer-by-hour
   - peak events: shave threshold, n_peaks, n_shaved, worst unshaved hour
   - tomorrow outlook (from LSTM forecast tail, labelled as prediction)
2. **`build_prompt()`** — wraps the facts in a Chinese system prompt
   with explicit anti-hallucination rules.
3. **`OUTPUT_SCHEMA`** — strict JSON Schema fed to Ollama (`format=schema`)
   so qwen3:4b must fill exactly the slots we define. Controlled
   vocabulary enums for `tuning_suggestion.parameter`,
   `tuning_suggestion.direction`, `user_advisory.action_type`.
4. **`validate()`** — post-check that the LLM's `worst_hour`,
   `most_active_defer_hour`, and other numeric citations actually match
   the facts. Any mismatch is surfaced in the rendered report.
5. **`render_markdown()`** — pretty-print to a Chinese advisory `.md`.

Entry: `python -m experiments.llm_advisory` →
`reports/agent_advisory.{json,md}`.

**Design note:** the LLM is positioned as an analyst on top of a
deterministic controller, NOT as a replacement for the real-time
decision. Latency (qwen3:4b ~13 s per run), hallucination risk, and
MPC's mathematical optimality all preclude using the LLM inside the
control loop.

---

## 6. Implementation milestones

| Step | Build | Done |
|---|---|---|
| 1a | `env.py` reset/step + synthetic price | ✅ |
| 1b | `state.py` + `reward.py` | ✅ |
| 1c | `rule_based.py` | ✅ — cost −2.6%, peak-win −5.6% |
| 1d | `mpc.py` (LP receding horizon) | ✅ — peak-shave −13.3% |
| 1e | 3-way evaluation (`experiments/run_agent.py`) | ✅ |
| 1f | LLM advisory v1 (facts + schema + validate) | ✅ |
| 1g | RL agent (PPO/DQN) | ☐ deferred |

---

## 7. Evaluation metrics (system level)

- **Forecasting**: MAE / RMSE / R² (overall + per hour-of-day)
- **System**: peak-window avg, 95th-pctile shave, PAR
- **Economics**: cost saving %
- **Energy integrity**: served + buffer leftover = demand (must hold)
- **Advisory quality** (new): post-validator issue count

---

## 8. Honesty checklist (state in every report)

- Price and action→load effect are **simulated** (dataset has neither).
- **Single household** only — no cross-household coordination yet.
- **Comfort** is a proxy (deviation from original load), not measured.
- MPC results assume **perfect foresight** → upper bound, not realistic.
- LLM advisory is post-hoc analysis, not real-time control.
