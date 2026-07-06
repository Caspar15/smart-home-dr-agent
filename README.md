# Smart-Home Demand-Response — Forecasting · Coordination · LLM Advisory

This repo holds **two related projects** on residential demand response (DR):

| Folder | Project | Status |
|---|---|---|
| **`multi_household/`** | **Journal extension** — N-household DR on the REFIT UK dataset: per-house forecast → grid coordination → appliance-level control → **LLM advisory + closed-loop learning**. | ✅ active, the current focus |
| **`conference/`** | **ISASD 2026 paper** — single-household forecasting (CNN-LSTM) + DR strategies + a single-household agent (rule / MPC) + v1 LLM advisory, on the UCI Appliances dataset. | ✅ published base (archived here) |

> Most current work is in **`multi_household/`**. The `conference/` tree is the
> already-written single-household paper, kept for reference and because its
> `conference/src/agent/{mpc,rl_agent,env}.py` are reused as **controller baselines** later.

---

## Repo layout

```
reproduction/                       ← run multi_household commands from here
├── multi_household/                ← THE journal system (see its own README)
│   ├── data/  forecasting/  aggregator/  agent/  llm/  experiments/  tests/
│   └── README.md                   ← how to run the system end-to-end
├── reports/multi_household/        ← results (JSON / npz)  ·  metrics_summary.json
├── figures/multi_household/        ← result figures (PNG)
├── conference/                     ← the ISASD single-household paper
│   ├── src/  experiments/  docs/  slides/  results/  figures/
│   └── (run conference code from inside conference/:  cd conference)
├── README.md  ROADMAP.md  requirements.txt  .gitignore
```

## The multi-household system (current focus)

```
REFIT 16 UK houses (10-min, cleaned + time-aligned)
   │
   ▼ per-house CNN-LSTM            next-step / 24h baseload forecast
   ▼ aggregator (price broadcast)  sum forecasts → dynamic ToU + peak flag + hold-release
   ▼ appliance-aware controller    defer flexible cycles (washer…), comfort cap, off-peak drain
   ▼ EV advisory coordinator       stagger the 5 EVs across the overnight trough (accept-gated)
   ▼ LLM advisory v2               facts → Llama 3.1 (local) → validate (no hallucinated units)
   ▼ closed-loop learning          accept/reject/modify → suppress rejected patterns
   ▼ evaluation                    peak / P95 / valley-fill / Jain fairness / energy conservation
```

**Validated result** (clean, time-aligned REFIT, 16 houses, ~14-day test, 85% accept):
coordinated peak **40.5 → 32.7 kW (−19%)**, P95 **27.99 → 19.99 kW (−29%)**,
**energy conserved** (0% drift); the EV reschedule is **advisory** (accept-gated) so
acceptance drives it — 0% → 0% shaving, 100% → **−29% peak (28.75 kW)**. 52 unit tests pass.

### How to run it

All commands from this directory (`reproduction/`):

```bash
# 0. one-time: local LLM (non-cloud, runs offline)
ollama pull llama3.1:8b

# 1. cache + train per-house forecasters (slow; writes multi_household/cache + models)
python -m multi_household.experiments.pre_cache
python -m multi_household.experiments.train_all

# 2. end-to-end rollout (baseline / independent / coordinated)
python -m multi_household.experiments.rollout --days 14 --mode all --user-accept 0.85

# 3. metrics + ablations
python -m multi_household.experiments.metrics
python -m multi_household.experiments.ablations --days 14

# 4. LLM advisory demos (House 7, a day in the test window 0–13)
python -m multi_household.experiments.personalized_demo --house 7 --day 5
python -m multi_household.experiments.daily_summary     --house 7 --day 5

# tests
python -m pytest multi_household/tests/ -q          # 52 tests
```

**Where results go** (not just the terminal — they persist as files):

| Output | Location |
|---|---|
| Rollout data | `reports/multi_household/rollout_*.npz / _recs.json / _waitlog.json` |
| Headline metrics | `reports/multi_household/metrics_summary.json` |
| Ablation | `reports/multi_household/ablation_results.json` + `figures/multi_household/ablation_*.png` |
| LLM advisory / closed loop | `reports/multi_household/personalized/`, `daily/`, `user_choices.json` |
| Figures | `figures/multi_household/*.png` |

## The conference system (archived)

The single-household ISASD paper code lives under `conference/` and runs from
**inside** that folder (its imports are `from src.…`):

```bash
cd conference
python -m src.forecasting.lstm_cnn      # CNN-LSTM ensemble
python -m experiments.run_agent         # rule-based vs MPC vs no-DR
```

## Requirements

```
numpy<2.0  pandas  scikit-learn  statsmodels  matplotlib  seaborn  torch
```
Plus a local **Ollama** for the LLM advisory layer (default `llama3.1:8b`).
Verified on Python 3.12 · PyTorch 2.5 + CUDA 12.1 · pandas 2.2 · numpy 1.26.

---

- **Roadmap** → [`ROADMAP.md`](ROADMAP.md)
- **System details / run guide** → [`multi_household/README.md`](multi_household/README.md)
- **Full status (honest: what's real vs demo)** → `../PROJECT_STATUS.md`
- **Weekly reporting plan** → `../../decks_workspace/WEEKLY_PLAN.md`
