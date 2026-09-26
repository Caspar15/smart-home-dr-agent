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
REFIT 15 non-solar UK houses (10-min, cleaned + time-aligned)
   │
   ▼ per-house CNN-LSTM            next-step / 24h baseload forecast
   ▼ aggregator (price broadcast)  sum forecasts → dynamic ToU + peak flag + hold-release
   ▼ appliance-aware controller    defer flexible cycles (washer…), comfort cap, off-peak drain
   ▼ EV advisory coordinator       arrival-feasible EDF placement of the 5 EVs (accept-gated)
   ▼ LLM advisory v2               facts → Llama 3.1 (local) → validate (no hallucinated units)
   ▼ closed-loop learning          accept/reject/modify → suppress rejected patterns
   ▼ evaluation                    peak / P95 / valley-fill / Jain fairness / energy conservation
```

**Validated result** (final freeze @6062100: 15 non-solar houses, noon-anchored
exact 14.0-day test, 85% accept, arrival-feasible EDF advisory coordinator,
train-derived threshold 17.7 kW): coordinated peak **40.37 → 32.41 kW (−19.7%)**,
P95 **26.79 → 18.65 kW (−30.4%)**; terminal buffer debt 1.78 kWh (−0.051%,
disclosed — not a conservation bug). Decision-level events: 80 appliance
(88.7% accepted) + 44 EV advisories (88.6%) = **124 user decisions** (trace
messages reported separately). Mechanism decomposition (factorial): the EV
advisory coordinator ALONE delivers −19.7% peak / −30.6% P95; the appliance
layer alone ~0. Scheduling is simple and replaceable (EDF beats the 2 h
stagger; random is useless) — the contribution is the acceptance-gated
advisory mechanism, with an honest grid-vs-bill trade-off reported (EDF grid-
optimal, near-zero bill saving; stagger bill-friendly, −27.3% P95). Multi-seed
(10 seeds): peak 27.94±2.46 kW. Cross-season P95 −28.5…−37.0%. 64 unit tests
pass. Provenance: `reports/multi_household/artifact_manifest.json`
(inputs/outputs/volatile SHA256); audit trail in `../AUDIT_2026-07-19.md`.

### How to run it

All commands from this directory (`reproduction/`):

```bash
# 0. one-time: local LLM (non-cloud, runs offline)
ollama pull llama3.1:8b

# 1. cache + train per-house forecasters (slow; writes multi_household/cache + models)
python -m multi_household.experiments.pre_cache
python -m multi_household.experiments.train_all

# 2. ⭐ ONE command → all formal artifacts + SHA256 manifest (paper numbers)
python -m multi_household.experiments.regen_all     # --skip-seasons / --skip-llm

# 2b. or individually: end-to-end rollout (baseline / independent / coordinated)
python -m multi_household.experiments.rollout --days 14 --mode all --user-accept 0.85

# 3. metrics + ablations
python -m multi_household.experiments.metrics          # incl. per-house NaN disclosure
python -m multi_household.experiments.ablations --days 14   # forecast / accept / closed-loop

# 3b. rigor + baselines
python -m multi_household.experiments.multiseed --days 14        # error bars (10 seeds)
python -m multi_household.experiments.mpc_baseline --days 14     # controller ladder (MPC bound)
python -m multi_household.experiments.fairness_sweep --days 14   # Jain vs shaving trade-off
python -m multi_household.experiments.forecast_eval              # CNN-LSTM vs persistence MAE
python -m multi_household.experiments.ev_factorial               # 360 EV scenarios (external validity)
python -m multi_household.experiments.peak_analysis              # PAR; what sets the peak, per acceptance seed
python -m multi_household.experiments.paper_numbers              # every citable number + its source

# 4. LLM advisory demos (House 7, a day in the test window 0–13)
python -m multi_household.experiments.llm_eval --repeats 3   # interface metrics + run-to-run spread
python -m multi_household.experiments.llm_contract          # output contract (post-hoc, no LLM calls)
python -m multi_household.experiments.llm_version_compare   # re-score the archived run under the same checks
python -m multi_household.experiments.personalized_demo --house 7 --day 5
python -m multi_household.experiments.daily_summary     --house 7 --day 5

# tests
python -m pytest multi_household/tests/ -q          # 64 tests
```

> **Cite numbers from `reports/multi_household/paper_numbers.md`, not from a
> draft.** It is generated straight from the result JSONs and prints the source
> file and JSON path next to every value. Hand-transcription has already cost us
> a headline P95 that was really the `coord_only` ablation row.

**Where results go** (not just the terminal — they persist as files):

| Output | Location |
|---|---|
| Rollout data | `reports/multi_household/rollout_*.npz / _recs.json / _waitlog.json` |
| Headline metrics | `reports/multi_household/metrics_summary.json` |
| Ablation | `reports/multi_household/ablation_results.json` + `figures/multi_household/ablation_*.png` |
| Multi-seed error bars | `reports/multi_household/multiseed_results.json` + `figures/multi_household/multiseed_accept.png` |
| Controller ladder (MPC bound) | `reports/multi_household/mpc_ladder.json` |
| Fairness trade-off | `reports/multi_household/fairness_sweep.json` + `figures/multi_household/fairness_tradeoff.png` |
| LLM advisory / closed loop | `reports/multi_household/personalized/`, `daily/`, `user_choices.json` |
| Forecast accuracy | `reports/multi_household/forecast_eval.json` |
| EV factorial (360 scenarios) | `reports/multi_household/ev_factorial.json` + `figures/multi_household/ev_factorial.png` |
| LLM output contract | `reports/multi_household/llm_contract.json` |
| LLM server-version comparison | `reports/multi_household/llm_version_compare.json` |
| Peak analysis (PAR, per-seed peak mechanism) | `reports/multi_household/peak_analysis.json` |
| **Paper numbers (cite from here)** | `reports/multi_household/paper_numbers.md` |
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
