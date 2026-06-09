# Smart-Home Multi-Agent Energy Management

End-to-end pipeline for residential demand response: **LSTM forecast →
single-household decision controller (rule / MPC) → LLM advisory**, on
the UCI Appliances Energy dataset. Built up from a reproduction of
*Durrani et al. (2025)*; now extended toward a multi-agent system.

- **Roadmap & progress** → [`ROADMAP.md`](ROADMAP.md)
- **Agent design** (env, controllers, advisory) → [`docs/agent_design.md`](docs/agent_design.md)
- **Literature comparison** → [`docs/literature.md`](docs/literature.md)
- **Historical artefacts** (reproduction report, paper outline) → [`docs/archive/`](docs/archive/)

## Pipeline at a glance

```
UCI Appliances (10-min, 19,735 rows)
   │
   ▼
[forecasting]  CNN-LSTM ensemble  →  cache/forecast_full.npz
   │
   ▼
[agent.env]    HouseholdEnv  (gym-style; ToU price; deferrable-load buffer)
   │
   ├─►  agent.rule_based   (heuristic: defer on peak price + forecast)
   ├─►  agent.mpc          (LP receding horizon, perfect-foresight upper bound)
   │
   ▼
[advisory]     compute_facts → qwen3:4b (schema-constrained) → validate
   │                                                         (src/agent/advisory.py)
   ▼
reports/agent_advisory.md   (Chinese advisory report)
```

## Project structure

```
reproduction/                  (project root — run commands from here)
├── src/
│   ├── config.py              paths, seeds, hyperparameters
│   ├── data/                  preprocess · eda
│   ├── forecasting/           classical · lstm · lstm_cnn (CNN-LSTM ensemble)
│   ├── dr/                    strategies (7 DR transforms)
│   ├── evaluation/            retrain · ensemble_v2 · smoothing · experiments
│   ├── viz/                   paper_figures · comparison_figures · experiment_figures
│   └── agent/                 env · state · reward · rule_based · mpc · advisory
├── experiments/               run_agent · llm_report (v0) · llm_advisory (v1)
├── docs/                      agent_design.md · literature.md · archive/
├── reports/                   agent_facts.json · agent_advisory.json · .md
├── results/                   metrics CSVs (E1–E6, comparison, pivots)
├── figures/                   paper figures (PNG)
├── slides/assets/             source slide figures (hand-made PNG)
├── cache/                     forecast / preds / split (gitignored, regenerable)
└── ROADMAP.md · README.md · requirements.txt · .gitignore
```

## How to run

All commands from the project root (so `src` is importable):

### Phase 1 — Forecasting + DR evaluation

```bash
python -m src.data.preprocess               # build + cache train/test split
python -m src.data.eda                      # EDA figures + Table 2
python -m src.forecasting.classical --fast  # LR / RF / SVR / kNN / baselines
python -m src.forecasting.lstm              # autoregressive LSTM (v1)
python -m src.forecasting.lstm_cnn          # CNN-LSTM ensemble (v2)
python -m src.evaluation.retrain --fast     # 8 models × 7 DR strategies
python -m src.evaluation.experiments        # E1–E6 paper experiment suite
python -m src.viz.paper_figures             # paper-style figures
```

### Phase 2 — Single-household agent

```bash
python -m src.agent.forecast                # precompute LSTM forecast cache
python -m experiments.run_agent             # rule-based vs MPC vs no-DR
```

### Phase 3 prelude — LLM advisory (NEW)

```bash
# Requires Ollama running locally with qwen3:4b
ollama pull qwen3:4b                        # one-time
python -m experiments.llm_advisory          # facts → schema-constrained LLM → validated md
```

Output: `reports/agent_advisory.md` (Chinese advisory report).

## Requirements

```
numpy pandas scikit-learn statsmodels matplotlib seaborn torch
```
Plus, for the LLM advisory layer: a local **Ollama** install with a chat
model (default `qwen3:4b`).

Verified on: Python 3.12 · PyTorch 2.5 + CUDA 12.1 · pandas 2.2 ·
scikit-learn 1.5 · Ollama 0.30.

## Key outputs

| Path | What |
|---|---|
| `results/metrics_retrain.csv`        | 8 models × 7 DR (long format) |
| `results/metrics_lstm_v2.csv`        | v2 ensemble, raw / MA6 / MA12 |
| `results/E1_…E6_*.csv`               | paper experiment suite |
| `figures/figF_agent_loadcurve.png`   | rule vs MPC vs no-DR load curves |
| `reports/agent_facts.json`           | structured facts fed to the LLM |
| `reports/agent_advisory.json`        | LLM schema-constrained output |
| `reports/agent_advisory.md`          | Chinese advisory report (human-facing) |
