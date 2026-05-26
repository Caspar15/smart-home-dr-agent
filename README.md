# Smart-Home Multi-Agent Energy Management

Phase 1 reproduces *Durrani et al. (2025) — AI-driven optimization of energy
consumption and demand response in smart homes* on the UCI Appliances Energy
dataset. Phase 2 (in progress) builds a single-household demand-response agent.

- **Roadmap & progress:** [`ROADMAP.md`](ROADMAP.md)
- **Reproduction findings:** [`docs/REPORT.md`](docs/REPORT.md)
- **Literature comparison (verified):** [`docs/literature.md`](docs/literature.md)
- **Agent design (Phase 2):** [`docs/agent_design.md`](docs/agent_design.md)

## Project structure

```
reproduction/                  (project root — run commands from here)
├── src/
│   ├── config.py              paths, seeds, all hyperparameters
│   ├── data/      preprocess.py · eda.py
│   ├── forecasting/ classical.py · lstm.py · lstm_cnn.py     ← Phase 1 (predict)
│   ├── dr/        strategies.py  (7 DR transforms)
│   ├── evaluation/ retrain.py · ensemble_v2.py · smoothing.py
│   ├── viz/       paper_figures.py · comparison_figures.py
│   └── agent/     env.py · state.py · reward.py · rule_based.py · rl_agent.py  ← Phase 2 (decide)
├── reports/       build_0522.py  (deck generator)
├── docs/          REPORT.md · agent_design.md · literature.md
├── archive/       superseded one-off scripts
├── results/ figures/ cache/ slide_assets/   (outputs, generated)
└── ROADMAP.md · README.md · requirements.txt
```

## How to run

Run as modules **from the project root** (so `src` is importable):

```bash
cd reproduction
python -m src.data.preprocess          # build + cache train/test split
python -m src.data.eda                 # EDA figures + Table 2
python -m src.forecasting.classical --fast   # LR/RF/SVR/kNN + baselines
python -m src.forecasting.lstm         # autoregressive LSTM (v1)
python -m src.forecasting.lstm_cnn     # improved CNN-LSTM ensemble (v2)
python -m src.evaluation.retrain --fast      # metrics: 8 models × 7 DR
python -m src.evaluation.ensemble_v2         # v2 across DR + raw/MA6/MA12
python -m src.evaluation.smoothing           # smoothing-granularity experiment
python -m src.viz.paper_figures              # paper-style figures
python -m src.viz.comparison_figures         # comparison/slide figures
```

## Requirements

`numpy pandas scikit-learn statsmodels matplotlib seaborn torch`
(Phase 2 RL will additionally need `gymnasium stable-baselines3`.)

Verified on: Python 3.12 · PyTorch 2.5 + CUDA 12.1 · pandas 2.2 · scikit-learn 1.5.

## Key outputs

- `results/metrics_retrain.csv` — 8 models × 7 DR (long format)
- `results/metrics_lstm_v2.csv` — v2 ensemble, raw/MA6/MA12
- `results/comparison_overview.csv` — paper vs ours, every cell
- `figures/*.png`, `slide_assets/*.png` — all figures
