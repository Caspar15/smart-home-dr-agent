# Multi-Household DR Coordination + LLM Advisory — Journal System

N-household demand response on the **REFIT** UK dataset (16 clean houses):
each home forecasts its own load, a central aggregator coordinates via a
broadcast price/peak signal, an appliance-aware controller shifts flexible
cycles, and an **LLM advisory layer** turns decisions into personalized,
hallucination-checked recommendations that the user can accept / reject /
modify — with the system **learning** from those choices.

> **Positioning:** this is a **human-in-the-loop advisory system**, not an
> automatic controller. The system *recommends*; the user decides; peak shaving
> depends on acceptance (the ablation shows 0% acceptance → 0% shaving).

## Pipeline

```
REFIT 16 houses (10-min, deglitched + clean-window + common-grid aligned)
   │
   ▼ forecasting/       per-house CNN-LSTM  (2 Conv1D + 2 LSTM, local-only)
   ▼ aggregator/        price_broadcast: Σ forecasts → dynamic ToU + peak flag
   │                    + hold-release (anti-rebound, broadcast to all houses)
   ▼ agent/             appliance_controller: rising-edge defer, per-appliance
   │                    cooldown, comfort cap (force-run), off-peak trickle drain
   ▼ llm/               advisor: facts → Llama 3.1 (local Ollama) → validate
   │                    (fact citations + kWh/MWh unit check) → personalized zh
   ▼ closed loop        accept/reject/modify → agent suppresses rejected patterns
   ▼ experiments/       rollout · metrics · ablations · daily_summary · demo
```

## Folder layout

```
multi_household/
├── config.py                 paths, clean-window, grid threshold, ToU prices
├── data/
│   ├── refit_loader.py       REFIT CSV → 10-min, deglitch, EV injection
│   ├── preprocess.py         clean window + common-grid reindex + features + split
│   └── appliance_map.py      per-house appliance dict + deferable classification
├── forecasting/per_house_lstm.py    CNN-LSTM (train-only scaler, chronological split)
├── aggregator/
│   ├── price_broadcast.py    aggregate + dynamic price + peak/hold-release
│   └── ev_coordinator.py     EV stagger scaffold (future, disabled)
├── agent/appliance_controller.py    the rule controller (defer/release/cooldown)
├── llm/advisor.py            Llama 3.1 personalized advice + validators + closed loop
├── experiments/              pre_cache · train_all · rollout · metrics · ablations
│                             · daily_summary · personalized_demo · run_full
└── tests/                    52 tests (energy conservation, cycle edge, cooldown, loop)
```

## Status — ✅ built & validated

- [x] REFIT loader + appliance map + EV injection (5 houses)
- [x] Data cleaning: deglitch (15 kW cap) · clean window (2014-04-30..07-14) · common-grid alignment
- [x] Per-house CNN-LSTM forecasters (train-only scaler, no leakage)
- [x] Aggregator price broadcast + peak detection + hold-release
- [x] Appliance-aware rule controller (rising-edge, cooldown, comfort cap, drain)
- [x] LLM advisory v2 — personalized, Llama 3.1 (local), fact-citation + unit validation
- [x] Closed-loop learning (accept/reject/modify → pattern suppression)
- [x] 52 unit tests passing
- [x] Ablations on clean data (LSTM vs persistence, accept-rate sweep, seeded)
- [ ] Controller baselines (MPC/RL — reuse `conference/src/agent/`) — next
- [ ] Federated learning · EV smart-charging coordinator · Seq2Seq · MARL — future

## Validated numbers (clean, time-aligned · 16 houses · 13.6-day test · energy 0% drift)

| Metric | Baseline | Independent | **Coordinated** |
|---|---|---|---|
| Peak (kW) | 40.50 | 38.69 (−4.5%) | **32.74 (−19.2%)** |
| P95 (kW) | 27.99 | 25.87 (−7.6%) | **24.70 (−11.8%)** |
| Rebound mean (kW) | 0 | +0.48 | **−2.06 (none)** |

**Ablation (seeded):** LSTM beats persistence on peak (32.7 vs 35.3 kW);
accept-rate 0% → 0% shaving, 85% → −12% P95 (quantifies the human-in-the-loop value).

## Data notes

- 16 clean houses: `1,2,3,4,5,6,7,8,9,10,13,15,16,17,18,20`.
  Excluded: 11/21 (rooftop solar → net load), 12 (no deferable), 14 (skipped in REFIT), 19 (1 deferable).
- 5 houses get a synthetic EV (7 kW, ~4 h nightly): 5, 7, 9, 13, 18 — they create the overnight peak.
- Test window is short (~14 days) because it's the intersection where all 16 houses
  are simultaneously clean+aligned. Stated as a limitation.

## Appliance classes

| Class | Examples | Behaviour |
|---|---|---|
| `deferable` | Washing Machine, Dishwasher, Tumble Dryer, Washer-Dryer, EV | Whole cycle shifted; runs uninterrupted; comfort cap 4–8 h |
| `semi_deferable` | Electric / Water Heater | Throttle, not shift far |
| `non_controllable` | Fridge, Freezer, Lighting, Cooking, TV | Never touched (comfort) |
