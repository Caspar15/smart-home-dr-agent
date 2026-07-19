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
REFIT 15 non-solar houses (10-min, deglitched + clean-window + common-grid aligned)
   │
   ▼ forecasting/       per-house CNN-LSTM  (2 Conv1D + 2 LSTM, local-only)
   ▼ aggregator/        price_broadcast: Σ forecasts → dynamic ToU + peak flag
   │                    + hold-release (anti-rebound, broadcast to all houses)
   ▼ agent/             appliance_controller: rising-edge defer, per-appliance
   │                    cooldown, comfort cap (force-run), off-peak trickle drain
   ▼ aggregator/        ev_coordinator: stagger the 5 EVs across the overnight trough
   │                    — ADVISORY (accept-gated), the dominant peak lever
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
│   └── ev_coordinator.py     EV advisory stagger (accept-gated; the big peak lever)
├── agent/appliance_controller.py    the rule controller (defer/release/cooldown)
├── llm/advisor.py            Llama 3.1 personalized advice + validators + closed loop
├── experiments/              pre_cache · train_all · rollout · metrics · ablations
│                             · multiseed (error bars) · mpc_baseline (ladder bound)
│                             · fairness_sweep · daily_summary · personalized_demo · run_full
└── tests/                    64 tests (data causality, energy conservation, cycle edge, cooldown, loop, EV advisory)
```

## Status — ✅ built & validated

- [x] REFIT loader + appliance map + EV injection (5 houses)
- [x] Data cleaning: deglitch (15 kW cap) · clean window (2014-04-30..07-14) · common-grid alignment
- [x] Per-house CNN-LSTM forecasters (train-only scaler, no leakage)
- [x] Aggregator price broadcast + peak detection + hold-release
- [x] Appliance-aware rule controller (rising-edge, cooldown, comfort cap, drain)
- [x] LLM advisory v2 — personalized, Llama 3.1 (local), fact-citation + unit validation
- [x] Closed-loop learning (accept/reject/modify → pattern suppression)
- [x] EV advisory coordinator (accept-gated stagger of the 5 EVs → the big peak lever)
- [x] 64 unit tests passing
- [x] Ablations on clean data (LSTM vs persistence, accept-rate sweep, seeded)
- [ ] Controller baselines (MPC/RL — reuse `conference/src/agent/`) — next
- [ ] Federated learning · Seq2Seq · MARL — future

## Validated numbers (v2 · 15 non-solar houses · exact 14.0-day noon-anchored test)

| Metric | Baseline | Independent | **Coordinated (85% accept)** |
|---|---|---|---|
| Peak (kW) | 40.37 | 35.56 (−11.9%) | **32.41 (−19.7%)** |
| P95 (kW) | 26.79 | 25.06 (−6.5%) | **19.44 (−27.4%)** |
| Energy (MWh) | 3.474 | 3.473 (−0.01%) | 3.473 (−0.01%) |
| User decisions | n/a | 148 appliance | **58 appliance (91.4% acc) + 44 EV (88.6%) = 102** |

Decision-level metrics only (trace messages counted separately). Full acceptance
(100%) → peak **28.53 kW**. The EV reschedule is **advisory** (accept-gated):
P95 cut 0% / 16.7% / 27.5% / 28.6% at accept 0 / 50 / 85 / 100%.

**Mechanism decomposition (factorial, `mechanism_decomposition.py`):** the EV
advisory stagger ALONE gives −19.7% peak / −27.2% P95; the appliance layer alone
~0.2pp; natural no-EV demand is 25.82 / 11.62 kW. The system is presented as a
**semi-synthetic REFIT + deterministic EV-adoption scenario**, decomposed openly.

**Rigor + baselines (v2, 2026-07-19):**
- Multi-seed (5 seeds incl. EV accept): 85% peak **29.40±2.77 kW**, P95 19.99±0.35.
  The fixed-EV-seed headline 32.41 sits at the high (conservative) end.
- Controller ladder: No-DR 40.37 | Rule@85% 32.41 (**50% of bound**) | Rule@100%
  28.53 (**74%**) | **MPC perfect-foresight bound 24.45 kW**.
- Grid threshold: train-window p85 = **17.7 kW**, frozen (`derive_threshold.py`);
  legacy 18 kW kept only as a sensitivity point.
- Data quality: mean NaN **0.50%** (worst house 1.21%); causal ffill ≤6 h only.
- Closed-loop stress: reject-all appliance history → appliance decisions
  suppressed (61), P95 27.45→27.27% — the appliance layer contributes +0.2pp.
- Fairness (decision-level): Jain appliance 0.384 / EV 0.988 / total 0.509;
  a B=1 daily budget skips 15 decisions at zero grid cost (low event rate:
  ~0.5 decisions/house/day, so the budget binds only weakly).
- Forecast honesty: CNN-LSTM one-step MAE **loses to persistence** (263 vs
  192 W per-house; wins 2/15) — DR results are insensitive to this (the EV
  coordinator uses no forecast); the LSTM is NOT a claimed contribution.

## Data notes

- 15 clean non-solar houses: `1,2,4,5,6,7,8,9,10,13,15,16,17,18,20`.
  Excluded: 3/11/21 (solar PV interferes with the aggregate — the non-directional
  clamp shows generation as additional positive consumption; per the REFIT paper;
  houses 1/6/7 also had PV but were re-wired by the REFIT team), 12 (no
  deferable), 14 (skipped in REFIT), 19 (1 deferable). H12/H19 exclusion is a
  DR-potential selection choice — an all-non-solar sensitivity is reported.
- 5 houses get a synthetic EV (7 kW, ~4 h nightly): 5, 7, 9, 13, 18 — they create
  the overnight peak (a deterministic adoption scenario, not calibrated).
- Test window: 2014-06-30 12:00 → 07-14 12:00 (noon-anchored, exactly 2016
  steps). Seasonal replicas use the same convention (spring is a 12-day window —
  the only Mar–May span where every house passes the ≤6 h gap bar).

## Appliance classes

| Class | Examples | Behaviour |
|---|---|---|
| `deferable` | Washing Machine, Dishwasher, Tumble Dryer, Washer-Dryer, EV | **Energy-buffer relaxation**: the running cycle's energy is banked step-by-step and re-released as a smoothed off-peak drain (pool/60 per step). Energy is conserved but the cycle waveform is NOT preserved — this is a virtual load-shifting lower bound, not an appliance-feasible schedule. Comfort cap 4–8 h. (EV blocks handled by the advisory coordinator ARE moved as whole blocks.) |
| `semi_deferable` | Electric / Water Heater | Throttle, not shift far |
| `non_controllable` | Fridge, Freezer, Lighting, Cooking, TV | Never touched (comfort) |
