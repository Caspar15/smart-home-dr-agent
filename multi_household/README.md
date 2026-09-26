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
   ▼ aggregator/        ev_coordinator: arrival-feasible EDF placement of the 5 EVs
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
│   └── ev_coordinator.py     EV advisory placement — EDF default, stagger/random baselines
├── agent/appliance_controller.py    the rule controller (defer/release/cooldown)
├── llm/advisor.py            Llama 3.1 personalized advice + validators + closed loop
├── experiments/              pre_cache · train_all · rollout · metrics · ablations
│                             · multiseed (error bars) · mpc_baseline (ladder bound)
│                             · fairness_sweep · daily_summary · personalized_demo · run_full
│                             · forecast_eval (LSTM vs persistence) · ev_factorial (360
│                             EV scenarios) · llm_contract (output contract, post-hoc)
│                             · paper_numbers (citable values + sources)
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
- [x] EV advisory coordinator (accept-gated, arrival-feasible EDF; stagger/random baselines)
- [x] Forecast accuracy artifact (`forecast_eval.py`) — CNN-LSTM 263.43 W vs
      persistence 192.22 W per house; LSTM wins 2/15
- [x] EV factorial (`ev_factorial.py`) — 360 paired scenarios over EV count,
      power, duration and arrival spread
- [x] 64 unit tests passing
- [x] Ablations on clean data (LSTM vs persistence, accept-rate sweep, seeded)
- [ ] Controller baselines (MPC/RL — reuse `conference/src/agent/`) — next
- [ ] Federated learning · Seq2Seq · MARL — future

## Validated numbers (v2 · 15 non-solar houses · exact 14.0-day noon-anchored test)

| Metric | Baseline | Independent | **Coordinated (85% accept)** |
|---|---|---|---|
| Peak (kW) | 40.37 | 34.91 (−13.5%) | **32.41 (−19.7%)** |
| P95 (kW) | 26.79 | 24.73 (−7.7%) | **18.65 (−30.4%)** |
| Energy (MWh) | 3.474 | 3.473 (−0.01%) | 3.473 (−0.01%) |
| User decisions | n/a | 162 appliance | **80 appliance (88.7% acc) + 44 EV (88.6%) = 124** |

Decision-level metrics only (trace messages counted separately). Full acceptance
(100%) → peak **26.84 kW / P95 18.11**. The EV reschedule is **advisory**
(accept-gated): P95 cut 0% / 7.7% / 30.4% / 32.4% at accept 0 / 50 / 85 / 100%.
Grid-vs-bill trade-off disclosed: EDF is grid-optimal but bills stay ~flat
(EVs placed at arrival-time tariffs); the stagger baseline saves 1.1–1.5% on
bills at −27.3% P95.

**Mechanism decomposition (factorial, `mechanism_decomposition.py`):** the EV
advisory coordinator ALONE gives −19.7% peak / −30.6% P95; the appliance layer
alone ~0; natural no-EV demand is 25.82 / 11.62 kW. The system is presented as a
**semi-synthetic REFIT + deterministic EV-adoption scenario**, decomposed openly.
Scheduling is replaceable (EDF > stagger > random, all arrival-feasible, paired
accept stream) — the contribution is the acceptance-gated advisory mechanism.

**External validity — EV factorial (360 paired scenarios, `ev_factorial.py`):**
3/5/10 EVs per night x 3.6/7.0 kW x 2/4 h x clustered (21-24 h) / dispersed
(17-24 h) arrivals, five load seeds x three accept seeds, p = 0.85. Mean paired
P95 reduction: random 10.34% | stagger 19.95% | **EDF 21.04%**. Every one of the
360 scenarios improves P95 under EDF and stagger (random makes 23 worse); EDF
raises the max in 14/360, worst case +22.38%. The effect grows with EV
penetration: P95 reduction 7.94% / 20.02% / **35.16%** at 3 / 5 / 10 EVs.
Building this experiment also surfaced a real defect: the `stagger` strategy's
rank offset was unbounded and broke the 8 h comfort cap at 10 EVs/night. Fixed
by clamping to `start + MAX_DEFER`; a no-op at the reference 5 EVs, where the
largest offset is exactly the cap, so all published numbers are unchanged.

**LLM interface: deterministic within a session, not across sessions.** Three
repeats in one process came back bit-identical (grounded-message rate 0.9435
x3), so at `temperature 0.2` and `seed 20260719` the harness is deterministic
against a given server. It is not stable over time: the committed July run
scored 0.9839 / ungrounded 0.0484 and the September re-run 0.9435 / 0.1613,
with the model blob digest provably unchanged
(`46e0c10c...`, modified 2026-06-27) — so the difference came from the Ollama
build, not the weights. `llm_eval.py --repeats N` (regen_all uses 3) reports
the spread and `server_env` records the build and digest. Cite the figure from
the artifact you actually shipped, and state the server version next to it;
the model tag alone does not pin the result down. `llm_contract.py` then re-scores those same saved
messages against the full output contract (number grounding, citation
coverage, units, explicit recommended time, consent wording) without any new
generation — the first four checks are mechanical, the last two are keyword
heuristics and a pass means only that the implemented check held.

**Rigor + baselines (v2, 2026-07-19):**
- Multi-seed (10 seeds incl. EV accept): 85% peak **27.94±2.46 kW**, P95 19.21±0.49.
  The fixed-EV-seed headline 32.41 sits at the high (conservative) end.
- Controller ladder: No-DR 40.37 | Rule@85% 32.41 (**50% of bound**) | Rule@100%
  26.84 (**85%**) | **MPC perfect-foresight bound 24.45 kW**.
- Grid threshold: train-window p85 = **17.7 kW**, frozen (`derive_threshold.py`);
  legacy 18 kW kept only as a sensitivity point.
- Data quality: mean NaN **0.50%** (worst house 1.21%); causal ffill ≤6 h only.
- Closed-loop stress: reject-all appliance history → 84 appliance decisions
  suppressed, P95 red 30.39→30.64% — the appliance layer's grid margin is ~0.
- Fairness (decision-level): Jain appliance 0.512 / EV 0.988 / total 0.635;
  a B=1 daily budget skips 22 decisions at zero grid cost (low event rate:
  ~0.6 decisions/house/day, so the budget binds only weakly).
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
