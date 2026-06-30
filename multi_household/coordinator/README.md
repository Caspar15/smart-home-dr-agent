# coordinator/ — optimization-based DR coordinator (管少棋 / guanchichi)

This is the **shadow-price coordination engine** contributed by co-author
**管少棋 (`guanchichi`)**: online rolling-horizon shadow-price coordination of
deferrable appliance schedules + a **day-ahead oracle MILP** upper bound, with a
strict two-layer **causal simulator**. It complements the main `multi_household/`
system (which provides the broadcast-rule coordination + LLM advisory).

> **Authorship:** the `phase*.py` files were authored by `guanchichi` (preserved
> via cherry-pick); the move into this folder is housekeeping only.

## Status

- ✅ **Verified** (2026-06-30): reproduced on our REFIT data — shadow-price
  coordination **−32% PAR**, **72% of the oracle**, 24-rule HARD-RULE self-check
  passes. See `decks_workspace/WEEKLY_PLAN.md §2.6`.
- 🔌 **Not yet wired into the main pipeline.** It runs as a **self-contained unit**.
  Integration (use our forecaster / our aligned-window community / feed the LLM
  advisor) is a deliberate later step.
- ⚠️ Uses a **synthetic community** (each house's own longest clean window aligned
  to a shared clock). The main system uses a **real time-aligned window** — the two
  community constructions are **not directly comparable**; pick one before reporting.

## How to run (self-contained — from THIS folder)

```bash
cd multi_household/coordinator

# 1. cycle extraction (needs REFIT CLEAN_House{N}.csv via --data-dir)
python phase1_cycles.py --data-dir <CLEAN_REFIT_dir> --out out --houses 3,8,20

# 2. train per-house baseload LSTMs  (writes out_phase2_17h/model_house*.pt)
python phase2_lstm.py --houses 3 8 20 --out_dir out_phase2_17h

# 3. shadow-price coordination demo (Greedy vs coord vs Oracle MILP, H3/8/20)
python phase4b_coordinator.py

# (4d community evaluation — synthetic community)
python phase4d_final.py
```

Spec lives in `PLAN.md`; hard rules in `CLAUDE.md`.

## Files

| File | Role |
|---|---|
| `phase1_cycles.py` | REFIT deferrable-cycle extraction (the "jobs") |
| `phase2_lstm.py` · `phase2_cnnlstm_compare.py` | per-house baseload LSTM (the simulator depends on these) |
| `phase3_simulator.py` | causal simulator (observe/forecast can't peek the future) |
| `phase4a_schedule.py` | single-house scheduling + must-run |
| `phase4b_coordinator.py` | **shadow-price coordination + oracle MILP** |
| `phase4c_rolling.py` | rolling horizon + commit-first (MPC) |
| `phase4d_*.py` | synthetic-community evaluation |
| `out_phase2_*/` · `results/` | scaler stats + committed results (models are gitignored) |
