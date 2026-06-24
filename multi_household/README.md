# Multi-Household DR Advisory — Journal Extension

Extending the single-household conference work to N households on the REFIT
UK dataset, with appliance-level load shifting and an LLM advisory layer.

## Scope (V1, no FL yet)

```
Training      Local-only:  each household trains its own CNN-LSTM
                            (no FL — that's a follow-up)
Coordination  Price broadcast: aggregator sums forecasts, sets dynamic
                            price + peak event flag, broadcasts to all
Decision      Appliance-aware rule controller: defer specific deferable
                            appliance cycles, not abstract 30% proxy
Advisory      LLM translates decisions to user-facing recommendations
                            (£ saved, suggested time, accept/reject)
```

## Folder layout

```
multi_household/
├── README.md                  this file
├── config.py                  paths + hyperparams
├── data/
│   ├── appliance_map.py       per-house appliance dict + deferable rules
│   └── refit_loader.py        CSV → 10-min DataFrame with split channels
├── forecasting/               per-house CNN-LSTM (Phase 1)
├── aggregator/                price broadcast logic (Phase 2)
├── agent/                     appliance-aware rule controller (Phase 2)
├── llm/                       advisory bridge (Phase 3)
└── experiments/               smoke tests + full rollouts
```

## What's done

- [x] REFIT archive structure inspected (20 houses, ~7M rows each, 6-8s)
- [x] Appliance map encoded
- [x] Loader with downsample + deferable/non-controllable split
- [ ] Per-house CNN-LSTM training
- [ ] Aggregator + agent + LLM bridge
- [ ] End-to-end 17-house rollout

## Data notes

- 20 REFIT houses available (numbered 1-21, skipping 14)
- House 11 + 21 have rooftop solar — aggregate is net load, not consumption.
  Excluded from the default "clean 17" set.
- House 12 has no deferable appliances — also excluded.
- Default 17 clean houses:
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20

## Appliance classification

| Class | Examples | Behaviour |
|---|---|---|
| `deferable` | Washing Machine, Dishwasher, Tumble Dryer, Washer Dryer | Whole cycle can be shifted, must complete uninterrupted |
| `semi_deferable` | Electric Heater, Water Heater | Can throttle but not shift far |
| `non_controllable` | Fridge, Freezer, Lighting, Cooking, TV, Computer | Always on / comfort, never deferred |

Deferable detection is based on appliance NAME string (see `data/appliance_map.py`).
