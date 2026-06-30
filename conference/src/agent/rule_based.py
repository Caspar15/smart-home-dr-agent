# -*- coding: utf-8 -*-
"""Rule-based decision core (Phase 2, Step 1 — the FIRST controller).

Heuristic: defer flexible load when electricity is expensive AND the forecast
indicates a peak; otherwise serve normally (the buffer auto-releases off-peak).
Deterministic and interpretable — used to validate the full loop before RL.
"""
from __future__ import annotations


def decide(state, env) -> int:
    t = env.t
    hour = env.hours[t]
    price = env._price_at(hour)
    fc_next = env.forecast[min(t + 1, env.n - 1)]
    thr = env.cfg.peak_threshold_wh
    room = env.cfg.buffer_max - env.buffer

    if room <= 1e-6:
        return 0                      # buffer full -> cannot defer
    if price >= env.cfg.price_peak and fc_next > thr:
        return 2                      # peak price + expected peak -> defer all
    if price >= env.cfg.price_mid and fc_next > thr:
        return 1                      # mid price + expected peak -> defer half
    return 0                          # otherwise serve normally
