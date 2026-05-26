# -*- coding: utf-8 -*-
"""State construction for the single-household agent.

Observation vector at step t:
  [ demand_now, forecast_next, price_now, hour_sin, hour_cos, buffer_level ]
all normalized to ~[0,1] so the policy/heuristic sees comparable scales.
"""
from __future__ import annotations
import numpy as np

STATE_FIELDS = ["demand_now", "forecast_next", "price_now",
                "hour_sin", "hour_cos", "buffer_level"]
STATE_DIM = len(STATE_FIELDS)


def build_state(env, t: int) -> np.ndarray:
    d = env.demand[t] / env.demand_scale
    f = env.forecast[min(t + 1, env.n - 1)] / env.demand_scale
    hour = env.hours[t]
    price = env._price_at(hour) / env.cfg.price_peak
    hsin = np.sin(2 * np.pi * hour / 24)
    hcos = np.cos(2 * np.pi * hour / 24)
    buf = env.buffer / max(env.cfg.buffer_max, 1e-6)
    return np.array([d, f, price, hsin, hcos, buf], dtype=np.float32)
