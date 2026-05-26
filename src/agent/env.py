# -*- coding: utf-8 -*-
"""Single-household demand-response environment (deferrable-load MVP).

Static log -> simulator. Each step:
  demand d_t is split into fixed (1-phi)·d_t and flexible phi·d_t.
  The agent defers a fraction of the flexible load into a buffer; the buffer is
  auto-released during off-peak hours. Served load = d_t - deferred + released.

Action (discrete 3):  0 = no defer, 1 = defer half flex, 2 = defer all flex
State:  src.agent.state.build_state
Reward: src.agent.reward.compute_reward

Design: docs/agent_design.md
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.config import CACHE, TARGET
from src.data.preprocess import build_dataset
from src.agent.state import build_state, STATE_DIM
from src.agent.reward import compute_reward

ACTION_DEFER_FRAC = {0: 0.0, 1: 0.5, 2: 1.0}


@dataclass
class EnvConfig:
    flex_frac: float = 0.30           # phi: shiftable share of each step's load
    buffer_max: float = 600.0         # Wh cap on deferred energy
    release_cap: float = 150.0        # Wh max released per off-peak step
    peak_threshold_wh: float = 200.0  # load above this is penalized
    # synthetic Time-of-Use price (relative units)
    price_peak: float = 0.30          # 17:00-22:00
    price_offpeak: float = 0.08       # 00:00-07:00
    price_mid: float = 0.15
    # reward weights
    w_cost: float = 1.0
    w_peak: float = 1.0
    w_comfort: float = 0.3
    w_switch: float = 0.05


class HouseholdEnv:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        split = build_dataset(verbose=False, split_mode="chronological",
                              feature_set="env_lags")
        df = split["df_clean"]
        self.demand = df[TARGET].to_numpy(dtype=float)
        self.hours = df["date"].dt.hour.to_numpy()
        self.n = len(df)
        fc = np.load(CACHE / "forecast_full.npz")
        self.forecast = fc["forecast"]
        self.test_start = int(fc["test_start"]); self.test_end = int(fc["test_end"])
        self.demand_scale = float(self.demand[:self.test_start].mean() or self.demand.mean())
        self.action_dim = 3
        self.state_dim = STATE_DIM
        self.reset()

    # -- price ---------------------------------------------------------------
    def _price_at(self, hour: int) -> float:
        if 17 <= hour < 22:
            return self.cfg.price_peak
        if 0 <= hour < 7:
            return self.cfg.price_offpeak
        return self.cfg.price_mid

    def _is_offpeak(self, hour: int) -> bool:
        return 0 <= hour < 7

    # -- gym-style API -------------------------------------------------------
    def reset(self, start_t: int | None = None, horizon: int | None = None):
        self.t = self.test_start if start_t is None else start_t
        self.t_end = self.test_end if horizon is None else min(self.t + horizon, self.n - 1)
        self.buffer = 0.0
        self.prev_action = 0
        return build_state(self, self.t)

    def step(self, action: int):
        cfg = self.cfg
        t = self.t
        d = self.demand[t]
        hour = self.hours[t]
        price = self._price_at(hour)

        # 1) release from buffer during off-peak (drain first)
        released = min(self.buffer, cfg.release_cap) if self._is_offpeak(hour) else 0.0
        buf = self.buffer - released
        # 2) defer flexible load (fill, capped by remaining buffer room)
        defer_frac = ACTION_DEFER_FRAC[int(action)]
        deferred = min(defer_frac * cfg.flex_frac * d, cfg.buffer_max - buf)
        buf = buf + deferred
        # 3) served load
        served = d - deferred + released

        r, parts = compute_reward(served, price, buf, action, self.prev_action,
                                  cfg, self.demand_scale)

        self.buffer = buf
        self.prev_action = int(action)
        self.t += 1
        terminated = self.t >= self.t_end
        info = dict(served=served, demand=d, price=price, released=released,
                    deferred=deferred, buffer=buf, hour=hour, **parts)
        obs = build_state(self, self.t) if not terminated else None
        return obs, r, terminated, False, info


if __name__ == "__main__":
    env = HouseholdEnv()
    obs = env.reset()
    print(f"[env] state_dim={env.state_dim} action_dim={env.action_dim} "
          f"test steps={env.t_end - env.t}  demand_scale={env.demand_scale:.1f}")
    # random rollout sanity check
    tot_r = 0.0; steps = 0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        obs, r, done, _, info = env.step(rng.integers(0, 3))
        tot_r += r; steps += 1
    print(f"[env] random rollout: {steps} steps, total reward={tot_r:.1f}")
