# -*- coding: utf-8 -*-
"""MPC (Model Predictive Control) decision core — LP receding horizon.

At each step we solve a small linear program over the next H steps:
  decide how much flexible load to defer / release so that
  total electricity cost + peak penalty is minimized,
  subject to the buffer dynamics and capacity.
Only the FIRST step's defer decision is executed (mapped to the env's discrete
action); next step we re-plan. This is the principled "optimal-given-forecast"
baseline that sits between the hand-crafted rule and a learned RL policy.

Forecast: perfect foresight over the horizon (actual demand) — labelled as the
UPPER BOUND of what planning can achieve. Price is the exact (known) ToU schedule.

Needs only scipy (already a dependency).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class MPCConfig:
    horizon: int = 18          # steps looked ahead (18 × 10 min = 3 h)
    lam_peak: float = 1.20     # weight on peak-excess in the LP objective
    lam_comfort: float = 0.05  # weight to drain the buffer (avoid hoarding)
    foresight: str = "perfect" # "perfect" = actual future demand


class MPCController:
    def __init__(self, cfg: MPCConfig | None = None):
        self.cfg = cfg or MPCConfig()

    def decide(self, state, env) -> int:
        cfg = self.cfg
        t = env.t
        H = min(cfg.horizon, env.t_end - t)
        if H <= 1:
            return 0
        ec = env.cfg
        d = env.demand[t:t + H].astype(float)                 # perfect foresight
        hours = env.hours[t:t + H]
        price = np.array([env._price_at(h) for h in hours], dtype=float)
        offpeak = np.array([env._is_offpeak(h) for h in hours], dtype=bool)
        b0 = float(env.buffer)
        phi = ec.flex_frac
        Bmax = ec.buffer_max
        rcap = ec.release_cap
        thr = ec.peak_threshold_wh

        # x = [defer(0..H-1), release(0..H-1), pk(0..H-1)]  -> 3H vars
        n = 3 * H
        c = np.zeros(n)
        for k in range(H):
            c[k] = -price[k] + cfg.lam_comfort          # defer
            c[H + k] = price[k] - cfg.lam_comfort        # release
            c[2 * H + k] = cfg.lam_peak                  # peak excess

        # bounds
        bounds = []
        for k in range(H):                               # defer
            bounds.append((0.0, max(0.0, phi * d[k])))
        for k in range(H):                               # release
            bounds.append((0.0, rcap if offpeak[k] else 0.0))
        for k in range(H):                               # pk
            bounds.append((0.0, None))

        A, bub = [], []
        # cumulative buffer upper / lower for each k
        for k in range(H):
            up = np.zeros(n); lo = np.zeros(n)
            for j in range(k + 1):
                up[j] = 1.0; up[H + j] = -1.0            # +defer -release <= Bmax-b0
                lo[j] = -1.0; lo[H + j] = 1.0            # -defer +release <= b0
            A.append(up); bub.append(Bmax - b0)
            A.append(lo); bub.append(b0)
        # peak: -defer[k] + release[k] - pk[k] <= thr - d[k]
        for k in range(H):
            row = np.zeros(n)
            row[k] = -1.0; row[H + k] = 1.0; row[2 * H + k] = -1.0
            A.append(row); bub.append(thr - d[k])

        res = linprog(c, A_ub=np.array(A), b_ub=np.array(bub), bounds=bounds,
                      method="highs")
        if not res.success:
            return 0
        defer0 = res.x[0]
        cap0 = phi * d[0]
        frac = defer0 / cap0 if cap0 > 1e-6 else 0.0
        if frac < 0.25:
            return 0
        if frac < 0.75:
            return 1
        return 2
