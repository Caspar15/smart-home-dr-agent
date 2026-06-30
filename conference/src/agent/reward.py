# -*- coding: utf-8 -*-
"""Reward for the single-household agent (deferrable-load MVP).

R_t = -( w_cost·cost + w_peak·peak + w_comfort·comfort + w_switch·switching )

  cost      = price · served_load                (electricity bill)
  peak      = max(0, served_load - peak_threshold)   (penalize peaks)
  comfort   = buffer_next                         (undelivered energy = discomfort)
  switching = |action - prev_action|              (penalize toggling)

All load terms are divided by `scale` (mean demand) so the components are O(1)
and the weights are interpretable.
"""
from __future__ import annotations


def compute_reward(served, price, buffer_next, action, prev_action, cfg, scale):
    cost = price * (served / scale)
    peak = max(0.0, (served - cfg.peak_threshold_wh) / scale)
    comfort = buffer_next / scale
    switching = abs(action - prev_action)
    r = -(cfg.w_cost * cost
          + cfg.w_peak * peak
          + cfg.w_comfort * comfort
          + cfg.w_switch * switching)
    return float(r), dict(cost=cost, peak=peak, comfort=comfort, switching=switching)
