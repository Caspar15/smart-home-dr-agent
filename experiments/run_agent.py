# -*- coding: utf-8 -*-
"""Run the single-household agent over the test period and compare to the
no-DR baseline. Reports cost / peak / PAR / comfort and plots the load curve.

Run:  python -m experiments.run_agent
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agent.env import HouseholdEnv, EnvConfig
from src.agent.rule_based import decide
from src.agent.mpc import MPCController

FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS.mkdir(exist_ok=True)
NAVY = "#124163"; ACCENT = "#E07A5F"; GREY = "#9AA7AD"; GREEN = "#75BDA7"


def rollout(env, policy):
    """policy(env) -> action, or None for the baseline (always 0)."""
    obs = env.reset()
    served, demand, price, hours = [], [], [], []
    done = False
    while not done:
        a = 0 if policy is None else policy(obs, env)
        obs, r, done, _, info = env.step(a)
        served.append(info["served"]); demand.append(info["demand"])
        price.append(info["price"]); hours.append(info["hour"])
    return (np.array(served), np.array(demand), np.array(price),
            np.array(hours), env.buffer)


def metrics(served, demand, price, hours, leftover):
    cost_b = float((price * demand).sum())
    cost_a = float((price * served).sum())
    peak_win = (hours >= 17) & (hours < 22)        # evening peak window
    pw_b = float(demand[peak_win].mean())
    pw_a = float(served[peak_win].mean())
    p95_b = float(np.percentile(demand, 95))
    p95_a = float(np.percentile(served, 95))
    return dict(
        cost_baseline=cost_b, cost_agent=cost_a,
        cost_saving_pct=100 * (cost_b - cost_a) / cost_b,
        peakwin_baseline=pw_b, peakwin_agent=pw_a,
        peakwin_reduction_pct=100 * (pw_b - pw_a) / pw_b,
        p95_baseline=p95_b, p95_agent=p95_a,
        p95_reduction_pct=100 * (p95_b - p95_a) / p95_b,
        PAR_baseline=float(demand.max() / demand.mean()),
        PAR_agent=float(served.max() / served.mean()),
        energy_demand=float(demand.sum()), energy_served=float(served.sum()),
        undelivered_buffer=float(leftover),
    )


def _report(name, s, d, p, h, leftover):
    m = metrics(s, d, p, h, leftover)
    print(f"\n[{name}]")
    print(f"  electricity cost     : saving {m['cost_saving_pct']:.1f}%")
    print(f"  peak-window avg       : reduction {m['peakwin_reduction_pct']:.1f}%  [17-22]")
    print(f"  95th-pctile (shaving) : reduction {m['p95_reduction_pct']:.1f}%")
    print(f"  PAR (peak/avg)        : {m['PAR_baseline']:.2f} -> {m['PAR_agent']:.2f}")
    print(f"  energy served/demand  : {m['energy_served']:.0f}/{m['energy_demand']:.0f}"
          f"  (undelivered {m['undelivered_buffer']:.0f} Wh)")
    return m


def main():
    import time
    env = HouseholdEnv(EnvConfig())
    s_b, d_b, p_b, h_b, lb = rollout(env, None)
    s_r, d_r, p_r, h_r, lr = rollout(env, lambda obs, e: decide(obs, e))
    mpc = MPCController()
    t0 = time.time()
    s_m, d_m, p_m, h_m, lm = rollout(env, lambda obs, e: mpc.decide(obs, e))
    mpc_t = time.time() - t0

    print("\n=== Single-household controllers vs no-DR baseline (test period) ===")
    _report("Rule-based", s_r, d_r, p_r, h_r, lr)
    _report(f"MPC (perfect-foresight upper bound, {mpc_t:.0f}s)", s_m, d_m, p_m, h_m, lm)

    # plot first 3 days (432 steps @ 10-min): baseline vs rule vs MPC
    w = min(432, len(d_b))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d_b[:w], color=GREY, lw=1.0, label="Baseline (no DR)")
    ax.plot(s_r[:w], color=GREEN, lw=1.0, label="Rule-based", alpha=0.85)
    ax.plot(s_m[:w], color=NAVY, lw=1.2, label="MPC")
    ax.axhline(env.cfg.peak_threshold_wh, ls="--", color=ACCENT, lw=1,
               label=f"peak threshold {env.cfg.peak_threshold_wh:.0f} Wh")
    ax.set_title("Single-household load shifting: baseline vs rule-based vs MPC (first 3 days)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("10-min step"); ax.set_ylabel("Load (Wh)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "figF_agent_loadcurve.png", dpi=150)
    plt.close(fig)
    print(f"\n  load curve -> {FIGS / 'figF_agent_loadcurve.png'}")


if __name__ == "__main__":
    main()
