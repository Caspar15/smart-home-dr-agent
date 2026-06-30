# -*- coding: utf-8 -*-
"""RL decision core (Phase 2, Step 1b — SKELETON, build AFTER rule_based works).

Train a reinforcement-learning agent inside HouseholdEnv and compare it to the
rule-based controller. Suggested: Stable-Baselines3 PPO (continuous action) or
DQN (discrete action).

Design: docs/agent_design.md
"""
from __future__ import annotations


def train(env, total_timesteps: int = 100_000):
    """Train an RL policy in the household environment.

    TODO:
      from stable_baselines3 import PPO   # or DQN
      model = PPO("MlpPolicy", env, verbose=1)
      model.learn(total_timesteps=total_timesteps)
      return model
    """
    raise NotImplementedError("TODO: wire up Stable-Baselines3 PPO/DQN")


def evaluate(model, env):
    """Roll out the trained policy and report cost%, peak reduction%, comfort."""
    raise NotImplementedError("TODO: rollout + system-level metrics")
