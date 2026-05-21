"""Trace one mjlab env step + one MetaSim env step side-by-side, log all values."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ghr/projects/RoboVerse/RoboVerse")
sys.path.insert(0, "/home/ghr/projects/mjlab/src")

import numpy as np
import torch

# ---- mjlab side ----
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

env_cfg = load_env_cfg("Mjlab-Cartpole-Balance", play=False)
env_cfg.scene.num_envs = 1
m_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
print(f"mjlab obs space: {m_env.observation_space}")
print(f"mjlab action space: {m_env.action_space}")

# Reset both at the same seed
np.random.seed(0)
m_obs_dict, _ = m_env.reset(seed=0)
m_obs = m_obs_dict["actor"] if isinstance(m_obs_dict, dict) else m_obs_dict
print("\n=== After reset (mjlab) ===")
print(f"  obs (5D): {m_obs.cpu().numpy().flatten()}")
print(f"  qpos: {m_env.sim.data.qpos[0].cpu().numpy()}")
print(f"  qvel: {m_env.sim.data.qvel[0].cpu().numpy()}")

# ---- MetaSim side ----
import roboverse_pack  # noqa
from metasim.task.registry import get_task_class

cls = get_task_class("mjlab.cartpole_balance_train")
np.random.seed(0)
ms_env = cls()
np.random.seed(0)  # reset RNG for env's internal random.default_rng
ms_obs, _ = ms_env.reset()
print("\n=== After reset (MetaSim) ===")
print(f"  obs (5D): {ms_obs.cpu().numpy().flatten()}")
print(f"  qpos: {ms_env.handler.physics.data.qpos[:2]}")
print(f"  qvel: {ms_env.handler.physics.data.qvel[:2]}")

# Apply same action to both
action_np = np.array([0.5], dtype=np.float32)
action_t = torch.tensor([[0.5]], dtype=torch.float32, device="cuda")

# mjlab step
m_obs_dict2, m_rew, m_term, m_trunc, _ = m_env.step(action_t)
m_obs2 = m_obs_dict2["actor"] if isinstance(m_obs_dict2, dict) else m_obs_dict2
print("\n=== After step action=0.5 (mjlab) ===")
print(f"  obs: {m_obs2.cpu().numpy().flatten()}")
print(f"  qpos: {m_env.sim.data.qpos[0].cpu().numpy()}")
print(f"  qvel: {m_env.sim.data.qvel[0].cpu().numpy()}")
print(f"  reward: {m_rew.cpu().numpy().flatten()}")

# MetaSim step
ms_obs2, ms_rew, ms_term, ms_trunc, _ = ms_env.step(action_np)
print("\n=== After step action=0.5 (MetaSim) ===")
print(f"  obs: {ms_obs2.cpu().numpy().flatten()}")
print(f"  qpos: {ms_env.handler.physics.data.qpos[:2]}")
print(f"  qvel: {ms_env.handler.physics.data.qvel[:2]}")
print(f"  reward: {ms_rew.cpu().numpy().flatten()}")

m_env.close()
ms_env.close()
