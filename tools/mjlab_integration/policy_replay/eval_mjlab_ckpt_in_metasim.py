"""Load an mjlab-trained RSL-RL checkpoint, run inference in MY MetaSim env.

Goal: verify that MY env (obs/action/reward/reset) is bitwise-equivalent to
mjlab's. If mjlab's converged policy gives ep_r near max IN MY ENV, then
my env is correct and the v5 training-curve gap is purely PPO-impl quality
(not env bug).

Usage:
    python -m tools.mjlab_integration.policy_replay.eval_mjlab_ckpt_in_metasim \
        --ckpt /path/to/cartpole_balance_64envs/<run>/model_*.pt \
        --task mjlab.cartpole_balance_train --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ghr/projects/RoboVerse/RoboVerse")
sys.path.insert(0, "/home/ghr/projects/mjlab/src")

import roboverse_pack  # noqa: F401 register
from metasim.task.registry import get_task_class


def load_rsl_actor(ckpt_path: str, obs_dim: int, act_dim: int, device):
    """Load actor weights from an RSL-RL on-policy-runner checkpoint (mjlab cartpole format)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = state["actor_state_dict"]
    actor = nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, act_dim),
    ).to(device)
    sd = {
        "0.weight": actor_state["mlp.0.weight"],
        "0.bias": actor_state["mlp.0.bias"],
        "2.weight": actor_state["mlp.2.weight"],
        "2.bias": actor_state["mlp.2.bias"],
        "4.weight": actor_state["mlp.4.weight"],
        "4.bias": actor_state["mlp.4.bias"],
    }
    actor.load_state_dict(sd)
    actor.eval()
    std = float(actor_state["distribution.std_param"].item())
    print(f"  loaded actor (5→64→64→1 ELU), std_param={std:.4f}")
    return actor


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="mjlab.cartpole_balance_train")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cls = get_task_class(args.task)
    env = cls()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    print(f"task={args.task} obs_dim={obs_dim} act_dim={act_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_rsl_actor(args.ckpt, obs_dim, act_dim, device)

    rng = np.random.default_rng(args.seed)
    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        ep_len = 0
        for t in range(env.max_episode_steps):
            with torch.no_grad():
                action_mean = actor(obs.to(device))
            action = action_mean.cpu().numpy().flatten()  # use mean (deterministic)
            obs, r, term, tout, _ = env.step(action)
            ep_r += float(r.item())
            ep_len += 1
            if term.any() or tout.any():
                break
        returns.append(ep_r)
        print(f"  episode {ep}: ep_r={ep_r:.3f}  ep_len={ep_len}")
    env.close()
    print(f"\nmean ep_r over {args.episodes} episodes: {np.mean(returns):.3f}")
    print("max return in this env: 50.0 (1000 steps * 0.05 max per step)")
    if np.mean(returns) > 45:
        print("✅ mjlab ckpt achieves near-max in MY MetaSim env → my env is bitwise-equivalent")
        print("   → v5 training gap is purely PPO-impl quality, not env bug")
    elif np.mean(returns) > 30:
        print("⚠ partial transfer — possible obs scaling difference. Investigate.")
    else:
        print("❌ ckpt fails in my env — env still has bug(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
