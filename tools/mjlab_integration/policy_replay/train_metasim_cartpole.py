"""Train an mjlab cartpole policy entirely inside MetaSim (no mjlab runtime).

Uses a minimal CleanRL-style PPO loop over a `gym.vector.SyncVectorEnv` of
MetaSim cartpole envs. Goal: show the reward curve converges to ~50 (mjlab's
own training plateau on the dm_control smooth reward), proving the 1:1 port
in `roboverse_pack.tasks.mjlab.cartpole_train` is sufficient to reproduce
mjlab native training results inside our framework.

Usage:
    PYTHONPATH=/home/ghr/projects/RoboVerse/RoboVerse MUJOCO_GL=egl \\
      python -m tools.mjlab_integration.policy_replay.train_metasim_cartpole \\
      --task mjlab.cartpole_balance_train --num-envs 8 --total-iter 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Avoid GLFW issues
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

import roboverse_pack  # noqa: F401  trigger task registration
from metasim.task.registry import get_task_class

# --- env factory --------------------------------------------------------------


class _GymWrap(gym.Env):
    """Thin wrapper exposing a single MetaSim cartpole env as gym.Env."""

    def __init__(self, task_name: str):
        cls = get_task_class(task_name)
        self._env = cls()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self._env.reset()
        return obs.cpu().numpy().squeeze(0).astype(np.float32), info

    def step(self, action):
        obs, r, term, tout, info = self._env.step(np.asarray(action, dtype=np.float32))
        return (
            obs.cpu().numpy().squeeze(0).astype(np.float32),
            float(r.item()),
            bool(term.any().item()),
            bool(tout.any().item()),
            info,
        )

    def close(self):
        self._env.close()


def make_factory(task_name: str):
    def _f():
        return _GymWrap(task_name)

    return _f


# --- PPO --------------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class RunningMeanStd:
    """Welford running mean/std for observation normalization."""

    def __init__(self, shape, device, eps: float = 1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = float(eps)

    def update(self, x: torch.Tensor) -> None:
        # x: (N, *shape)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, activation: str = "elu", init_std: float = 1.0):
        super().__init__()
        act_cls = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            act_cls(),
            layer_init(nn.Linear(64, 64)),
            act_cls(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            act_cls(),
            layer_init(nn.Linear(64, 64)),
            act_cls(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # mjlab uses init_std=1.0 (logstd=0). With dt-scaled reward (~0.05/step max),
        # large initial exploration is fine — pole just oscillates without big penalty.
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(math.log(init_std))))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="mjlab.cartpole_balance_train")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--total-iter", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--mini-batches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--activation", default="elu", choices=["elu", "tanh", "relu"])
    p.add_argument("--init-std", type=float, default=1.0, help="initial action std (mjlab default = 1.0)")
    p.add_argument(
        "--obs-norm", type=int, default=0, choices=[0, 1], help="observation normalization (mjlab default = 0)"
    )
    p.add_argument("--desired-kl", type=float, default=0.01, help="target KL for adaptive lr schedule")
    p.add_argument("--lr-schedule", default="adaptive", choices=["fixed", "adaptive"])
    p.add_argument(
        "--value-clip", type=int, default=1, choices=[0, 1], help="RSL-RL style value loss clipping (mjlab default = 1)"
    )
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--out", default="/home/ghr/projects/RoboVerse/RoboVerse/training_logs/metasim_native")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out) / args.task.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    print(f"[metasim_train] task={args.task} num_envs={args.num_envs} device={device}")
    print(f"[metasim_train] log → {log_path}")

    envs = SyncVectorEnv([make_factory(args.task) for _ in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    print(f"[metasim_train] obs_dim={obs_dim} act_dim={act_dim}")

    agent = Agent(obs_dim, act_dim, activation=args.activation, init_std=args.init_std).to(device)
    opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    # Running obs norm (optional — mjlab disables for cartpole).
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=device) if args.obs_norm else None
    current_lr = args.lr
    # Best-ep_r ckpt saving — fixes the "max during train but not at final" plateau-overfit issue.
    best_ep_r = float("-inf")
    best_path = out_dir / "agent_best.pt"

    # Rollout buffers
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_raw, _ = envs.reset(seed=0)
    next_obs_raw = torch.as_tensor(next_obs_raw, dtype=torch.float32, device=device)
    if obs_rms is not None:
        obs_rms.update(next_obs_raw)
        next_obs = obs_rms.normalize(next_obs_raw)
    else:
        next_obs = next_obs_raw
    next_done = torch.zeros(args.num_envs, device=device)

    # Episode reward tracking
    episode_rew = np.zeros(args.num_envs)
    episode_len = np.zeros(args.num_envs)
    recent_rewards = []  # finished episode returns
    recent_lens = []

    t0 = time.time()
    for iteration in range(1, args.total_iter + 1):
        for t in range(args.num_steps):
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
                val_buf[t] = value.squeeze(-1)
            act_buf[t] = action
            logp_buf[t] = logp

            act_np = action.cpu().numpy()
            next_obs_np, rew_np, terms, touts, _infos = envs.step(act_np)
            done_np = np.logical_or(terms, touts)
            rew_buf[t] = torch.as_tensor(rew_np, dtype=torch.float32, device=device)

            episode_rew += rew_np
            episode_len += 1
            for i, d in enumerate(done_np):
                if d:
                    recent_rewards.append(float(episode_rew[i]))
                    recent_lens.append(int(episode_len[i]))
                    if len(recent_rewards) > 64:
                        recent_rewards = recent_rewards[-64:]
                        recent_lens = recent_lens[-64:]
                    episode_rew[i] = 0.0
                    episode_len[i] = 0

            next_obs_raw = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            if obs_rms is not None:
                obs_rms.update(next_obs_raw)
                next_obs = obs_rms.normalize(next_obs_raw)
            else:
                next_obs = next_obs_raw
            next_done = torch.as_tensor(done_np.astype(np.float32), device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rew_buf)
            lastgae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterm = 1.0 - next_done
                    nextval = next_value
                else:
                    nextnonterm = 1.0 - done_buf[t + 1]
                    nextval = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextval * nextnonterm - val_buf[t]
                advantages[t] = lastgae = delta + args.gamma * args.gae_lambda * nextnonterm * lastgae
            returns = advantages + val_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        # PPO update — RSL-RL aligned: value clipping + full-batch mean KL for adaptive lr
        batch_size = args.num_steps * args.num_envs
        mb_size = batch_size // args.mini_batches
        inds = np.arange(batch_size)
        kls = []
        for _ in range(args.epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, mb_size):
                mb = inds[start : start + mb_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = torch.exp(logratio)
                with torch.no_grad():
                    kl = float(((torch.exp(logratio) - 1) - logratio).mean().item())
                    kls.append(kl)

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO clipped surrogate
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss with clipping (RSL-RL style)
                if args.value_clip:
                    newval_sq = newval.squeeze(-1)
                    v_clipped = b_val[mb] + torch.clamp(newval_sq - b_val[mb], -args.clip_coef, args.clip_coef)
                    v_loss_unclipped = (newval_sq - b_ret[mb]) ** 2
                    v_loss_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval.squeeze(-1) - b_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()

        # Adaptive LR (RSL-RL style): use MEAN KL across the full epoch's mini-batches.
        last_kl = float(np.mean(kls)) if kls else 0.0
        if args.lr_schedule == "adaptive":
            if last_kl > 2.0 * args.desired_kl:
                current_lr = max(1e-5, current_lr / 1.5)
            elif last_kl < 0.5 * args.desired_kl and last_kl > 0:
                current_lr = min(1e-2, current_lr * 1.5)
            for pg in opt.param_groups:
                pg["lr"] = current_lr

        # Mean per-step reward in this iteration (always available; scaled by ep_len gives episode return)
        mean_step_r = float(rew_buf.mean().item())
        # Mean completed-episode return (only meaningful once episodes finish)
        mean_r = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
        # Save best ckpt: when recent-episode mean reward exceeds prior best.
        if recent_rewards and mean_r > best_ep_r and len(recent_rewards) >= 8:
            best_ep_r = mean_r
            torch.save({"agent": agent.state_dict(), "args": vars(args), "iter": iteration, "ep_r": mean_r}, best_path)
        mean_l = float(np.mean(recent_lens)) if recent_lens else float("nan")
        elapsed = time.time() - t0
        steps_so_far = iteration * args.num_steps * args.num_envs
        rec = {
            "iter": iteration,
            "global_step": steps_so_far,
            "mean_step_reward": mean_step_r,
            "mean_reward_recent": mean_r,
            "mean_ep_len_recent": mean_l,
            "pg_loss": float(pg_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(ent_loss.item()),
            "kl": last_kl,
            "lr": current_lr,
            "elapsed_sec": elapsed,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if iteration % 5 == 0 or iteration == 1:
            print(
                f"[iter {iteration:>3}/{args.total_iter}] steps={steps_so_far:>7}  "
                f"step_r={mean_step_r:.3f}  ep_r={mean_r:6.2f} ep_len={mean_l:5.0f}  "
                f"pg={pg_loss.item():.4f} v={v_loss.item():.4f} ent={ent_loss.item():.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

    # Save checkpoint
    ckpt = out_dir / "agent_final.pt"
    torch.save({"agent": agent.state_dict(), "args": vars(args)}, ckpt)
    print(f"[metasim_train] saved {ckpt}")
    envs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
