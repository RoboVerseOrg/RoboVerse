"""Cross-sim eval: load cartpole policy trained on Newton, roll out on both
mujoco scene-MJCF + Newton RobotCfg paths, report ep_r mean.

Tests that the dual-path implementation preserves policy semantics across sims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import roboverse_pack  # noqa: F401
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import get_task_class


def _build_actor(state_dict: dict, obs_dim: int, act_dim: int) -> nn.Module:
    """Reconstruct rsl_rl ActorCritic actor MLP from state_dict.

    Default rsl_rl ActorCritic uses ELU activations, hidden [128, 128].
    Probe state_dict for input/hidden/output dims.
    """
    keys = list(state_dict.keys())
    layers = []
    in_dim = obs_dim
    # rsl_rl saves actor as "actor.0.weight", "actor.2.weight", "actor.4.weight" (with elu between)
    for k in keys:
        if k.endswith(".weight") and k.startswith("actor"):
            w = state_dict[k]
            out_dim = w.shape[0]
            layers.append((in_dim, out_dim))
            in_dim = out_dim
    # build sequential
    modules = []
    for i, (i_d, o_d) in enumerate(layers):
        modules.append(nn.Linear(i_d, o_d))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules)
    # Load weights from prefixed state_dict
    prefix_stripped = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            new_k = k[len("actor.") :]  # "0.weight", "2.weight"
            # collapse: ELU has no params; nn.Sequential stores indices skipping ELU
            # Probe: ours is [Linear(0), ELU(1), Linear(2), ELU(3), Linear(4)]
            # Original may be [Linear(0), ELU(1), Linear(2), ELU(3), Linear(4)] — same
            prefix_stripped[new_k] = v
    actor.load_state_dict(prefix_stripped, strict=False)
    return actor


def rollout(
    task_name: str,
    robot_name: str,
    sim: str,
    ckpt_path: str,
    n_envs: int = 4,
    steps: int = 1000,
    device: str = "cuda:0",
) -> float:
    """Roll out trained policy and return mean ep_r."""
    if sim == "mujoco":
        scn = None  # use task's default scene-MJCF scenario
    else:
        scn = ScenarioCfg(
            robots=[robot_name],
            objects=[],
            cameras=[],
            sim_params=SimParamCfg(dt=0.005),
            decimation=4,
            simulator=sim,
            num_envs=n_envs,
            headless=True,
            add_default_ground=True,
        )
    TCls = get_task_class(task_name)
    env = TCls(scenario=scn, device=device) if scn else TCls(device=device)

    obs_dim = env._obs_buf["actor"].shape[-1] if env._obs_buf is not None else None
    if obs_dim is None:
        env.reset()
        obs_dim = env._obs_buf["actor"].shape[-1]
    act_dim = env.num_actions

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]
    actor = _build_actor(actor_sd, obs_dim, act_dim).to(device)
    actor.eval()

    env.reset()
    total_r = torch.zeros(env.num_envs, device=device)
    ep_count = torch.zeros(env.num_envs, device=device)
    cur_r = torch.zeros(env.num_envs, device=device)
    for _ in range(steps):
        with torch.no_grad():
            actor_obs = env._obs_buf["actor"]
            mean_action = actor(actor_obs)  # rsl_rl actor outputs mean (no log_std)
        ret = env.step(mean_action)
        r = ret[1]
        cur_r += r
        # Detect dones to roll up episode rewards
        if isinstance(ret[2], torch.Tensor):
            done = ret[2]
            for env_i in range(env.num_envs):
                if done[env_i]:
                    total_r[env_i] += cur_r[env_i]
                    ep_count[env_i] += 1
                    cur_r[env_i] = 0.0
    mean_ep_r = (total_r / ep_count.clamp(min=1)).mean().item()
    return mean_ep_r, ep_count.sum().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="mjlab.cartpole_balance_v2")
    p.add_argument("--robot", default="mjlab_cartpole")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()

    print(f"=== {args.task} cross-sim eval (ckpt={Path(args.ckpt).name}) ===")
    for sim, n_envs in [("mujoco", 1), ("newton", 16)]:
        try:
            ep_r, n_eps = rollout(args.task, args.robot, sim, args.ckpt, n_envs=n_envs, steps=args.steps)
            print(f"  {sim:8s} n_envs={n_envs:2d}: mean_ep_r={ep_r:.2f} (n_eps={n_eps:.0f})")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  {sim:8s}: FAIL {type(e).__name__}: {str(e)[:120]}")
