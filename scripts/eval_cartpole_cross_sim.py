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
    """Reconstruct the rsl_rl actor MLP from an ``actor_state_dict``.

    rsl_rl saves the actor as ``mlp.{0,2,4}.{weight,bias}`` (Linear layers with
    ELU between), so dims are read straight from the checkpoint -- ``obs_dim``
    and ``act_dim`` are kept only for signature compatibility. (The previous
    impl probed for an ``actor.`` prefix that this checkpoint format does not
    use, which built an empty Sequential -- an identity that passed the raw
    observation through as the action and broke the reward broadcast.)
    """
    keys = sorted(
        [k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    modules: list[nn.Module] = []
    for i, k in enumerate(keys):
        w = state_dict[k]
        modules.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(keys) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules)
    actor.load_state_dict(
        {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")},
        strict=False,
    )
    actor.eval()
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
    total_step_r = 0.0
    n_steps = 0
    for _ in range(steps):
        with torch.no_grad():
            actor_obs = env._obs_buf["actor"]
            # The actor lives on `device`; the env (e.g. mujoco) may run on CPU.
            # Compute the action on the actor's device, hand it back on the env's.
            mean_action = actor(actor_obs.to(device)).to(actor_obs.device)
        ret = env.step(mean_action)
        r = ret[1].to(device)
        cur_r += r
        total_step_r += float(r.sum())
        n_steps += env.num_envs
        # Episode boundary = terminated OR truncated. Balance tasks end by
        # timeout (truncation), not termination, so ret[2] alone stays False
        # and no episodes would ever be counted.
        done = ret[2].clone() if isinstance(ret[2], torch.Tensor) else None
        if len(ret) > 3 and isinstance(ret[3], torch.Tensor):
            done = ret[3].clone() if done is None else (done | ret[3])
        if done is not None:
            for env_i in range(env.num_envs):
                if done[env_i]:
                    total_r[env_i] += cur_r[env_i]
                    ep_count[env_i] += 1
                    cur_r[env_i] = 0.0
    mean_ep_r = (total_r / ep_count.clamp(min=1)).mean().item()
    mean_step_r = total_step_r / max(n_steps, 1)
    # The reward manager scales each term by step_dt; divide it back out so the
    # number is interpretable (the raw cartpole smooth_reward is ~1.0 when the
    # pole is perfectly balanced). Useful for reading cross-sim transfer.
    step_dt = float(getattr(env, "step_dt", 1.0)) or 1.0
    return mean_ep_r, ep_count.sum().item(), mean_step_r, mean_step_r / step_dt


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
            ep_r, n_eps, step_r, step_r_raw = rollout(
                args.task, args.robot, sim, args.ckpt, n_envs=n_envs, steps=args.steps
            )
            print(
                f"  {sim:8s} n_envs={n_envs:2d}: smooth_reward~{step_r_raw:.3f}/step "
                f"(dt-scaled {step_r:.4f}, ep_r={ep_r:.2f}, n_eps={n_eps:.0f})"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  {sim:8s}: FAIL {type(e).__name__}: {str(e)[:120]}")
