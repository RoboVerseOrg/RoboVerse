"""Cross-sim eval: load Go1 trained policy and roll out on Newton + mujoco.

Reports ep_r mean, fell_over count, upright_cos. Useful to verify dual-path
behavior consistency and quantify zero-shot transfer gap across sims.

Usage:
  python scripts/eval_go1_cross_sim.py --ckpt path/to/model_999.pt [--steps 1000]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import roboverse_pack  # noqa: F401
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import get_task_class


def build_actor(state_dict: dict, device: torch.device) -> nn.Module:
    """Reconstruct rsl_rl ActorCritic MLP from state_dict.

    Expected keys: ``mlp.{0,2,4,6}.{weight,bias}`` (Linear layers, with ELU between).
    """
    keys = sorted(
        [k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    mods = []
    for i, k in enumerate(keys):
        w = state_dict[k]
        mods.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(keys) - 1:
            mods.append(nn.ELU())
    actor = nn.Sequential(*mods).to(device)
    stripped = {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")}
    actor.load_state_dict(stripped, strict=False)
    actor.eval()
    return actor


def rollout(sim: str, n_envs: int, ckpt_path: str, steps: int, task_name: str, robot_name: str):
    device = torch.device("cuda:0" if sim == "newton" else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = build_actor(ckpt["actor_state_dict"], device)

    if sim == "newton":
        scn = ScenarioCfg(
            robots=[robot_name],
            objects=[],
            cameras=[],
            sim_params=SimParamCfg(dt=0.005),
            decimation=4,
            simulator="newton",
            num_envs=n_envs,
            headless=True,
            add_default_ground=True,
        )
        env = get_task_class(task_name)(scenario=scn, device=device)
    else:
        env = get_task_class(task_name)(device=device)

    env.reset()
    cur_r = torch.zeros(env.num_envs, device=device)
    total_r = torch.zeros(env.num_envs, device=device)
    ep_count = torch.zeros(env.num_envs, device=device)
    upright_samples = []
    fell_count = 0
    for _ in range(steps):
        with torch.no_grad():
            obs_t = env._obs_buf["actor"]
            if obs_t.device != device:
                obs_t = obs_t.to(device)
            a = actor(obs_t)
        _, r, term, trunc, _ = env.step(a)
        if r.device != device:
            r = r.to(device)
        if term.device != device:
            term = term.to(device)
        cur_r += r
        states = env.handler.get_states(mode="tensor")
        rkey = robot_name.replace("mjlab_", "")
        if rkey in states.robots:
            root = states.robots[rkey].root_state
            x, y = root[:, 4], root[:, 5]
            cos_tilt = 1 - 2 * (x * x + y * y)
            upright_samples.append(cos_tilt.mean().item())
        done = term | (trunc if isinstance(trunc, torch.Tensor) else torch.zeros_like(term))
        for e in range(env.num_envs):
            if done[e]:
                total_r[e] += cur_r[e]
                ep_count[e] += 1
                cur_r[e] = 0
                if term[e]:
                    fell_count += 1
    total_r += cur_r
    ep_r = (total_r / ep_count.clamp(min=1)).mean().item()
    n_eps = int(ep_count.sum().item())
    upr = (sum(upright_samples) / len(upright_samples)) if upright_samples else None
    return {"ep_r": ep_r, "n_eps": n_eps, "fell": fell_count, "upright_cos": upr}


def main() -> int:
    """Roll the checkpoint out on each backend. A sim that fails to run is not a result:
    it makes the cross-sim comparison unevaluable, so the exit status must reflect it.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="mjlab.velocity_flat_go1_v2")
    p.add_argument("--robot", default="mjlab_go1")
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()

    failed = []
    for sim, n_envs in [("newton", 16), ("mujoco", 1)]:
        print(f"=== {sim} (n_envs={n_envs}) ===")
        try:
            res = rollout(sim, n_envs, args.ckpt, args.steps, args.task, args.robot)
            upr_s = f"{res['upright_cos']:.3f}" if res["upright_cos"] is not None else "n/a"
            print(f"  ep_r_mean={res['ep_r']:.2f}  eps={res['n_eps']}  fell={res['fell']}  upright_cos={upr_s}")
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"  FAIL: {type(exc).__name__}: {str(exc)[:120]}")
            failed.append(sim)
    if failed:
        print(f"RESULT: ERROR — {', '.join(failed)} did not run; there is no cross-sim comparison.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
