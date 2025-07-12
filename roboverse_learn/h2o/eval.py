# eval.py
# Minimal evaluator for an H2O / LeggedRobot policy.
# ---------------------------------------------------
from __future__ import annotations

import argparse
import gc
import os
import pickle

import numpy as np
import torch
from legged_gym.envs.h2o.legged_robot import LeggedRobot
from tqdm import tqdm

# -- project-specific paths (edit) ------------------------------------
from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import load_policy  # helper: returns torch.nn.Module

from .h2o_util import compute_metrics_lite  # metrics utils

# --------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    scenario: ScenarioCfg,
    policy_path: str,
    device: str = "cpu",
    T_stack: int = 1,
    clip_actions: bool = False,
):
    """Run one full sweep over the motion library and print metrics."""

    # ----------------------------------------------------------------
    # 1. Build env + policy
    # ----------------------------------------------------------------
    cfg      = scenario.robots[0].export_legged_cfg()
    env      = LeggedRobot(cfg, scenario.sim_params,
                           scenario.physics_engine,
                           scenario.sim_device, scenario.headless)
    env.begin_seq_motion_samples()          # start from the first mocap
    env.compute_observations()

    policy   = load_policy(policy_path)     # your helper should load weights
    policy.to(device).eval()

    # ----------------------------------------------------------------
    # 2. Helper tensors & logging
    # ----------------------------------------------------------------
    obs_stack  = [env.obs_buf.clone() for _ in range(T_stack)]
    stack_cat  = lambda: torch.stack(obs_stack, dim=1)

    success_hist   : list[np.ndarray] = []
    pred_pos_full  : list[np.ndarray] = []
    gt_pos_full    : list[np.ndarray] = []

    max_frames = env._motion_lib.get_motion_num_steps().max()
    max_motions = env._motion_lib._num_unique_motions
    bar = tqdm(total=max_motions // env.num_envs, desc="eval")

    # ----------------------------------------------------------------
    # 3. Main loop
    # ----------------------------------------------------------------
    t_global = 0
    done_counter = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    timeout_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    while bar.n < bar.total:
        actions = policy.predict_action({"obs": stack_cat()})["action"][:, 0]

        if clip_actions:
            centered = actions*0.25 + env.default_dof_pos
            clamped  = torch.clamp(centered,
                                   env.dof_pos_limits[:, 0],
                                   env.dof_pos_limits[:, 1])
            actions  = (clamped - env.default_dof_pos) * 4

        obs, _, _, done, info = env.step(actions)

        # ---- bookkeeping ------------------------------------------
        t_global += 1
        obs_stack.append(env.obs_buf.clone())
        if len(obs_stack) > T_stack:
            obs_stack.pop(0)

        # first time each env hits done ⇒ record step count
        freshly_done = done & (done_counter == 0)
        timeout_mask = timeout_mask | (freshly_done & info["time_outs"])
        done_counter = torch.where(freshly_done,
                                   torch.full_like(done_counter, t_global),
                                   done_counter)

        # collect per-step logs for MPJPE etc.
        pred_pos_full.append(info["body_pos"])
        gt_pos_full.append(info["body_pos_gt"])

        # ---- batch finished? --------------------------------------
        if done.all():
            success_hist.append(timeout_mask.cpu().numpy())
            env.forward_motion_samples()       # load next chunk
            env.compute_observations()
            obs_stack = [env.obs_buf.clone() for _ in range(T_stack)]
            t_global  = 0
            done_counter.zero_()
            timeout_mask.zero_()
            bar.update(1)

    bar.close()
    # ----------------------------------------------------------------
    # 4. Metrics
    # ----------------------------------------------------------------
    succ_flags = np.concatenate(success_hist)[:max_motions]
    succ_idx   = np.flatnonzero(succ_flags)

    preds = np.concatenate(pred_pos_full)[:max_motions]
    gts   = np.concatenate(gt_pos_full)  [:max_motions]

    metrics_all  = compute_metrics_lite(preds, gts)
    metrics_succ = compute_metrics_lite(preds[succ_idx], gts[succ_idx]) if len(succ_idx) else metrics_all

    print("\n==========  EVAL SUMMARY  ==========")
    print(f"Success rate        : {succ_flags.mean():.4f}")
    for k, v in metrics_all.items():
        print(f"{k:16s}: {v.mean():.4f}   (succ {metrics_succ[k].mean():.4f})")

    # ----------------------------------------------------------------
    # 5. Optional: save animation pkl for Blender
    # ----------------------------------------------------------------
    anim = dict(body_pred=preds, body_gt=gts, succ=succ_flags)
    os.makedirs("blender", exist_ok=True)
    with open("blender/animation.pkl", "wb") as f:
        pickle.dump(anim, f)
    print("Saved blender/animation.pkl")

    # tidy up
    torch.cuda.empty_cache(); gc.collect()
    return metrics_all, metrics_succ


# ======================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",    required=True, help="ScenarioCfg yaml / pkl")
    p.add_argument("--ckpt",   required=True, help="trained policy .pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stack",  type=int, default=1, help="obs frame-stack")
    p.add_argument("--clip_actions", action="store_true")
    args = p.parse_args()

    scenario = ScenarioCfg.load(args.cfg)    # adapt if your loader differs
    evaluate(scenario,
             policy_path=args.ckpt,
             device=args.device,
             T_stack=args.stack,
             clip_actions=args.clip_actions)
