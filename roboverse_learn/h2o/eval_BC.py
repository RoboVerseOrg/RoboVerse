#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ONE BC checkpoint and append metrics to a CSV file.

Invoked by run_eval_BC_batch.sh, e.g.:

python eval_BC_single.py \
    ckpt=/path/file.ckpt \
    motion.motion_file=resources/motions/h1/kit_6.pkl \
    num_envs=2048 \
    csv_out=./eval_results/result.csv \
    <other Hydra overrides>

If the CSV does not exist, a header row is written automatically.
"""

import os
import csv
import gc
from typing import Any

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import pathlib
from legged_gym.utils import task_registry
from rsl_rl.runners.eval_runner_BC_modified import EvalRunnerBCModified


# ---------- Helpers -------------------------------------------------
def to_py(obj: Any):
    """Convert Numpy / Torch types to native Python for CSV/JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj


def load_BC_workspace(train_cfg, checkpoint_path):
    OmegaConf.resolve(train_cfg)                       # expand ${} refs
    cls = hydra.utils.get_class(train_cfg._target_)    # workspace class
    ws  = cls(train_cfg)                               # instantiate
    ck  = pathlib.Path(checkpoint_path)
    if ck.is_file():
        print(f"[loader] restore {ck}")
        ws.load_checkpoint(path=ck)
    else:
        raise FileNotFoundError(ck)
    return ws

# ---------- Hydra entrypoint ---------------------------------------
@hydra.main(version_base=None, config_path="../cfg", config_name="config_base")
def main(cfg: DictConfig) -> None:
    # Required CLI arguments check
    for key in ("ckpt", "motion.motion_file", "num_envs", "csv_out"):
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"Missing required override: {key}=...")

    # Convert to plain EasyDict for convenience
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))

    # -------- Environment tweaks (evaluation-only settings) --------
    env_cfg, train_cfg = cfg, cfg.train
    env_cfg.env.test = True
    env_cfg.env.num_envs = cfg.num_envs

    env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.add_eval_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20

    # -------- Build environment and load policy --------
    env, _ = task_registry.make_env_hydra(
        name=cfg.task,
        hydra_cfg=cfg,
        env_cfg=env_cfg,
    )
    To = cfg.humanoid_workspace.n_obs_steps

    workspace = load_BC_workspace(
        checkpoint_path=cfg.ckpt,
        train_cfg=cfg.humanoid_workspace,
    )
    policy = workspace.model.to(env.device)

    runner = EvalRunnerBCModified(
        env=env,
        policy=policy,
        train_cfg=train_cfg,
        device=env.device,
        To=To,
        clip_action=True,
    )
    metrics = to_py(runner.eval())

    # -------- Append results to CSV --------
    csv_path = cfg.csv_out
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if header_needed:
            writer.writerow(["ckpt"] + list(metrics.keys()))
        writer.writerow([os.path.basename(cfg.ckpt).replace(".ckpt", "")] +
                        list(metrics.values()))
    print(f"✓ Results written to {csv_path}")

    # Cleanup GPU memory
    del env, policy, runner, workspace
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
