"""This is a training script for skillblender framework"""

from __future__ import annotations

import os
import shutil

from loguru import logger as log

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

import wandb
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.skillblender_rl.utils import get_args, get_log_dir, get_wrapper


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=[],
    )

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=args.wandb, name=args.run_name)

    log_dir = get_log_dir(args, scenario)
    task_wrapper = get_wrapper(args.task)
    env = task_wrapper(scenario)

    # dump snapshot of training config
    task_path = f"metasim/cfg/tasks/skillblender/{scenario.task.task_name}_cfg.py"
    if not os.path.exists(task_path):
        log.error(f"Task path {task_path} does not exist, please check your task name in config carefully")
        return
    shutil.copy2(task_path, log_dir)

    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        wandb=use_wandb,
    )
    ppo_runner.learn(num_learning_iterations=args.learning_iterations)


if __name__ == "__main__":
    args = get_args()
    train(args)
