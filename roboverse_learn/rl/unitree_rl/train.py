from __future__ import annotations

import os
import shutil

from loguru import logger as log

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import random

import numpy as np
import torch
import wandb
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.rl.unitree_rl.helper.utils import get_args, get_class, get_log_dir, make_robots


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    # only support single robot for now
    _robots_name, _robots = make_robots(args)
    robots_name, robots = [_robots_name[0]], [_robots[0]]
    config_wrapper = get_class(args.task, suffix="Cfg")
    task_config = config_wrapper(robots=robots)
    scenario = ScenarioCfg(
        robots=robots,
        sim_params=task_config.sim_params,
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
        decimation=args.decimation,
    )

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=args.wandb, name=args.run_name)

    if args.load_run:
        datetime = args.load_run.split("/")[-2]
    else:
        datetime = None
    log_dir = get_log_dir(args, task_config, datetime)
    task_wrapper = get_class(args.task, suffix="Task")
    task_env = task_wrapper(task_config, scenario)

    # dump snapshot of training config
    task_path = f"roboverse_learn/rl/unitree_rl/tasks/{task_env.cfg.task_name}.py"
    if not os.path.exists(task_path):
        log.error(f"Task path {task_path} does not exist, please check your task name in config carefully")
        return
    shutil.copy2(task_path, log_dir)

    try:
        ppo_runner = OnPolicyRunner(
            env=task_env,
            train_cfg=task_env.train_cfg,
            device=task_env.device,
            log_dir=log_dir,
            # wandb=use_wandb,
            args=args,
        )
    except Exception as e:
        ppo_runner = OnPolicyRunner(
            env=task_env,
            train_cfg=task_env.train_cfg,
            device=task_env.device,
            log_dir=log_dir,
            # wandb=use_wandb,
            # args=args,
        )
    if args.load_run:
        ppo_runner.load(args.load_run)
    ppo_runner.learn(num_learning_iterations=task_config.ppo_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    set_seed(1)
    args = get_args()
    # args.task = "dof12_walking"
    # args.sim = "isaacgym"
    # args.num_envs = 128
    # args.robot = 'g1_dof12'
    # args.headless = True
    train(args)
