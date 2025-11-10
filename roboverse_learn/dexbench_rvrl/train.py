from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import numpy as np
import torch
import tyro
import wandb
import yaml

from roboverse_learn.dexbench_rvrl.algos.agent_factory import create_agent
from roboverse_learn.dexbench_rvrl.envs import create_vector_env

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax


########################################################
## Standalone utils
########################################################
class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards: torch.Tensor):
        self.deque.extend(rewards.cpu().numpy().tolist())

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


########################################################
## Args
########################################################
@dataclass
class Args:
    env_id: str = "dexbench/HandOver"
    sim: Literal["isaacsim", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    num_envs: int = 1
    headless: bool = False
    test: bool = False
    device: str = "cuda"  # Device for IsaacLab environments
    sim_device: str = None
    logdir: str = "logs/"
    experiment: str = "Base"
    algo: str = "ppo"
    seed: int = 0
    eval_episodes: int = 5
    model_dir: str = None
    use_wandb: bool = False
    wandb_project: str = "roboverse_dexbench_rl"
    objects: str = None
    obs_type: str = "state"  # "state" or "rgb"
    robot: str = "shadow"  # "allegro" or "shadow"
    no_prio: bool = False
    use_dr: bool = False  # whether to use domain randomization
    use_materials: bool = False  # whether to randomize object materials


################################################
## Load train cfg
################################################
def get_config_path(args):
    task = args.env_id.split("/")[-1]
    args.task = task
    if args.task in [
        "HandOver",
        "CatchUnderarm",
        "Over2Underarm",
        "PushBlock",
        "CatchAbreast",
        "DoorCloseInward",
        "DoorCloseOutward",
        "TwoCatchUnderarm",
        "GraspPlace",
        "Kettle",
    ]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/config.yaml",
        )
    elif args.task in ["StackBlock"]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/stack_block_config.yaml",
        )
    elif args.task in ["DoorOpenInward", "DoorOpenOutward"]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/open_config.yaml",
        )
    elif args.task in ["LiftUnderarm"]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/lift_config.yaml",
        )
    elif args.task in ["ReOrientation"]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/re_orientation_config.yaml",
        )
    elif args.task in ["TurnButton", "Scissor", "Pen", "SwingCup"]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/dexbench_rvrl/cfg/{args.algo}/{args.obs_type}/smooth_config.yaml",
        )
    else:
        raise ValueError(f"Unrecognized task: {args.task}. Please specify a valid task.")


def load_cfg(args, train_cfg_path, logdir):
    with open(os.path.join(os.getcwd(), train_cfg_path)) as f:
        train_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override seed if passed on the command line
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.eval_episodes is not None:
        train_cfg["eval_episodes"] = args.eval_episodes

    logdir = os.path.realpath(logdir)

    return logdir, train_cfg


def main():
    args = tyro.cli(Args)
    logdir, train_cfg_path = get_config_path(args)
    logdir, train_cfg = load_cfg(args, train_cfg_path, logdir)
    env_cfg = train_cfg.get("env", None)
    print("Algorithm: ", args.algo)
    ## set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # jax.config.update("jax_platform_name", "gpu")
    # device_num = int(args.device.split(":")[-1])
    # jax.config.update("jax_default_device", jax.devices()[device_num])
    log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))

    ## set seed
    seed_everything(args.seed)

    # Use the unified environment interface
    envs = create_vector_env(
        args.env_id,
        args=args,
        env_cfg=env_cfg,
    )

    logdir = logdir + f"_seed{args.seed}" + f"_{args.obs_type}"
    if args.objects is not None:
        logdir += f"_{args.objects}"
    if args.robot is not None:
        logdir += f"_{args.robot}"
    if args.experiment != "Base":
        logdir += f"_{args.experiment}"

    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    wandb_run = None
    if args.use_wandb and not args.test:
        wandb_name = f"{args.task}_{args.algo}_{args.experiment}"
        if args.objects is not None:
            wandb_name += f"_{args.objects}"
        if args.robot is not None:
            wandb_name += f"_{args.robot}"
        if args.experiment != "Base":
            wandb_name += f"_{args.experiment}"
        wandb_name += f"_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=train_cfg,
            name=wandb_name,
            dir=logdir,
        )

    agent = create_agent(
        algo=args.algo,
        env=envs,
        train_cfg=train_cfg,
        device=device,
        log_dir=logdir,
        model_dir=args.model_dir,
        is_testing=args.test,
        print_log=True,
        wandb_run=wandb_run,
    )

    log.info(f"{envs.single_action_space.shape=}")
    log.info(f"{envs.single_observation_space=}")

    log.info(f"Algorithm: {args.algo}")
    log.info(f"Number of environments: {args.num_envs}")
    agent.run()


if __name__ == "__main__":
    main()
