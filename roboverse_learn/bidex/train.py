# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
from typing import Literal

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

try:
    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
except ImportError:
    log.warning("IsaacGym is not installed. Some functionalities may not work as expected.")

import tyro
import yaml
from rl_env_wrapper import BiDexEnvWrapper

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_task

ALGOS = ["ppo"]


@configclass
class Args:
    """Arguments for RL Policy training."""

    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    num_envs: int = 1
    headless: bool = False
    test: bool = False
    task: str = "ShadowHandOver"
    device: str = "cuda:0"
    logdir: str = "logs/"
    name: str = "Base"
    experiment: str = "Base"
    cfg_train: str = "Base"
    seed: int = 0
    max_iterations: int = -1
    mini_batch_size: int = -1
    torch_deterministic: bool = False
    algo: str = "ppo"
    model_dir: str = ""
    randomize: bool = False
    episode_length: int = 0

    train_cfg = None

    train = not test  # if test is True, then train is False


def get_config_path(args):
    if args.task in [
        "ShadowHandOver",
    ]:
        return (
            os.path.join(args.logdir, f"{args.task}/{args.algo}"),
            f"roboverse_learn/bidex/cfg/{args.algo}/config.yaml",
        )

    else:
        raise ValueError(f"Unrecognized task: {args.task}. Please specify a valid task.")


def load_cfg(args, train_cfg_path, logdir):
    with open(os.path.join(os.getcwd(), train_cfg_path)) as f:
        train_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Set deterministic mode
    if args.torch_deterministic:
        train_cfg["torch_deterministic"] = True

    # Override seed if passed on the command line
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    log_id = logdir
    if args.experiment != "Base":
        log_id = args.logdir + f"_{args.experiment}"

    logdir = os.path.realpath(log_id)
    # os.makedirs(logdir, exist_ok=True)

    return logdir, train_cfg


def train(args):
    print("Algorithm: ", args.algo)
    assert args.algo in ALGOS, "Unrecognized algorithm!\nAlgorithm should be one of: [ppo]"
    algo = args.algo
    task = get_task(args.task)
    task.num_envs = args.num_envs
    task.device = args.device
    print(task.robots)
    scenario = ScenarioCfg(
        task=task,
        robots=task.robots,
        sensors=task.sensors,
        sim=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
        sim_params=task.sim_params,
    )
    scenario.cameras = []
    env = BiDexEnvWrapper(
        scenario=scenario,
        seed=args.seed,
    )

    learn_cfg = args.train_cfg["learn"]
    is_testing = learn_cfg["test"]
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    if args.max_iterations != -1:
        args.train_cfg["learn"]["max_iterations"] = args.max_iterations

    logdir = args.logdir + f"_seed{args.seed}"

    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    """Set up the algo system for training or inferencing."""
    model = eval(args.algo.upper())(
        vec_env=env,
        cfg_train=args.train_cfg,
        device=args.device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
    )

    if is_testing and args.model_dir != "":
        print(f"Loading model from {chkpt_path}")
        model.test(chkpt_path)
    elif args.model_dir != "":
        print(f"Loading model from {chkpt_path}")
        model.load(chkpt_path)

    iterations = args.train_cfg["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    log.info(f"Algorithm: {args.algo}")
    log.info(f"Number of environments: {args.num_envs}")

    model.run(num_learning_iterations=iterations, log_interval=args.train_cfg["learn"]["save_interval"])


def main():
    args = tyro.cli(Args)
    logdir, train_cfg_path = get_config_path(args)
    logdir, train_cfg = load_cfg(args, train_cfg_path, logdir)
    args.logdir = logdir
    args.train_cfg = train_cfg
    train(args)


if __name__ == "__main__":
    main()
