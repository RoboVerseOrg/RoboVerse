"""This is a training script that train legged robot"""

from __future__ import annotations

import datetime
import importlib
import os

from loguru import logger as log

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)
import argparse

import wandb
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils import is_camel_case, is_snake_case, to_camel_case


def parse_arguments(description="humanoid rl task arguments", custom_parameters=None):
    """Parse arguments."""

    if custom_parameters is None:
        custom_parameters = []
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"], type=argument["type"], default=argument["default"], help=help_str
                    )
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    return parser.parse_args()


# TODO
# 1. add resume training from checkpoint


def get_wrapper(task_id: str):
    if ":" in task_id:
        prefix, task_name = task_id.split(":")
        if prefix not in ["skillblender", "Skillblender"]:
            raise ValueError(f"Invalid task name: {task_id}, should be skillblender:task_name in the format")
    else:
        raise ValueError(f"Invalid task name: {task_id}, should be skillblender:task_name in the format")

    if is_camel_case(task_name):
        task_name_camel = task_name

    elif is_snake_case(task_name):
        task_name_camel = to_camel_case(task_name)

    wrapper_module = importlib.import_module("roboverse_learn.skillblender_rl.env_wrappers")
    wrapper_cls = getattr(wrapper_module, f"{task_name_camel}Wrapper")
    return wrapper_cls


def get_args(test=False):
    """Get the command line arguments."""

    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "skillblender:Walking",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--robot",
            "type": str,
            "default": "h1",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 128,
            "help": "number of parallel environments.",
        },
        {
            "name": "--sim",
            "type": str,
            "default": "isaacgym",
            "help": "simulator type, currently only isaacgym is supported",
        },
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {
            "name": "--run_name",
            "type": str,
            "required": True if not test else False,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--learning_iterations",
            "type": int,
            "default": 15000,
            "help": "Path to the config file. If provided, will override command line arguments.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--use_wandb", "action": "store_true", "default": True, "help": "Use wandb for logging"},
        {"name": "--wandb", "type": str, "default": "h1_walking", "help": "Wandb project name"},
    ]
    args = parse_arguments(custom_parameters=custom_parameters)
    return args


def get_log_dir(args: argparse.Namespace, scenario: ScenarioCfg) -> str:
    """Get the log directory."""
    robot_name = args.robot
    task_name = scenario.task.task_name
    task_name = f"{robot_name}_{task_name}"
    now = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    log_dir = f"./outputs/skillblender/{task_name}/{now}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log.info("Log directory: {}", log_dir)
    return log_dir


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO add camera
    # cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    cameras = []
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=cameras,
    )
    log_dir = get_log_dir(args, scenario)
    task_wrapper = get_wrapper(args.task)
    env = task_wrapper(scenario)
    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=args.wandb, name=args.run_name)
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        wandb=use_wandb,
        args=args,
    )
    ppo_runner.learn(num_learning_iterations=args.learning_iterations)


# TODO expose algorithm api to let user define their own nerual network
if __name__ == "__main__":
    args = get_args()
    train(args)
