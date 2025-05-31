"""This is a training script that train legged robot"""

from __future__ import annotations

# after handler initialization, assign dof limits to task cfg class for reward computation.
import datetime
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
from roboverse_learn.skillblender_rl.legged_robot_wrapper import LeggedRobotWrapper


def parse_arguments(
    description="humanoid rl task arguments", headless=False, no_graphics=False, custom_parameters=None
):
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


def get_args(test=False):
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
            "help": "number of parrallel environments.",
        },
        {
            "name": "--sim",
            "type": str,
            "default": "isaacgym",
            "help": "simulator type, currently only isaacgym is supported",
        },
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {
            "name": "--resume_stop_at_max",
            "action": "store_true",
            "default": False,
            "help": "If resume, will stop at max_iterations as in config",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "required": True if not test else False,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "default": "",
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Use jit to play"},
        {"name": "--use_wandb", "action": "store_true", "default": True, "help": "Use wandb for logging"},
        {"name": "--wandb", "type": str, "default": "h1_walking", "help": "Wandb project name"},
        {"name": "--visualize", "action": "store_true", "default": False, "help": "Only show one environment"},
        {"name": "--baseline", "type": str, "default": "None", "help": "Baseline name (e.g. H2O, ExBody)"},
    ]
    # parse arguments
    args = parse_arguments(custom_parameters=custom_parameters)
    return args


def get_log_dir(scenario: ScenarioCfg):
    """Get the log directory based on the scenario configuration."""
    robot_name = scenario.task.robotname
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
    cameras = []
    scenario = ScenarioCfg(
        task=args.task, robot=args.robot, num_envs=args.num_envs, sim=args.sim, headless=args.headless, cameras=cameras
    )
    log_dir = get_log_dir(scenario)
    env = LeggedRobotWrapper(scenario)
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
    ppo_runner.learn(num_learning_iterations=20000)  # TODO fix hard coding


# TODO expose algorithm api to let user define their own nerual network
if __name__ == "__main__":
    args = get_args()
    train(args)
