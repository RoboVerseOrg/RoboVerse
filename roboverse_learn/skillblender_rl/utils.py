from __future__ import annotations

import argparse
import copy
import datetime
import importlib
import os

import torch
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.utils import is_camel_case, is_snake_case, to_camel_case


def parse_arguments(description="humanoid rl task arguments", custom_parameters=None):
    """Parse command line arguments."""

    if custom_parameters is None:
        custom_parameters = []
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
            log.error("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            log.error("supported keys: name, type, default, action, help")

    return parser.parse_args()


def get_wrapper(task_id: str):
    """Get the environment wrapper class for the given task ID."""

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


def get_load_root_dir(args: argparse.Namespace, scenario: ScenarioCfg) -> str:
    """Get the root directory to load the model from."""

    robot_name = args.robot
    task_name = scenario.task.task_name
    task_name = f"{robot_name}_{task_name}"
    if args.load_run is None:
        raise ValueError("Please provide a run name to load the model from using --load_run")
    load_root = f"./outputs/skillblender/{task_name}/{args.load_run}"
    return load_root


def get_load_path(args: argparse.Namespace, scenario: ScenarioCfg) -> str:
    """Get the path to load the model from."""

    load_root = get_load_root_dir(args, scenario)
    if args.checkpoint == -1:
        models = [file for file in os.listdir(load_root) if "model" in file]
        models.sort(key=lambda m: f"{m!s:0>15}")
        model = models[-1]
        load_path = f"{load_root}/model_{model}.pt"
    else:
        load_path = f"{load_root}/model_{args.checkpoint}.pt"
    return load_path


def get_export_jit_path(args: argparse.Namespace, scenario: ScenarioCfg) -> str:
    """Get the path to export the JIT model."""
    load_root = get_load_root_dir(args, scenario)
    exported_root_dir = f"{load_root}/exported"
    os.makedirs(exported_root_dir, exist_ok=True)
    return f"{load_root}/exported/model_exported_jit.pt"


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
            "name": "--load_run",
            "type": str,
            "default": None,
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


def export_policy_as_jit(actor_critic, path, filename=None):
    """Export the policy as a JIT model."""
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def get_body_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, body_names: list[str], device
):
    """given substrings of body name, find all the bodies indices in sorted order."""

    matches = []
    sorted_names = sim_handler.get_body_names(obj_name, sort=True)

    for name in body_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index


def get_joint_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, joint_names: list[str], device: str
):
    """given substrings of joint name, find all the bodies indices in sorted order."""

    matches = []
    sorted_names = sim_handler.get_joint_names(obj_name, sort=True)

    for name in joint_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index


def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate a tensor of random floats in the range [lower, upper]."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower
