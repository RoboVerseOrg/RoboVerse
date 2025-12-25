from __future__ import annotations

import argparse
import os
import random
import re
from functools import lru_cache
from typing import Callable

import numpy as np
import torch
from loguru import logger as log


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
                        argument["name"],
                        type=argument["type"],
                        default=argument["default"],
                        help=help_str,
                    )
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            log.error("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            log.error("supported keys: name, type, default, action, help")

    return parser.parse_args()


def get_args(test=False):
    """Get the command line arguments."""
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "walk_g1_dof29",
            "help": "Task name for training/testing.",
        },
        {"name": "--robots", "type": str, "default": "", "help": "The used robots."},
        {
            "name": "--objects",
            "type": str,
            "default": None,
            "help": "The used objects.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 128,
            "help": "number of parallel environments.",
        },
        {
            "name": "--iter",
            "type": int,
            "default": 15000,
            "help": "Max number of training iterations.",
        },
        {
            "name": "--sim",
            "type": str,
            "default": "isaacgym",
            "help": "simulator type, currently only isaacgym is supported",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": True,
            "help": "Force display off at all times",
        },
        {
            "name": "--resume",  # TODO
            "type": str,
            "default": None,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--checkpoint",  # TODO
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "default": -1,
            "help": "The random seed for the run. If -1, will be randomly generated.",
        },
        {
            "name": "--eval",
            "action": "store_true",
            "default": False,
            "help": "Whether to run in eval mode",
        },
        {
            "name": "--jit_load",
            "action": "store_true",
            "default": False,
            "help": "Whether to load the JIT model",
        },
        # {"name": "--run_name", "type": str, "required": True if not test else False, "help": "Name of the run. Overrides config file if provided."},
        # {"name": "--load_run", "type": str, "default": None, "help": "Path to the config file. If provided, will override command line arguments."},
        # {"name": "--use_wandb", "action": "store_true", "default": True, "help": "Use wandb for logging"},
        # {"name": "--wandb", "type": str, "default": "g1_walking", "help": "Wandb project name"},
        # {"name": "--log", "type": str, "default": None, "help": "log directory. If None, will be set automatically."},
    ]
    args = parse_arguments(custom_parameters=custom_parameters)
    return args


def set_seed(seed=-1):
    """Set the seed for the random number generators."""
    if seed == -1:
        seed = np.random.randint(0, 10000)
    log.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_indices_from_substring(
    candidates_list: list[str] | tuple[str] | str,
    data_base: list[str],
    fullmatch: bool = True,
) -> torch.Tensor:
    """Get indices of items matching the candidates patterns.

    Args:
        candidates_list: Single pattern or list of patterns (supports regex if use_regex=True)
        data_base: List of names to search in
        fullmatch: If True, require full regex match; otherwise allow substring search.

    Returns:
        Sorted tensor of matching indices

    Examples:
        >>> get_indices_from_substring(".*ankle.*", ["left_ankle", "right_ankle", "knee"])
        tensor([0, 1])
        >>> get_indices_from_substring([".*ankle.*", ".*knee.*"], ["left_ankle", "knee"])
        tensor([0, 1])
    """
    found_indices = []
    if isinstance(candidates_list, str):
        candidates_list = (candidates_list,)
    assert isinstance(candidates_list, (list, tuple)), "candidates_list must be a list, tuple or string."

    for candidate in candidates_list:
        # Compile regex pattern for efficiency
        try:
            pattern = re.compile(candidate)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{candidate}': {e}") from e

        for i, name in enumerate(data_base):
            if fullmatch and pattern.fullmatch(name):
                found_indices.append(i)
            elif not fullmatch and pattern.search(name):
                found_indices.append(i)

    # Remove duplicates and sort
    found_indices = sorted(set(found_indices))
    return torch.tensor(found_indices, dtype=torch.int32, requires_grad=False)


def pattern_match(sub_names: dict[str, any], all_names: list[str]) -> dict[str, any]:
    """Pattern match the sub_names to all_names using regex."""
    matched_names = {_key: 0.0 for _key in all_names}
    for sub_key, sub_val in sub_names.items():
        pattern = re.compile(sub_key)
        for name in all_names:
            if pattern.fullmatch(name):
                matched_names[name] = sub_val
    return matched_names


def get_reward_fn(target: str, reward_functions: list[Callable] | str) -> Callable:
    """Resolve a reward function by name from a list or module path."""
    if isinstance(reward_functions, (list, tuple)):
        fn = next((f for f in reward_functions if f.__name__ == target), None)
    elif isinstance(reward_functions, str):
        reward_module = __import__(reward_functions, fromlist=[target])
        fn = getattr(reward_module, target, None)
    else:
        raise ValueError("reward_functions should be a list of functions or a string module path")
    if fn is None:
        raise KeyError(f"No reward function named '{target}'")
    return fn


def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
    """Construct arguments to `Vec` according to axis index."""
    zs = torch.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = torch.where(zs == 1.0, value, zs)
    params[0] = x_value
    return params.tolist()


@lru_cache(maxsize=128)
def hash_names(names: str | tuple[str]) -> str:
    """Hash a string or tuple of strings into a stable key."""
    if isinstance(names, str):
        names = (names,)
    assert isinstance(names, tuple) and all(isinstance(_, str) for _ in names), (
        "body_names must be a string or a list of strings."
    )
    hash_key = "_".join(sorted(names))
    return hash_key


class Indexer:
    """Utility that assigns consecutive slice ranges."""

    def __init__(self):
        self.i = 0

    def take(self, n: int) -> slice:
        """Return slice(i, i+n) and advance internal counter."""
        s = slice(self.i, self.i + n)
        self.i += n
        return s
