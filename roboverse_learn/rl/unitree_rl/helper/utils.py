from __future__ import annotations

from typing import Callable
import re
import os
import copy
import argparse
import datetime
import importlib
from loguru import logger as log
from functools import lru_cache

import random
import torch
import numpy as np

from metasim.utils.setup_util import get_robot
from metasim.utils.string_util import is_camel_case, is_snake_case, to_camel_case
from metasim.scenario.scenario import ScenarioCfg


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
                    parser.add_argument(
                        argument["name"], type=argument["type"], help=help_str
                    )
            elif "action" in argument:
                parser.add_argument(
                    argument["name"], action=argument["action"], help=help_str
                )

        else:
            log.error(
                "ERROR: command line argument name, type/action must be defined, argument not added to parser"
            )
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
            "name": "--resume",
            "type": str,
            "default": None,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--checkpoint",
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
    if seed == -1:
        seed = np.random.randint(0, 10000)
    log.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_log_dir(task_name: str, now=None) -> str:
    """Get the log directory."""
    if now is None:
        now = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    log_dir = f"./outputs/unitree_rl/{task_name}/{now}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log.info("Log directory: {}", log_dir)
    return log_dir


def get_class(name: str, suffix: str, library="roboverse_learn.rl.unitree_rl"):
    """Get the class wrappers.
    Example:
        get_class("ReachOrigin", "Cfg") -> ReachOriginCfg
        get_class("reach_origin", "Cfg") -> ReachOriginCfg
    """
    if is_camel_case(name):
        task_name_camel = name
    elif is_snake_case(name):
        task_name_camel = to_camel_case(name)

    wrapper_module = importlib.import_module(library)
    wrapper_cls = getattr(wrapper_module, f"{task_name_camel}{suffix}")
    return wrapper_cls


def get_load_path(load_root: str, checkpoint: int | str = None) -> str:
    """Get the path to load the model from."""
    if isinstance(checkpoint, int):
        if checkpoint == -1:
            models = [
                file
                for file in os.listdir(load_root)
                if "model" in file and file.endswith(".pt")
            ]
            models.sort(key=lambda m: f"{m!s:0>15}")
            model = models[-1]
            load_path = f"{load_root}/{model}"
        else:
            load_path = f"{load_root}/model_{checkpoint}.pt"
    else:
        load_path = f"{load_root}/{checkpoint}.pt"
    log.info(f"Loading checkpoint {checkpoint} from {load_root}")
    return load_path


def make_robots(robots_str: str) -> list[any]:
    robot_names = robots_str.split()
    robots = []
    for _name in robot_names:
        robots.append(get_robot(_name))
    return robots


def make_objects(objects_str: str) -> list[any]:
    object_names = objects_str.split()
    objects = []
    for _name in object_names:
        objects.append(
            get_class(
                _name,
                suffix="Cfg",
                library="roboverse_learn.rl.unitree_rl.configs.cfg_objects",
            )()
        )
    return objects


def find_unique_candidate(candidates: list[any], data_base: list[any]) -> int:
    found_candidates = []
    found_indices = []

    for candidate in candidates:
        if candidate in data_base:
            found_candidates.append(candidate)
            found_indices.append(data_base.index(candidate))

    if len(found_candidates) == 0:
        raise ValueError(f"None of the candidates {candidates} found in {data_base}")
    elif len(found_candidates) > 1:
        raise ValueError(
            f"Multiple candidates found: {found_candidates}. Only one naming convention should be used."
        )

    return found_indices[0]


def get_indices_from_substring(
    candidates_list: list[str] | tuple[str] | str,
    data_base: list[str],
    fullmatch: bool = True,
) -> torch.Tensor:
    """Get indices of items matching the candidates patterns.

    Args:
        candidates_list: Single pattern or list of patterns (supports regex if use_regex=True)
        data_base: List of names to search in
        use_regex: If True, treat candidates as regex patterns. If False, use substring matching.

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
    assert isinstance(
        candidates_list, (list, tuple)
    ), "candidates_list must be a list, tuple or string."

    for candidate in candidates_list:
        # Compile regex pattern for efficiency
        try:
            pattern = re.compile(candidate)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{candidate}': {e}")

        for i, name in enumerate(data_base):
            if fullmatch and pattern.fullmatch(name):
                found_indices.append(i)
            elif not fullmatch and pattern.search(name):
                found_indices.append(i)

    # Remove duplicates and sort
    found_indices = sorted(set(found_indices))
    return torch.tensor(found_indices, dtype=torch.int32, requires_grad=False)


def reindex_func(
    data: torch.Tensor, new_idx: torch.Tensor, start_idx: int | torch.Tensor
) -> torch.Tensor:
    assert data.dim() == 2, "data must be a 2D tensor"
    assert new_idx.dim() == 1, "new_idx must be a 1D tensor"
    reindex_length = len(new_idx)
    for start in start_idx:
        data[:, start : start + reindex_length] = data[
            :, start : start + reindex_length
        ][:, new_idx]
    return data


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            "hidden_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )
        self.register_buffer(
            "cell_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        if not path.endswith(".pt"):
            path = os.path.join(path, "policy.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def export_policy_as_jit(actor, path, filename=None):
    """Export the policy as a JIT model."""
    model = copy.deepcopy(actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def get_export_jit_path(load_root: str, scenario: ScenarioCfg) -> str:
    """Get the path to export the JIT model."""
    exported_root_dir = f"{load_root}/exported"
    os.makedirs(exported_root_dir, exist_ok=True)
    return f"{load_root}/exported/model_exported_jit.pt"


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
        raise ValueError(
            "reward_functions should be a list of functions or a string module path"
        )
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
    if isinstance(names, str):
        names = (names,)
    assert isinstance(names, tuple) and all(
        isinstance(_, str) for _ in names
    ), "body_names must be a string or a list of strings."
    hash_key = "_".join(sorted(names))
    return hash_key
