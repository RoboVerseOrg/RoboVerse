# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab (https://github.com/isaac-sim/IsaacLab).
# Changes: only get_checkpoint_path() is Isaac Lab code (copied from isaaclab_tasks.utils.parse_cfg); the remaining helpers are RoboVerse code written for the BeyondMimic port.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.isaaclab

from __future__ import annotations

import argparse
import importlib
import os
import random
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger as log

from metasim.utils.setup_util import get_robot
from metasim.utils.string_util import is_camel_case, is_snake_case, to_camel_case

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.metasim.mdp.commands import MotionCommand


def get_args():
    """Get the command line arguments."""
    parser = argparse.ArgumentParser(description="Arguments for BeyondMimic motion tracking task")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    parser.add_argument("--robots", type=str, default=None, help="Names of the robots to use")
    parser.add_argument("--objects", type=str, default=None, help="Names of the objects to use")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--sim", type=str, default="isaacsim", help="Simulator type")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")

    # for logging
    parser.add_argument(
        "--exp_name", type=str, default="tracking_g1", help="Name of the experiment folder where logs will be stored"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory")
    parser.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    parser.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using WandB or neptune"
    )

    # for loading
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Whether to resume from a checkpoint. Should only be used for training",
    )
    parser.add_argument(
        "--load_run", type=str, default=None, help="Name of the local folder to resume from if not using WandB"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to resume from if not using WandB. Format: model_xxx.pt",
    )

    # evaluation args
    parser.add_argument("--motion_file", type=str, default=None, help="Path to the local motion file")
    parser.add_argument(
        "--wandb_path",
        type=str,
        default=None,
        help="The WandB run path to load from. Format: org/project/run_id(/model_xxx.pt)",
    )

    # training args
    parser.add_argument("--max_iterations", type=int, default=30000, help="Max number of training iterations")
    parser.add_argument("--registry_name", type=str, default=None, help="Name of the WandB registry")  # required

    return parser.parse_args()


def set_seed(seed: int | None = None):
    """Seed will be randomly initialized if it's None."""
    if not seed:
        seed = np.random.randint(0, 10000)
    log.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class(name: str, suffix: str, library="roboverse_learn.rl.beyondmimic"):
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


def make_robots(robots_str: str) -> list[any]:
    """Make the robots."""
    robot_names = robots_str.split()
    robots = []
    for _name in robot_names:
        robots.append(get_robot(_name))
    return robots


def make_objects(objects_str: str) -> list[any]:
    """Make the objects."""
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


def get_body_indexes(command: MotionCommand, body_names: list[str] | None = None) -> list[int]:
    """Get the indexes of the bodies matching the body names."""
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
    """Construct arguments to `Vec` according to axis index."""
    zs = torch.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = torch.where(zs == 1.0, value, zs)
    params[0] = x_value
    return params.tolist()


# copied from `isaaclab_tasks.utils.parse_cfg.py`


def get_checkpoint_path(
    log_path: str,
    run_dir: str = ".*",
    checkpoint: str = ".*",
    other_dirs: list[str] | None = None,
    sort_alpha: bool = True,
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Returns:
        The path to the model checkpoint.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError as e:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.") from e

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)
