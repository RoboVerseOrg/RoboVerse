"""Initialize a task from a config file."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro
from loguru import logger as log
from packaging.version import Version

from metasim.utils import configclass
from metasim.utils.setup_util import register_task


@configclass
class Args:
    """Arguments for the static scene."""

    task: str = "close_box"

    ## Handlers
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
TASK_NAME = args.task
NUM_ENVS = args.num_envs
SIM = args.sim
register_task(TASK_NAME)
if Version(gym.__version__) < Version("1"):
    metasim_env = gym.make(TASK_NAME, num_envs=NUM_ENVS, sim=SIM)
else:
    metasim_env = gym.make_vec(TASK_NAME, num_envs=NUM_ENVS, sim=SIM)
scenario = metasim_env.scenario

actions = np.zeros((NUM_ENVS, len(scenario.robots[0].joint_limits)))

obs, _ = metasim_env.reset()
for step_i in range(100):
    action_dicts = [
        {
            scenario.robots[0].name: {
                "dof_pos_target": {
                    joint_name: (
                        torch.rand(1).item()
                        * (
                            scenario.robots[0].joint_limits[joint_name][1]
                            - scenario.robots[0].joint_limits[joint_name][0]
                        )
                        + scenario.robots[0].joint_limits[joint_name][0]
                    )
                    for joint_name in scenario.robots[0].joint_limits.keys()
                }
            }
        }
        for _ in range(NUM_ENVS)
    ]
    obs, _, _, _, _ = metasim_env.step(action_dicts)
