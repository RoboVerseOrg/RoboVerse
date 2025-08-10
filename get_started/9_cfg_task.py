"""Initialize a task from a config file."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.utils import configclass

# from metasim.utils.setup_util import register_task
from roboverse_learn.tasks.registry import load_task
from scenario_cfg.scenario import ScenarioCfg


@configclass
class Args:
    """Arguments for the static scene."""

    task: str = "close_box"
    robot: str = "franka"
    ## Handlers
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False
    device: str = "cuda"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
TASK_NAME = args.task
NUM_ENVS = args.num_envs
SIM = args.sim

scenario = ScenarioCfg(
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=True,
    cameras=[],
)

env = load_task(args.task, scenario, device=args.device)

actions = np.zeros((NUM_ENVS, len(scenario.robots[0].joint_limits)))
obs = env.reset()
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
    obs, _, _, _, _ = env.step(action_dicts)
