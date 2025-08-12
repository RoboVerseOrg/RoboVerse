"""Initialize a task from a config file."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


import torch

from metasim.utils import configclass
from tasks.gym_registration import make_vec, register_all_tasks_with_gym


@configclass
class Args:
    """Arguments for the static scene."""

    task: str = "obj_env"
    robot: str = "franka"
    ## Handlers
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaacgym"

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

register_all_tasks_with_gym()

env_id = f"RoboVerse/{args.task}"
env = make_vec(
    env_id,
    num_envs=args.num_envs,
    robots=[args.robot],
    simulator=args.sim,
    headless=args.headless,
    cameras=[],
    device=args.device,
)
obs, info = env.reset()

robot = env.scenario.robots[0]
for step_i in range(100):
    # batch actions: (num_envs, act_dim)
    actions = [
        {
            robot.name: {
                "dof_pos_target": {
                    joint_name: (
                        torch.rand(1).item() * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                        + robot.joint_limits[joint_name][0]
                    )
                    for joint_name in robot.joint_limits.keys()
                }
            }
        }
        for _ in range(args.num_envs)
    ]
    obs, reward, terminated, truncated, info = env.step(actions)
env.close()
