"""Initialize a task from a config file."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import gymnasium as gym
import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.utils import configclass
from tasks.gym_registration import register_all_tasks_with_gym


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

register_all_tasks_with_gym()

env_id = f"RoboVerse/{args.task}-v0"
env = gym.make(
    env_id,
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
    cameras=[],
    device=args.device,
)

obs, info = env.reset()
for step_i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
