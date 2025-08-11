"""Initialize a task from a config file."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import gymnasium as gym
import numpy as np
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

    task: str = "reach_origin"
    robot: str = "franka"
    ## Handlers
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mjx"

    ## Others
    num_envs: int = 100
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
env = gym.make_vec(
    env_id,
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
    cameras=[],
    device=args.device,
    prefer_backend_vectorization=False,
)

obs, info = env.reset()
for step_i in range(100):
    # batch actions: (num_envs, act_dim)
    actions = np.stack(
        [env.single_action_space.sample() for _ in range(env.num_envs)],
        axis=0,
    ).astype(np.float32, copy=False)
    obs, reward, terminated, truncated, info = env.step(actions)
env.close()
