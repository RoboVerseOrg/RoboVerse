"""Initialize a task from a config file (XHand version)."""

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

# Ensure the new env class is imported so its @register_task decorators run.
# Adjust the import path if you place xhand_env.py elsewhere.
from metasim.example.example_pack.tasks.dexhands_env import XHandEnv  # noqa: F401
from metasim.task.gym_registration import make_vec, register_all_tasks_with_gym
from metasim.utils import configclass


@configclass
class Args:
    """Arguments for the static scene (XHand)."""

    # Use the new environment
    task: str = "xhand_env"
    # Default robot alias; override with --robot xhand if your alias differs
    robot: str = "xhand_right"

    # Supported simulator backends
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaacgym"

    # Misc
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

# Register tasks with Gym (must happen after the env module import above)
register_all_tasks_with_gym()
env_id = f"RoboVerse/{args.task}"
env = make_vec(
    env_id,
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
    cameras=[],
    device=args.device,
    # For single-env simulators (e.g., mujoco), consider False; for isaacgym True is fine
    prefer_backend_vectorization=True,
)

obs, info = env.reset()

robot = env.scenario.robots[0]
log.info(f"Loaded robot: {robot.name}")
log.info(f"DOFs: {len(robot.joint_limits)} | joints: {list(robot.joint_limits.keys())}")

for step_i in range(1000000):
    # breakpoint()
    # Batch actions: (num_envs, act_dim) — random target pose within per-joint limits
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
