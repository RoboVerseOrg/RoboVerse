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

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.gym_registration import make_vec
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver


@configclass
class Args:
    """Arguments for the static scene."""

    task: str = "stack_cube"
    robot: str = "franka"
    ## Handlers
    sim: Literal["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = (
        "isaacsim"
    )

    ## Others
    num_envs: int = 1
    headless: bool = False
    device: str = "cuda"
    save_video: bool = True

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
TASK_NAME = args.task
NUM_ENVS = args.num_envs
SIM = args.sim

# Add camera for video recording if needed
camera = (
    PinholeCameraCfg(
        name="main_camera",
        pos=[4.0, 4.0, 3.0],
        look_at=[0.0, 0.0, 1.0],
        width=640,
        height=480,
        data_types=["rgb"],
    )
    if args.save_video
    else None
)

env_id = f"RoboVerse/{args.task}"


env = make_vec(
    env_id,
    num_envs=args.num_envs,
    simulator=args.sim,
    headless=args.headless,
    cameras=[camera] if args.save_video else [],
    device=args.device,
)
obs, info = env.reset()

# Initialize video saver
obs_saver = None
if args.save_video:
    import os

    os.makedirs("get_started/output", exist_ok=True)
    video_path = f"get_started/output/9_cfg_task_{args.sim}.mp4"
    obs_saver = ObsSaver(video_path=video_path)
    log.info(f"Will save video to: {video_path}")

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

    # Save observations for video
    if obs_saver is not None:
        try:
            raw_states = env.env.handler.get_states()  # Access the underlying simulator
            obs_saver.add(raw_states)
        except Exception as e:
            log.debug(f"Could not get camera data: {e}")

# Save video at the end
if obs_saver is not None:
    obs_saver.save()
    log.info("Video saved successfully!")

try:
    env.close()
except NotImplementedError:
    log.debug("env.close() not implemented, ignoring")
