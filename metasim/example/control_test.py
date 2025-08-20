"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        sim: Literal["isaacsim", "isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = (
            "mujoco"
        )
        num_envs: int = 1
        headless: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # initialize scenario
    scenario = ScenarioCfg(
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    log.info(f"Using simulator: {args.sim}")
    task_cls = get_task_class("obj_env")

    # Get default scenario from task class and update with specific parameters
    scenario = task_cls.scenario.update(simulator=args.sim, num_envs=args.num_envs, headless=args.headless, cameras=[])
    # add cameras
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    env = task_cls(scenario, device=device)

    os.makedirs("test_output", exist_ok=True)

    obs, extra = env.reset()
    ## Main loop
    obs_saver = ObsSaver(video_path="test_output/test_control.mp4")
    obs_saver.add(obs)

    step = 0
    robot = scenario.robots[0]
    for _ in range(100):
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        joint_name: (
                            torch.rand(1).item()
                            * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                            + robot.joint_limits[joint_name][0]
                        )
                        for joint_name in robot.joint_limits.keys()
                    }
                }
            }
            for _ in range(scenario.num_envs)
        ]
        flatten_obs, reward, terminated, time_out, info = env.step(actions)
        raw_obs = env.handler.get_states()
        obs_saver.add(raw_obs)
        step += 1

    obs_saver.save()
