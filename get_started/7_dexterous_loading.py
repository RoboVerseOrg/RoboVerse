"""This script provides a minimal example of loading dexterous hand."""

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

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    # TODO currently, only support for isaacgym. Adding support for other simulators.
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

robot = "unitree_dex3_1"
# initialize scenario
scenario = ScenarioCfg(
    robot=robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {},
        "robots": {
            "unitree_dex3_1": {
                "pos": torch.tensor([0.0, 0.0, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "left_hand_thumb_0_joint": 0.0,
                    "left_hand_thumb_1_joint": 0.0,
                    "left_hand_thumb_2_joint": 0.8,
                    "left_hand_middle_0_joint": -0.8,
                    "left_hand_middle_1_joint": -0.4,
                    "left_hand_index_0_joint": -0.8,
                    "left_hand_index_1_joint": -0.8,
                },
            },
        },
    }
]
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)


step = 0
robot_joint_limits = scenario.robot.joint_limits
for _ in range(10000):
    log.debug(f"Step {step}")
    actions = [
        {
            "dof_pos_target": {
                joint_name: (
                    torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
                    + robot_joint_limits[joint_name][0]
                )
                for joint_name in robot_joint_limits.keys()
            }
        }
        for _ in range(scenario.num_envs)
    ]
    obs, reward, success, time_out, extras = env.step(actions)
    step += 1
