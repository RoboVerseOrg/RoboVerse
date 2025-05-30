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

from get_started.utils import ObsSaver
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

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

# initialize scenario
scenario = ScenarioCfg(
    robots=["shadow_hand", "shadow_hand_1"],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, -0.2, 0.0))]

scenario.objects = [
    RigidObjCfg(
        name="cube",
        scale=(1, 1, 1),
        physics=PhysicStateType.RIGIDBODY,
        # usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="roboverse_data/assets/bidex/objects/cube_multicolor.urdf",
        # mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.0, -0.39, 0.54]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "shadow_hand": {
                "pos": torch.tensor([0.0, 0.0, 0.5]),
                "rot": torch.tensor([0.0, 0.0, -0.707, 0.707]),
                "dof_pos": {
                    "robot0_WRJ1": 0.0,
                    "robot0_WRJ0": 0.0,
                    "robot0_FFJ3": 0.0,
                    "robot0_FFJ2": 0.0,
                    "robot0_FFJ1": 0.0,
                    "robot0_FFJ0": 0.0,
                    "robot0_MFJ3": 0.0,
                    "robot0_MFJ2": 0.0,
                    "robot0_MFJ1": 0.0,
                    "robot0_MFJ0": 0.0,
                    "robot0_RFJ3": 0.0,
                    "robot0_RFJ2": 0.0,
                    "robot0_RFJ1": 0.0,
                    "robot0_RFJ0": 0.0,
                    "robot0_LFJ4": 0.0,
                    "robot0_LFJ3": 0.0,
                    "robot0_LFJ2": 0.0,
                    "robot0_LFJ1": 0.0,
                    "robot0_LFJ0": 0.0,
                    "robot0_THJ4": 0.0,
                    "robot0_THJ3": 0.0,
                    "robot0_THJ2": 0.0,
                    "robot0_THJ1": 0.0,
                    "robot0_THJ0": 0.0,
                },
            },
            "shadow_hand_1": {
                "pos": torch.tensor([0.0, -1.0, 0.5]),
                "rot": torch.tensor([-0.707, 0.707, 0.0, 0.0]),
                "dof_pos": {
                    "robot1_WRJ1": 0.0,
                    "robot1_WRJ0": 0.0,
                    "robot1_FFJ3": 0.0,
                    "robot1_FFJ2": 0.0,
                    "robot1_FFJ1": 0.0,
                    "robot1_FFJ0": 0.0,
                    "robot1_MFJ3": 0.0,
                    "robot1_MFJ2": 0.0,
                    "robot1_MFJ1": 0.0,
                    "robot1_MFJ0": 0.0,
                    "robot1_RFJ3": 0.0,
                    "robot1_RFJ2": 0.0,
                    "robot1_RFJ1": 0.0,
                    "robot1_RFJ0": 0.0,
                    "robot1_LFJ4": 0.0,
                    "robot1_LFJ3": 0.0,
                    "robot1_LFJ2": 0.0,
                    "robot1_LFJ1": 0.0,
                    "robot1_LFJ0": 0.0,
                    "robot1_THJ4": 0.0,
                    "robot1_THJ3": 0.0,
                    "robot1_THJ2": 0.0,
                    "robot1_THJ1": 0.0,
                    "robot1_THJ0": 0.0,
                },
            },
        },
    }
]
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/8_shadowhand_loading_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot_joint_limits = {}
for robot in scenario.robots:
    robot_joint_limits.update(robot.joint_limits)
for _ in range(100):
    log.debug(f"Step {step}")
    actions = [
        {
            "dof_pos_target": {
                joint_name: (
                    torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
                    + robot_joint_limits[joint_name][0]
                )
                for joint_name in robot_joint_limits.keys()
                # if scenario.robot.actuators[joint_name].fully_actuated
            }
        }
        for _ in range(scenario.num_envs)
    ]
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs)
    step += 1

obs_saver.save()
