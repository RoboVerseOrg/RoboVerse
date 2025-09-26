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

from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


@configclass
class Args:
    """Arguments for the static scene."""

    ## Handlers
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
        "mjx",
    ] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

robot = RobotCfg(
    name="new_robot_h1",
    num_joints=26,
    usd_path="roboverse_data/robots/h1/usd/h1.usd",
    mjcf_path="roboverse_data/robots/h1/mjcf/h1.xml",
    urdf_path="roboverse_data/robots/h1/urdf/h1.urdf",
    enabled_gravity=True,
    fix_base_link=False,
    enabled_self_collisions=False,
    isaacgym_flip_visual_attachments=False,
    collapse_fixed_joints=True,
    actuators={
        "left_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "left_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "left_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "right_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "right_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "right_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "torso": BaseActuatorCfg(stiffness=300, damping=6),
        "left_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "left_elbow": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "right_elbow": BaseActuatorCfg(stiffness=100, damping=2),
    },
    joint_limits={
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
    },
    control_type={
        "left_hip_yaw": "position",
        "left_hip_roll": "position",
        "left_hip_pitch": "position",
        "left_knee": "position",
        "left_ankle": "position",
        "right_hip_yaw": "position",
        "right_hip_roll": "position",
        "right_hip_pitch": "position",
        "right_knee": "position",
        "right_ankle": "position",
        "torso": "position",
        "left_shoulder_pitch": "position",
        "left_shoulder_roll": "position",
        "left_shoulder_yaw": "position",
        "left_elbow": "position",
        "right_shoulder_pitch": "position",
        "right_shoulder_roll": "position",
        "right_shoulder_yaw": "position",
        "right_elbow": "position",
    },
    default_joint_positions={
        "left_hip_yaw": 0.0,
        "left_hip_roll": 0.0,
        "left_hip_pitch": -0.4,
        "left_knee": 0.8,
        "left_ankle": -0.4,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0,
        "right_hip_pitch": -0.4,
        "right_knee": 0.8,
        "right_ankle": -0.4,
        "torso": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    },
)
# initialize scenario
scenario = ScenarioCfg(
    robots=[robot],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveSphereCfg(
        name="sphere",
        radius=0.1,
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
        urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
        mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
    ),
]


optional_queries = {
    "imu_pos": SitePos("h1/imu"),
    "head_pos": SitePos("h1/head"),
}

log.info(f"Using simulator: {args.sim}")
handler = get_handler(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce": {
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([0.5, 0.2, 0.1]),
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        },
        "robots": {
            "new_robot_h1": {
                "pos": torch.tensor([0.0, 0.0, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "left_hip_yaw": 0.0,
                    "left_hip_roll": 0.0,
                    "left_hip_pitch": -0.4,
                    "left_knee": 0.8,
                    "left_ankle": -0.4,
                    "right_hip_yaw": 0.0,
                    "right_hip_roll": 0.0,
                    "right_hip_pitch": -0.4,
                    "right_knee": 0.8,
                    "right_ankle": -0.4,
                    "torso": 0.0,
                    "left_shoulder_pitch": 0.0,
                    "left_shoulder_roll": 0.0,
                    "left_shoulder_yaw": 0.0,
                    "left_elbow": 0.0,
                    "right_shoulder_pitch": 0.0,
                    "right_shoulder_roll": 0.0,
                    "right_shoulder_yaw": 0.0,
                    "right_elbow": 0.0,
                },
            },
        },
    }
]
handler.set_states(init_states)
obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/13_get_exras_{args.sim}.mp4")
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
                        torch.rand(1).item() * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                        + robot.joint_limits[joint_name][0]
                    )
                    for joint_name in robot.joint_limits.keys()
                }
            }
        }
        for _ in range(scenario.num_envs)
    ]
    handler.set_dof_targets(actions)
    handler.simulate()
    obs_tensor = handler.get_states(mode="tensor")  # get tensor type states
    obs_saver.add(obs_tensor)
    extras = handler.get_extra()
    print("Extras:", extras)  # noqa: T201
    step += 1

obs_saver.save()
