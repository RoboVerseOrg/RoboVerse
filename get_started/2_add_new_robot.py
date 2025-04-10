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

from get_started.utils import ObsSaver
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaaclab"
    )

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

robot = BaseRobotCfg(
    name="new_robot_h1",
    num_joints=26,
    usd_path="get_started/example_assets/h1/usd/h1.usd",
    mjcf_path="get_started/example_assets/h1/mjcf/h1.xml",
    urdf_path="get_started/example_assets/h1/urdf/h1_wrist.urdf",
    enabled_gravity=True,
    fix_base_link=False,
    enabled_self_collisions=False,
    isaacgym_flip_visual_attachments=False,
    collapse_fixed_joints=True,
    actuators={
        "left_hip_yaw": BaseActuatorCfg(),
        "left_hip_roll": BaseActuatorCfg(),
        "left_hip_pitch": BaseActuatorCfg(),
        "left_knee": BaseActuatorCfg(),
        "left_ankle": BaseActuatorCfg(),
        "right_hip_yaw": BaseActuatorCfg(),
        "right_hip_roll": BaseActuatorCfg(),
        "right_hip_pitch": BaseActuatorCfg(),
        "right_knee": BaseActuatorCfg(),
        "right_ankle": BaseActuatorCfg(),
        "torso": BaseActuatorCfg(),
        "left_shoulder_pitch": BaseActuatorCfg(),
        "left_shoulder_roll": BaseActuatorCfg(),
        "left_shoulder_yaw": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
        "right_shoulder_pitch": BaseActuatorCfg(),
        "right_shoulder_roll": BaseActuatorCfg(),
        "right_shoulder_yaw": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
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
)
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

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
        mjcf_path="get_started/example_assets/cube/cube.mjcf",
    ),
    PrimitiveSphereCfg(
        name="sphere",
        radius=0.1,
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
        mjcf_path="get_started/example_assets/sphere/sphere.mjcf",
    ),
    RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="get_started/example_assets/box_base/usd/box_base.usd",
        urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
        mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

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
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/2_add_new_robot_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot_joint_limits = scenario.robot.joint_limits
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
            }
        }
        for _ in range(scenario.num_envs)
    ]
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs)
    step += 1

obs_saver.save()
