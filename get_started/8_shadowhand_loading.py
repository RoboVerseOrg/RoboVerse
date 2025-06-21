"""This script provides a minimal example of loading dexterous hand."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import numpy as np
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
from metasim.cfg.sensors import ContactForceSensorCfg, PinholeCameraCfg
from metasim.cfg.simulator_params import SimParamCfg
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
sim_params = SimParamCfg()
sim_params.dt = 1.0 / 60.0
sim_params.bounce_threshold_velocity = 0.2
sim_params.contact_offset = 0.002
sim_params.num_velocity_iterations = 0
sim_params.num_threads = 4
sim_params.use_gpu_pipeline = True
sim_params.use_gpu = True
sim_params.substeps = 2


# initialize scenario
scenario = ScenarioCfg(
    robots=["shadow_hand_right", "shadow_hand_left"],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    sim_params=sim_params,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, -0.2, 0.0))]
scenario.sensors = [
    ContactForceSensorCfg(base_link=("shadow_hand_right", "robot0_ffdistal"), source_link=None, name="test_sensor")
]

scenario.objects = [
    RigidObjCfg(
        name="cube",
        scale=(1, 1, 1),
        physics=PhysicStateType.RIGIDBODY,
        urdf_path="roboverse_data/assets/bidex/objects/cube_multicolor.urdf",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.01, -0.385, 0.54]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "shadow_hand_right": {
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
            "shadow_hand_left": {
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
joint_names = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
    "robot1_WRJ1",
    "robot1_WRJ0",
    "robot1_FFJ3",
    "robot1_FFJ2",
    "robot1_FFJ1",
    "robot1_MFJ3",
    "robot1_MFJ2",
    "robot1_MFJ1",
    "robot1_RFJ3",
    "robot1_RFJ2",
    "robot1_RFJ1",
    "robot1_LFJ4",
    "robot1_LFJ3",
    "robot1_LFJ2",
    "robot1_LFJ1",
    "robot1_THJ4",
    "robot1_THJ3",
    "robot1_THJ2",
    "robot1_THJ1",
    "robot1_THJ0",
]
traj = np.load("roboverse_data/trajs/bidex/shadow_hand_over/test_traj.npy", allow_pickle=True)
traj = np.clip(traj, -1.0, 1.0)  # Ensure the trajectory is within the joint limits
for _ in range(len(traj)):
    log.debug(f"Step {step}")
    actions = [
        {
            robot.name: {
                "dof_pos_target": {
                    joint_name: (
                        0.5
                        * (traj[step][joint_names.index(joint_name)] + 1.0)
                        * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                        + robot.joint_limits[joint_name][0]
                    )
                    for joint_name in robot.joint_limits.keys()
                    if robot.actuators[joint_name].fully_actuated
                }
            }
            for robot in scenario.robots
        }
        for _ in range(scenario.num_envs)
    ]
    # print(actions[0]["dof_pos_target"])
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs)
    step += 1

obs_saver.save()
