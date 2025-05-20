"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


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

import rootutils
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from scipy.spatial.transform import Rotation as R

from get_started.utils import ObsSaver
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils import configclass
from metasim.utils.kinematics_utils import get_curobo_models
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

# initialize scenario
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.0, 1.5), look_at=(1.0, 0.0, 0.0))]
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -0.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = []

log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {},
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
    for _ in range(args.num_envs)
]

robot = scenario.robot
*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_open_q)

obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/motion_planning/0_franka_planning_{args.sim}.mp4")
obs_saver.add(obs)


def move_to_pose(
    obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=10, open_gripper=False
):
    """Move the robot to the target pose."""
    curr_robot_q = obs.robots[robot.name].joint_pos

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    ]
    for i in range(steps):
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)
    return obs


step = 0
robot_joint_limits = scenario.robot.joint_limits
for step in range(4):
    log.debug(f"Step {step}")
    states = env.handler.get_states()
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    rotation_transform_for_franka = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )
    if step == 0:
        gripper_out = torch.tensor([0.0, 0.0, -1.0])
        gripper_long = torch.tensor([0.0, 1.0, 0.0])
        gripper_short = torch.tensor([1.0, 0.0, 0.0])
    elif step == 1:
        gripper_out = torch.tensor([1.0, 0.0, 0.0])
        gripper_long = torch.tensor([0.0, 1.0, 0.0])
        gripper_short = torch.tensor([0.0, 0.0, 1.0])
    elif step == 2:
        gripper_out = torch.tensor([0.0, -1.0, 0.0])
        gripper_long = torch.tensor([1.0, 0.0, 0.0])
        gripper_short = torch.tensor([0.0, 0.0, 1.0])
    elif step == 3:
        gripper_out = torch.tensor([0.0, 0.0, 1.0])
        gripper_long = torch.tensor([0.0, 1.0, 0.0])
        gripper_short = torch.tensor([-1.0, 0.0, 0.0])
    log.info(f"gripper_out: {gripper_out}, gripper_long: {gripper_long}, gripper_short: {gripper_short}")
    rotation_target = torch.stack(
        [
            gripper_out + 1e-4,
            gripper_long + 1e-4,
            gripper_short + 1e-4,
        ],
        dim=0,
    ).float()
    rotation = rotation_target @ rotation_transform_for_franka

    quat = R.from_matrix(rotation).as_quat()
    position = torch.tensor([0.6, 0.0, 0.6], device="cuda:0")

    ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
    ee_quat_target = torch.zeros((args.num_envs, 4), device="cuda:0")

    ee_pos_target[0] = torch.tensor(position, device="cuda:0")
    ee_quat_target[0] = torch.tensor(quat, device="cuda:0")

    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=100, open_gripper=True
    )
    step += 1

obs_saver.save()
