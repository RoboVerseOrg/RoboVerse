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
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from scipy.spatial.transform import Rotation as R

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

from metasim.utils.ik_solver import setup_ik_solver

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    decimation=4,
)

# add cameras
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.0, 1.5), look_at=(1.0, 0.0, 0.0))]
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -0.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = []

log.info(f"Using simulator: {args.sim}")
handler = get_handler(scenario)

# Select device: use CUDA if available, otherwise fall back to CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

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
robot = scenario.robots[0]

# Setup IK solver, disable seed for pyroki
ik_solver = setup_ik_solver(robot, args.solver, use_seed=False)

handler.set_states(init_states)
obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/motion_planning/0_franka_planning_{args.sim}.mp4")
obs_saver.add(obs)


def move_to_pose(
    obs,
    obs_saver,
    ik_solver,
    robot,
    scenario,
    inverse_reorder_idx,
    ee_pos_target,
    ee_quat_target,
    steps=10,
    open_gripper=False,
):
    """Move the robot to the target pose."""
    # IK solver expects original joint order, but state uses alphabetical order
    curr_robot_q = obs.robots[robot.name].joint_pos[:, inverse_reorder_idx]

    # Solve IK using the unified interface
    q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

    # Process gripper command
    from metasim.utils.ik_solver import process_gripper_command

    gripper_open_tensor = torch.tensor([1.0 if open_gripper else 0.0] * scenario.num_envs, device=ee_pos_target.device)
    gripper_widths = process_gripper_command(gripper_open_tensor, robot, ee_pos_target.device)

    # Compose full joint command
    actions = ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)
    for i in range(steps):
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        obs_saver.add(obs)
    return obs


# Calculate joint reordering once
# IK solver expects original joint order, but state uses alphabetical order
reorder_idx = handler.get_joint_reindex(robot.name)
inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]

step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(4):
    log.debug(f"Step {step}")
    states = handler.get_states()
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
    position = torch.tensor([0.6, 0.0, 0.6], device=device)

    ee_pos_target = torch.zeros((args.num_envs, 3), device=device)
    ee_quat_target = torch.zeros((args.num_envs, 4), device=device)

    # position is already a tensor; move it to the selected device
    ee_pos_target[0] = position.to(device)
    ee_quat_target[0] = torch.tensor(quat, device=device)

    obs = move_to_pose(
        obs,
        obs_saver,
        ik_solver,
        robot,
        scenario,
        inverse_reorder_idx,
        ee_pos_target,
        ee_quat_target,
        steps=100,
        open_gripper=True,
    )
    step += 1

obs_saver.save()
