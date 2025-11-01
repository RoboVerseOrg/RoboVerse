"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

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
    headless: bool = True
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# IK solver imports are now handled in the unified solver
log.info(f"Using IK solver: {args.solver}")

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
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

log.info(f"Using simulator: {args.sim}")
handler = get_handler(scenario)

if args.robot == "franka":
    robot_dict = {
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
        }
    }
elif args.robot == "kinova_gen3_robotiq_2f85":
    robot_dict = {
        "kinova_gen3_robotiq_2f85": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "joint_1": 0.0,
                "joint_2": math.pi / 6,
                "joint_3": 0.0,
                "joint_4": math.pi / 2,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_7": 0.0,
                "finger_joint": 0.0,
            },
        }
    }
else:
    robot_dict = {}

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
        "robots": robot_dict,
    }
    for _ in range(args.num_envs)
]


robot = scenario.robots[0]

from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver

# Setup unified IK solver
ik_solver = setup_ik_solver(robot, args.solver)

handler.set_states(init_states)
obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(200):
    log.debug(f"Step {step}")
    states = handler.get_states()

    if scenario.robots[0].name == "franka":
        x_target = 0.3 + 0.1 * (step / 100)
        y_target = 0.5 - 0.5 * (step / 100)
        z_target = 0.6 - 0.2 * (step / 100)

        # Randomly assign x/y/z target for each env
        def pick_device():
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Optional: Apple Silicon (PyTorch 1.12+ with MPS)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        device = pick_device()
        ee_pos_target = torch.zeros((args.num_envs, 3), device=device)
        for i in range(args.num_envs):
            if i % 3 == 0:
                ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device=device)
            elif i % 3 == 1:
                ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device=device)
            else:
                ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device=device)
        ee_quat_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
            device=device,
        )
    elif scenario.robots[0].name == "kinova_gen3_robotiq_2f85":
        ee_pos_target = torch.tensor([[0.2 + 0.2 * (step / 100), 0.0, 0.4]], device=device).repeat(args.num_envs, 1)
        ee_quat_target = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]] * args.num_envs,
            device=device,
        )

    # Get current robot state for seeding
    # IK solver expects original joint order, but state uses alphabetical order
    reorder_idx = handler.get_joint_reindex(scenario.robots[0].name)
    inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
    curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]

    # Solve IK
    q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

    # Process gripper command (fixed open position)
    gripper_binary = torch.ones(scenario.num_envs, device=device)  # all open
    gripper_widths = process_gripper_command(gripper_binary, robot, device)
    # Compose full joint command
    actions = ik_solver.compose_joint_action(q_solution, gripper_widths, curr_robot_q, return_dict=True)

    handler.set_dof_targets(actions)
    handler.simulate()
    obs = handler.get_states(mode="tensor")
    # obs, reward, success, time_out, extras = handler.step(actions)

    obs_saver.add(obs)
    step += 1

obs_saver.save()
