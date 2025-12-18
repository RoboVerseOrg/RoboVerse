"""Save simulation trajectory as Rerun recording file (.rrd).

This example demonstrates:
- Running a dynamic simulation with robot motion
- Recording all states to Rerun timeline
- Saving as .rrd file for later replay

Usage:
    python get_started/rerun/save_trajectory.py --sim mujoco --output trajectory.rrd

    # Replay the saved recording:
    rerun trajectory.rrd
"""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

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
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.hf_util import check_and_download_recursive
from metasim.utils.setup_util import get_handler
from metasim.utils.state import state_tensor_to_nested


@configclass
class Args:
    """Arguments for saving trajectory recording."""

    robot: str = "franka"
    """Robot to use."""

    sim: Literal[
        "isaacsim",
        "isaacgym",
        "isaaclab",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
    ] = "mujoco"
    """Simulator backend."""

    num_envs: int = 1
    """Number of parallel environments."""

    output: str = "trajectory.rrd"
    """Output path for the .rrd recording file."""

    num_steps: int = 200
    """Number of simulation steps to record."""

    solver: Literal["curobo", "pyroki"] = "pyroki"
    """IK solver to use."""

    spawn_viewer: bool = False
    """Whether to spawn Rerun viewer during recording."""


def extract_states_from_obs(obs, handler, key):
    """Extract states from observation tensor."""
    env_states = state_tensor_to_nested(handler, obs)
    result = {}
    if env_states and len(env_states) > 0:
        state = env_states[0]
        if key in state:
            for name, item in state[key].items():
                state_dict = {}
                if "pos" in item and item["pos"] is not None:
                    state_dict["pos"] = (
                        item["pos"].cpu().numpy().tolist() if hasattr(item["pos"], "cpu") else list(item["pos"])
                    )
                if "rot" in item and item["rot"] is not None:
                    state_dict["rot"] = (
                        item["rot"].cpu().numpy().tolist() if hasattr(item["rot"], "cpu") else list(item["rot"])
                    )
                if "dof_pos" in item and item["dof_pos"] is not None:
                    state_dict["dof_pos"] = item["dof_pos"]
                result[name] = state_dict
    return result


def download_urdf_files(scenario):
    """Download URDF files for visualization."""
    urdf_paths = []
    for obj in scenario.objects:
        if hasattr(obj, "urdf_path") and obj.urdf_path:
            urdf_paths.append(obj.urdf_path)
    for robot in scenario.robots:
        if hasattr(robot, "urdf_path") and robot.urdf_path:
            urdf_paths.append(robot.urdf_path)
    if urdf_paths:
        log.info(f"Downloading {len(urdf_paths)} URDF files...")
        check_and_download_recursive(urdf_paths, n_processes=16)


def main():
    args = tyro.cli(Args)

    log.info(f"Recording trajectory to: {args.output}")
    log.info(f"Simulation steps: {args.num_steps}")

    # ========================================================================
    # Setup Scenario
    # ========================================================================
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=True,
        num_envs=args.num_envs,
    )

    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=640,
            height=480,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.06, 0.06, 0.06),
            color=[1.0, 0.3, 0.3],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.04,
            color=[0.3, 0.3, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bbq_sauce",
            scale=(1.0, 1.0, 1.0),
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
    ]

    handler = get_handler(scenario)

    # Set initial states
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.4, 0.2, 0.03]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.5, -0.2, 0.04]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.6, 0.0, 0.14]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
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
    ]

    handler.set_states(init_states * scenario.num_envs)
    obs = handler.get_states(mode="tensor")

    # ========================================================================
    # Setup Rerun Recording
    # ========================================================================
    from metasim.utils.rerun.rerun_util import RerunVisualizer

    download_urdf_files(scenario)

    visualizer = RerunVisualizer(
        app_name="Trajectory Recording",
        spawn=args.spawn_viewer,
        save_path=args.output,
    )
    visualizer.add_frame("world/origin")

    # Initial visualization
    default_object_states = extract_states_from_obs(obs, handler, "objects")
    default_robot_states = extract_states_from_obs(obs, handler, "robots")
    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    # ========================================================================
    # Setup IK Solver
    # ========================================================================
    from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver

    robot = scenario.robots[0]
    ik_solver = setup_ik_solver(robot, args.solver)

    # ========================================================================
    # Run Simulation and Record
    # ========================================================================
    log.info("Starting trajectory recording...")

    trajectory_points = []

    for step in range(args.num_steps):
        states = handler.get_states(mode="tensor")
        visualizer.set_time(step)

        # Generate circular motion for end effector
        t = step / args.num_steps * 2 * 3.14159
        radius = 0.15
        x_target = 0.4 + radius * torch.cos(torch.tensor(t))
        y_target = radius * torch.sin(torch.tensor(t))
        z_target = 0.4 + 0.1 * torch.sin(torch.tensor(t * 2))

        ee_pos_target = torch.tensor([[x_target, y_target, z_target]], device="cuda:0").repeat(args.num_envs, 1)
        ee_quat_target = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * args.num_envs, device="cuda:0")

        # Log target position
        target_pos = ee_pos_target[0].cpu().numpy().tolist()
        trajectory_points.append(target_pos)
        visualizer.log_trajectory_point("ee_target", target_pos, color=[255, 100, 100])

        # Solve IK
        curr_robot_q = states.robots[robot.name].joint_pos.cuda()
        q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

        # Gripper control
        gripper_binary = torch.ones(scenario.num_envs, device="cuda:0")
        gripper_widths = process_gripper_command(gripper_binary, robot, "cuda:0")

        # Apply actions
        actions = ik_solver.compose_joint_action(q_solution, gripper_widths, curr_robot_q, return_dict=True)
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")

        # Settle physics on first step
        if step == 0:
            for _ in range(30):
                handler.simulate()
                obs = handler.get_states(mode="tensor")

        # Update visualization
        object_states = extract_states_from_obs(obs, handler, "objects")
        robot_states = extract_states_from_obs(obs, handler, "robots")

        for name, state in object_states.items():
            visualizer.update_item_pose(name, state)
        for name, state in robot_states.items():
            visualizer.update_item_pose(name, state)

        if step % 20 == 0:
            log.info(f"Recording step {step}/{args.num_steps}")

    # Log complete trajectory
    if trajectory_points:
        visualizer.log_trajectory("ee_trajectory", trajectory_points, color=[100, 255, 100])

    visualizer.close()

    log.info("")
    log.info("=" * 60)
    log.info("Recording Complete!")
    log.info("=" * 60)
    log.info(f"Saved to: {args.output}")
    log.info(f"Total steps: {args.num_steps}")
    log.info("")
    log.info("To replay the recording:")
    log.info(f"  rerun {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
