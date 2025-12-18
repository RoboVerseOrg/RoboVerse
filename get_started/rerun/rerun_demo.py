"""Comprehensive Rerun visualization demo with multiple control modes.

This demo supports:
- Static/Dynamic scene visualization
- URDF robot and object visualization
- Primitive shapes (cubes, spheres)
- Trajectory visualization
- Camera image logging

Rerun is an open-source SDK for logging, storing, querying, and visualizing
multimodal data. It provides a powerful viewer with timeline-based exploration.

Usage:
    python get_started/rerun/rerun_demo.py --sim mujoco
    python get_started/rerun/rerun_demo.py --sim pybullet --dynamic
    python get_started/rerun/rerun_demo.py --sim mujoco --save-recording output.rrd
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
    ArticulationObjCfg,
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
    """Arguments for the Rerun demo."""

    robot: str = "franka"
    """Robot to use in the demo."""

    ## Simulator
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
    """Simulator backend to use."""

    num_envs: int = 1
    """Number of parallel environments."""

    headless: bool = True
    """Run simulator headless (use Rerun for visualization)."""

    ## Control modes
    dynamic: bool = False
    """Enable dynamic simulation with IK motion."""

    ## IK solver (only used if dynamic=True)
    solver: Literal["curobo", "pyroki"] = "pyroki"
    """IK solver to use for dynamic motion."""

    ## Recording
    save_recording: str | None = None
    """Path to save Rerun recording file (.rrd). If None, no recording is saved."""

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


def extract_states_from_obs(obs, handler, key):
    """Extract states from observation tensor.

    Args:
        obs: TensorState observation
        handler: Simulator handler
        key: "objects" or "robots"

    Returns:
        dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
    """
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
    """Download URDF files for all objects and robots in the scenario."""
    urdf_paths = []

    for obj in scenario.objects:
        if hasattr(obj, "urdf_path") and obj.urdf_path:
            urdf_paths.append(obj.urdf_path)

    for robot in scenario.robots:
        if hasattr(robot, "urdf_path") and robot.urdf_path:
            urdf_paths.append(robot.urdf_path)

    if urdf_paths:
        log.info(f"Downloading {len(urdf_paths)} URDF files and all related meshes...")
        check_and_download_recursive(urdf_paths, n_processes=16)
        log.info("URDF files and meshes download completed!")


def main():
    args = tyro.cli(Args)

    # ========================================================================
    # Setup Scenario
    # ========================================================================
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    # Add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=640,
            height=480,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    # Add objects for demonstration
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
            scale=(2.0, 2.0, 2.0),
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

    # Set initial states
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.1]),
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
    # Setup Rerun Visualization
    # ========================================================================
    from metasim.utils.rerun.rerun_util import RerunVisualizer

    # Download URDF files before visualization
    download_urdf_files(scenario)

    # Initialize the Rerun visualizer
    visualizer = RerunVisualizer(
        app_name="RoboVerse Demo",
        spawn=True,
        save_path=args.save_recording,
    )
    visualizer.add_frame("world/origin")

    # Extract states from objects and robots
    default_object_states = extract_states_from_obs(obs, handler, "objects")
    default_robot_states = extract_states_from_obs(obs, handler, "robots")

    # Visualize all objects and robots
    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    log.info("Rerun visualization initialized! The viewer should have opened automatically.")

    # Scene info
    scene_info = ["Scene includes:"]
    for obj in scenario.objects:
        scene_info.append(f"  • {obj.name} ({type(obj).__name__})")
    for robot in scenario.robots:
        scene_info.append(f"  • {robot.name} ({type(robot).__name__})")
    log.info("\n".join(scene_info))

    # ========================================================================
    # Dynamic Simulation Loop (if enabled)
    # ========================================================================
    if args.dynamic:
        log.info("Starting dynamic simulation with IK motion...")

        from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver

        robot = scenario.robots[0]

        log.info(f"Using IK solver: {args.solver}")
        ik_solver = setup_ik_solver(robot, args.solver)

        trajectory_points = []

        for step in range(200):
            states = handler.get_states(mode="tensor")
            visualizer.set_time(step)

            # Generate target end-effector pose
            if robot.name == "franka":
                x_target = 0.3 + 0.1 * (step / 100)
                y_target = 0.5 - 0.5 * (step / 100)
                z_target = 0.6 - 0.2 * (step / 100)
                ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
                for i in range(args.num_envs):
                    if i % 3 == 0:
                        ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device="cuda:0")
                    elif i % 3 == 1:
                        ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device="cuda:0")
                    else:
                        ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device="cuda:0")
                ee_quat_target = torch.tensor(
                    [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
                    device="cuda:0",
                )
            else:
                # Default motion for other robots
                ee_pos_target = torch.tensor([[0.3, 0.0, 0.6]], device="cuda:0").repeat(args.num_envs, 1)
                ee_quat_target = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * args.num_envs, device="cuda:0")

            # Log target position as trajectory point
            target_pos = ee_pos_target[0].cpu().numpy().tolist()
            trajectory_points.append(target_pos)
            visualizer.log_trajectory_point("ee_target", target_pos, color=[255, 0, 0])

            # Get current robot state for IK seeding
            curr_robot_q = states.robots[robot.name].joint_pos.cuda()

            # Solve IK
            q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

            # Process gripper command (fixed open position)
            gripper_binary = torch.ones(scenario.num_envs, device="cuda:0")  # all open
            gripper_widths = process_gripper_command(gripper_binary, robot, "cuda:0")

            # Compose full joint command
            actions = ik_solver.compose_joint_action(q_solution, gripper_widths, curr_robot_q, return_dict=True)

            handler.set_dof_targets(actions)
            handler.simulate()
            obs = handler.get_states(mode="tensor")

            # Settle for first step
            if step == 0:
                for _ in range(50):
                    handler.simulate()
                    obs = handler.get_states(mode="tensor")

            # Update visualization
            object_states = extract_states_from_obs(obs, handler, "objects")
            robot_states = extract_states_from_obs(obs, handler, "robots")

            for name, state in object_states.items():
                visualizer.update_item_pose(name, state)
            for name, state in robot_states.items():
                visualizer.update_item_pose(name, state)

            if step % 10 == 0:
                log.info(f"Step {step}/200 completed")

        # Log complete trajectory
        if trajectory_points:
            visualizer.log_trajectory("ee_trajectory", trajectory_points, color=[0, 255, 0])

        log.info("Dynamic simulation completed!")

    # ========================================================================
    # Print Usage Instructions
    # ========================================================================
    log.info("")
    log.info("=" * 70)
    log.info("Rerun Demo Ready!")
    log.info("=" * 70)

    mode_description = "Static Scene" if not args.dynamic else "Dynamic Scene (simulation completed)"
    log.info(f"Mode: {mode_description}")

    log.info("\nRerun Viewer Controls:")
    log.info("  • Rotate: Left mouse drag")
    log.info("  • Pan: Middle mouse drag or Shift+Left drag")
    log.info("  • Zoom: Scroll wheel")
    log.info("  • Timeline: Use the timeline at the bottom to scrub through simulation")

    if args.save_recording:
        log.info(f"\nRecording saved to: {args.save_recording}")
        log.info("You can replay it with: rerun {args.save_recording}")

    log.info("=" * 70)

    # Keep running for static mode
    if not args.dynamic:
        log.info("\nPress Ctrl+C to exit...")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            log.info("\nShutting down...")

    visualizer.close()


if __name__ == "__main__":
    main()
