"""Save simulation trajectory as Rerun recording file (.rrd) - CPU-only version.

This is a simplified version that doesn't require GPU or IK solvers (pyroki/curobo).
It generates random/sinusoidal joint motions directly.

Usage:
    python get_started/rerun/save_trajectory_simple.py --sim mujoco --output trajectory.rrd

    # Replay the saved recording:
    rerun trajectory.rrd
"""

from __future__ import annotations

import math
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
    """Arguments for saving trajectory recording (CPU-only, no IK solver needed)."""

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

    spawn_viewer: bool = False
    """Whether to spawn Rerun viewer during recording."""

    motion_type: Literal["sinusoidal", "random"] = "sinusoidal"
    """Type of joint motion to generate."""


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


def generate_franka_joint_targets(step: int, num_steps: int, motion_type: str) -> dict:
    """Generate joint targets for Franka robot without IK solver.

    Args:
        step: Current step number
        num_steps: Total number of steps
        motion_type: "sinusoidal" or "random"

    Returns:
        Dictionary mapping joint names to target positions
    """
    # Franka joint limits (approximate, in radians)
    joint_limits = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    }

    # Default rest positions
    rest_positions = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
    }

    t = step / num_steps  # Normalized time [0, 1]

    targets = {}

    if motion_type == "sinusoidal":
        # Smooth sinusoidal motion around rest positions
        # Each joint oscillates with different frequencies and phases
        for i, (joint_name, (lower, upper)) in enumerate(joint_limits.items()):
            rest = rest_positions[joint_name]
            amplitude = min(0.5, (upper - lower) * 0.15)  # Small amplitude for safety
            frequency = 1.0 + i * 0.3  # Different frequency for each joint
            phase = i * 0.5  # Phase offset

            targets[joint_name] = rest + amplitude * math.sin(2 * math.pi * frequency * t + phase)

    elif motion_type == "random":
        # Random walk around rest positions
        import random

        random.seed(step)  # Reproducible random motion

        for joint_name, (lower, upper) in joint_limits.items():
            rest = rest_positions[joint_name]
            # Random offset within 20% of range, smoothed
            offset_range = (upper - lower) * 0.1
            offset = random.uniform(-offset_range, offset_range)

            # Smooth it by mixing with rest position based on time
            smooth_factor = 0.5 + 0.5 * math.sin(2 * math.pi * t)
            targets[joint_name] = rest + offset * smooth_factor

    # Keep fingers at fixed position
    targets["panda_finger_joint1"] = 0.04
    targets["panda_finger_joint2"] = 0.04

    return targets


def main():
    args = tyro.cli(Args)

    log.info(f"Recording trajectory to: {args.output}")
    log.info(f"Simulation steps: {args.num_steps}")
    log.info(f"Motion type: {args.motion_type}")
    log.info("NOTE: This is the CPU-only version (no GPU/IK solver required)")

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
        app_name="Trajectory Recording (CPU)",
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
    # Run Simulation and Record (No IK solver needed!)
    # ========================================================================
    log.info("Starting trajectory recording...")

    robot = scenario.robots[0]

    for step in range(args.num_steps):
        visualizer.set_time(step)

        # Generate joint targets directly (no IK needed)
        joint_targets = generate_franka_joint_targets(step, args.num_steps, args.motion_type)

        # Convert to action format expected by handler (needs "dof_pos_target" key)
        actions = {robot.name: {"dof_pos_target": joint_targets}}

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
