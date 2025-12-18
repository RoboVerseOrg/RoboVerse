"""Replay an existing task demo and save as Rerun recording (.rrd).

This example demonstrates:
- Loading and replaying a pre-recorded task trajectory (e.g., stack_cube)
- Recording all states to Rerun timeline
- Saving as .rrd file for later replay

No GPU or IK solver needed - just replays existing trajectories!

Usage:
    # Replay stack_cube task
    python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd

    # Replay close_box task
    python get_started/rerun/replay_task_demo.py --task close_box --sim mujoco --output close_box.rrd

    # With live viewer
    python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --spawn-viewer

    # Replay the saved recording:
    rerun stack_cube.rrd
"""

from __future__ import annotations

import os
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

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_recursive
from metasim.utils.state import state_tensor_to_nested


@configclass
class Args:
    """Arguments for replaying task demo and saving Rerun recording."""

    task: str = "stack_cube"
    """Task name to replay (e.g., stack_cube, close_box, pick_cube, etc.)."""

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

    output: str = "task_replay.rrd"
    """Output path for the .rrd recording file."""

    spawn_viewer: bool = False
    """Whether to spawn Rerun viewer during recording."""

    max_steps: int | None = None
    """Maximum number of steps to record. If None, replay entire trajectory."""

    scene: str | None = None
    """Optional scene override."""


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


def get_actions(all_actions, action_idx: int, num_envs: int):
    """Get actions for current step across all environments."""
    envs_actions = all_actions[:num_envs]
    return [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]


def get_runout(all_actions, action_idx: int):
    """Check if all environments have run out of actions."""
    return all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])


def main():
    args = tyro.cli(Args)

    log.info(f"Task: {args.task}")
    log.info(f"Recording to: {args.output}")
    log.info("NOTE: This script replays pre-recorded trajectories (no IK solver needed)")

    # ========================================================================
    # Setup Task and Environment
    # ========================================================================
    task_cls = get_task_class(args.task)

    camera = PinholeCameraCfg(
        name="camera",
        pos=(1.5, -1.5, 1.5),
        look_at=(0.0, 0.0, 0.0),
        width=640,
        height=480,
    )

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    env = task_cls(scenario, device=device)

    # ========================================================================
    # Load Trajectory
    # ========================================================================
    traj_filepath = env.traj_filepath
    if not os.path.exists(traj_filepath):
        log.error(f"Trajectory file not found: {traj_filepath}")
        log.info("Attempting to download trajectory file...")
        check_and_download_recursive([traj_filepath], n_processes=4)

    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    log.info(f"Loading trajectory from: {traj_filepath}")

    init_states, all_actions, _ = get_traj(traj_filepath, scenario.robots[0], env.handler)

    # Get total number of steps
    max_traj_steps = max(len(actions) for actions in all_actions)
    num_steps = min(args.max_steps, max_traj_steps) if args.max_steps else max_traj_steps
    log.info(f"Trajectory has {max_traj_steps} steps, will record {num_steps} steps")

    # ========================================================================
    # Setup Rerun Recording
    # ========================================================================
    from metasim.utils.rerun.rerun_util import RerunVisualizer

    download_urdf_files(scenario)

    visualizer = RerunVisualizer(
        app_name=f"Task Replay: {args.task}",
        spawn=args.spawn_viewer,
        save_path=args.output,
    )
    visualizer.add_frame("world/origin")

    # Reset environment and get initial observation
    obs, _ = env.reset()

    # Initial visualization
    default_object_states = extract_states_from_obs(obs, env.handler, "objects")
    default_robot_states = extract_states_from_obs(obs, env.handler, "robots")
    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    # Log initial state
    visualizer.set_time(0)
    for name, state in default_object_states.items():
        visualizer.update_item_pose(name, state)
    for name, state in default_robot_states.items():
        visualizer.update_item_pose(name, state)

    # ========================================================================
    # Replay Trajectory and Record
    # ========================================================================
    log.info("Starting trajectory replay and recording...")

    for step in range(num_steps):
        visualizer.set_time(step + 1)  # +1 because we logged initial state at t=0

        # Get actions for this step
        actions = get_actions(all_actions, step, args.num_envs)

        # Step the environment
        obs, reward, success, time_out, extras = env.step(actions)

        # Update visualization
        object_states = extract_states_from_obs(obs, env.handler, "objects")
        robot_states = extract_states_from_obs(obs, env.handler, "robots")

        for name, state in object_states.items():
            visualizer.update_item_pose(name, state)
        for name, state in robot_states.items():
            visualizer.update_item_pose(name, state)

        # Log progress
        if step % 20 == 0:
            log.info(f"Recording step {step}/{num_steps}")

        # Check for success
        if success.any():
            log.info(f"Task succeeded at step {step}!")

        # Check if all trajectories are exhausted
        if get_runout(all_actions, step + 1):
            log.info(f"Trajectory exhausted at step {step + 1}")
            break

    visualizer.close()
    env.close()

    log.info("")
    log.info("=" * 60)
    log.info("Recording Complete!")
    log.info("=" * 60)
    log.info(f"Task: {args.task}")
    log.info(f"Saved to: {args.output}")
    log.info(f"Total steps: {step + 1}")
    log.info("")
    log.info("To replay the recording:")
    log.info(f"  rerun {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
