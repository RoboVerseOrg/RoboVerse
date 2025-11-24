"""Replay script for lift states.

Continuously replays saved lift states for visualization.
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
import time

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure repository root is on sys.path for local package imports
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
import numpy as np
from loguru import logger as log
import imageio.v2 as iio

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from roboverse_pack.tasks.pick_place.track import convert_state_dict_to_initial_state


def load_states_from_pkl(pkl_path: str):
    """Load state list from pkl file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"State file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        states_list = pickle.load(f)

    log.info(f"Loaded {len(states_list)} states from {pkl_path}")
    return states_list


def convert_state_to_dict_format(state_dict: dict, device: torch.device, robot_name: str = "franka") -> dict:
    """Convert flat state dict to nested format for handler.set_states."""
    nested_state = {
        "objects": {},
        "robots": {},
    }

    if "objects" in state_dict and "robots" in state_dict:
        return state_dict

    # Convert flat format to nested
    for name, entity_state in state_dict.items():
        if name in ["objects", "robots"]:
            continue

        pos = entity_state.get("pos")
        rot = entity_state.get("rot")

        if isinstance(pos, (list, tuple, np.ndarray)):
            pos = torch.tensor(pos, device=device, dtype=torch.float32)
        elif isinstance(pos, torch.Tensor):
            pos = pos.to(device).float()

        if isinstance(rot, (list, tuple, np.ndarray)):
            rot = torch.tensor(rot, device=device, dtype=torch.float32)
        elif isinstance(rot, torch.Tensor):
            rot = rot.to(device).float()

        entity_entry = {
            "pos": pos,
            "rot": rot,
        }

        if "dof_pos" in entity_state:
            entity_entry["dof_pos"] = entity_state["dof_pos"]

        if name == robot_name:
            nested_state["robots"][name] = entity_entry
        else:
            nested_state["objects"][name] = entity_entry

    return nested_state


def main():
    parser = argparse.ArgumentParser(description='Replay Lift States')
    parser.add_argument('--state_file', type=str,
                       default='eval_states/pick_place.approach_grasp_simple_franka_lift_states_101states_20251122_180651.pkl',
                       help='State file path (pkl format)')
    parser.add_argument('--task', type=str, default='pick_place.track',
                       help='Task name')
    parser.add_argument('--sim', type=str, default='isaacsim',
                       help='Simulator type')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of environments')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Enable rendering')
    parser.add_argument('--save_video', action='store_true',
                       help='Save video')
    parser.add_argument('--video_path', type=str, default='replay_output/replay.mp4',
                       help='Video output path')
    parser.add_argument('--loop', action='store_true', default=True,
                       help='Loop replay')
    parser.add_argument('--max_loops', type=int, default=None,
                       help='Max loop count (None for infinite)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    parser.add_argument('--delay', type=float, default=0.0,
                       help='Delay between frames (seconds)')
    parser.add_argument('--disable_dr', action='store_true', default=True,
                       help='Disable domain randomization (default: True)')

    args = parser.parse_args()

    log.info(f"Loading states from {args.state_file}")
    states_list = load_states_from_pkl(args.state_file)

    if len(states_list) == 0:
        log.error("No states found in file")
        return

    if args.disable_dr:
        log.info("Disabling domain randomization")
        try:
            from roboverse_pack.tasks.pick_place import base as pick_base

            pick_base.DEFAULT_CONFIG["randomization"]["box_pos_range"] = 0.0
            pick_base.DEFAULT_CONFIG["randomization"]["robot_pos_noise"] = 0.0
            pick_base.DEFAULT_CONFIG["randomization"]["joint_noise_range"] = 0.0
        except Exception as e:
            log.warning(f"Failed to disable DR: {e}")
    task_cls = get_task_class(args.task)

    cameras = []
    if args.render:
        cameras = [
            PinholeCameraCfg(
                width=1024,
                height=1024,
                pos=(4.0, -4.0, 4.0),
                look_at=(0.0, 0.0, 0.0),
            )
        ]

    scenario = task_cls.scenario.update(
        robots=["franka"],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless or not args.render,
        cameras=cameras,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    frames = [] if args.save_video else None
    if args.save_video:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)

    log.info(f"Replaying {len(states_list)} states...")
    log.info(f"  - Loop mode: {args.loop}")
    log.info(f"  - Max loops: {args.max_loops if args.max_loops else 'infinite'}")
    log.info(f"  - Frame delay: {args.delay}s")

    loop_count = 0
    robot_name = scenario.robots[0].name

    try:
        while True:
            if args.max_loops is not None and loop_count >= args.max_loops:
                log.info(f"Reached max loops {args.max_loops}, stopping")
                break

            loop_count += 1
            log.info(f"\n{'='*60}")
            log.info(f"Loop {loop_count}")
            log.info(f"{'='*60}")

            for state_idx, state_dict in enumerate(states_list):
                nested_state = convert_state_to_dict_format(state_dict, device, robot_name=robot_name)
                states_to_set = [nested_state] * args.num_envs
                try:
                    env.handler.set_states(states=states_to_set, env_ids=list(range(args.num_envs)))
                    env.handler.refresh_render()
                except Exception as e:
                    log.error(f"Failed to set state {state_idx}: {e}")
                    continue

                handler_states = env.handler.get_states(mode="tensor")
                if handler_states is not None:
                    box_pos = handler_states.objects["object"].root_state[0, :3].cpu().numpy()
                    try:
                        gripper_pos, _ = env._get_ee_state(handler_states)
                        gripper_pos_np = gripper_pos[0].cpu().numpy()
                        gripper_box_dist = np.linalg.norm(box_pos - gripper_pos_np)

                        robot_state_tensor = handler_states.robots[robot_name]
                        joint_positions = robot_state_tensor.joint_pos[0].cpu().numpy()
                        joint_names = sorted(scenario.robots[0].actuators.keys())
                        gripper_joint_names = [name for name in joint_names if "finger" in name.lower()]

                        if len(gripper_joint_names) >= 2:
                            gripper_angles = [
                                joint_positions[joint_names.index(gripper_joint_names[0])],
                                joint_positions[joint_names.index(gripper_joint_names[1])]
                            ]
                        else:
                            gripper_angles = joint_positions[:2].tolist()

                        if state_idx == 0 or state_idx % 10 == 0 or state_idx == len(states_list) - 1:
                            log.info(
                                f"[State {state_idx:3d}/{len(states_list)-1}] "
                                f"Distance: {gripper_box_dist:.4f}m | "
                                f"Gripper: [{gripper_angles[0]:.4f}, {gripper_angles[1]:.4f}]"
                            )
                    except Exception as e:
                        log.debug(f"Error computing distance (state {state_idx}): {e}")

                if args.render:
                    frame = env.render()
                    if args.save_video and frame is not None:
                        frames.append(frame)

                if args.delay > 0:
                    time.sleep(args.delay)

            if not args.loop:
                break

            log.info(f"Completed loop {loop_count}")

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    if args.save_video and frames:
        log.info(f"Saving video to {args.video_path} ({len(frames)} frames)")
        iio.mimsave(args.video_path, frames, fps=args.fps)
        log.info("Video saved")

    env.close()
    log.info("Replay completed")


if __name__ == "__main__":
    main()
