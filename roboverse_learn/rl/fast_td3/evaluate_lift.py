"""Evaluation script for collecting successful lift trajectories.

Records state when first entering lift phase, saves traj and state after successful lift (maintained for 10 frames).
Loops until collecting target number of successful trajectories (default: 100).
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
from typing import Any

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
from torch.amp import autocast
from datetime import datetime

from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import save_traj_file


def extract_state_dict(env, scenario, env_idx=0):
    """Extract state dictionary from handler states.

    Args:
        env: Environment with handler
        scenario: Scenario configuration to get joint names
        env_idx: Environment index to extract state from

    Returns:
        Dictionary containing positions, rotations, and joint positions for all objects and robots
    """
    state_dict = {}

    # Get states from handler (returns TensorState object)
    if not hasattr(env, 'handler') or env.handler is None:
        log.warning("Handler not available, returning empty state")
        return state_dict

    handler_states = env.handler.get_states(mode="tensor")
    if handler_states is None:
        log.warning("Handler.get_states() returned None")
        return state_dict

    # Create lookup dicts for configurations
    obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
    robot_cfg_dict = {robot.name: robot for robot in scenario.robots}

    # Extract object states
    if hasattr(handler_states, 'objects'):
        for obj_name, obj_state in handler_states.objects.items():
            pos = obj_state.root_state[env_idx, :3].cpu().numpy()  # [x, y, z]
            quat = obj_state.root_state[env_idx, 3:7].cpu().numpy()  # [w, x, y, z]

            state_entry = {
                "pos": pos,
                "rot": quat,
            }

            # Add joint positions if the object has joints
            if obj_state.joint_pos is not None and obj_name in obj_cfg_dict:
                obj_cfg = obj_cfg_dict[obj_name]
                if hasattr(obj_cfg, "actuators") and obj_cfg.actuators is not None:
                    # Joint names are sorted alphabetically (standard in handlers)
                    joint_names = sorted(obj_cfg.actuators.keys())
                    joint_positions = obj_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

            state_dict[obj_name] = state_entry

    # Extract robot states
    if hasattr(handler_states, 'robots'):
        for robot_name, robot_state in handler_states.robots.items():
            pos = robot_state.root_state[env_idx, :3].cpu().numpy()  # [x, y, z]
            quat = robot_state.root_state[env_idx, 3:7].cpu().numpy()  # [w, x, y, z]

            state_entry = {
                "pos": pos,
                "rot": quat,
            }

            # Add joint positions for robot
            if robot_name in robot_cfg_dict:
                robot_cfg = robot_cfg_dict[robot_name]
                if robot_cfg.actuators is not None:
                    # Joint names are sorted alphabetically (standard in handlers)
                    joint_names = sorted(robot_cfg.actuators.keys())
                    joint_positions = robot_state.joint_pos[env_idx].cpu().numpy()
                    state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

            state_dict[robot_name] = state_entry

    return state_dict


def tensor_to_list(data):
    """Recursively convert tensors to lists/numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def evaluate_lift_collection(
    env,
    actor,
    obs_normalizer,
    target_count: int,
    device: torch.device,
    scenario=None,
    task_name: str = "eval",
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    traj_dir: str = "eval_trajs",
    state_dir: str = "eval_states",
    lift_stable_frames: int = 10,
) -> dict:
    """Evaluate and collect successful lift trajectories."""
    actor.eval()
    obs_normalizer.eval()

    num_eval_envs = env.num_envs
    collected_trajs = []
    collected_states = []

    lift_start_state = {}
    lift_frame_count = {}
    in_lift_phase = {}
    recording_traj = {}
    for i in range(num_eval_envs):
        lift_start_state[i] = None
        lift_frame_count[i] = 0
        in_lift_phase[i] = False
        recording_traj[i] = False

    current_episode_actions = {}
    current_episode_states = {}
    current_episode_init_state = {}

    for i in range(num_eval_envs):
        current_episode_actions[i] = []
        current_episode_states[i] = []
        current_episode_init_state[i] = None

    episodes_completed = 0
    # Track how many episodes produced at least one successful lift
    successful_episodes_count = 0
    # Per-env flag indicating whether the current episode already had a success
    success_in_episode = {i: False for i in range(num_eval_envs)}

    current_returns = torch.zeros(num_eval_envs, device=device)
    current_lengths = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

    obs, info = env.reset()

    for i in range(num_eval_envs):
        current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)

    max_steps_per_episode = env.max_episode_steps
    max_total_steps = max_steps_per_episode * 10000

    log.info(f"Starting lift trajectory collection, target: {target_count}")

    for step in range(max_total_steps):
        if len(collected_trajs) >= target_count:
            log.info(f"Collected {len(collected_trajs)} successful trajectories, reached target {target_count}")
            break

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())
        dones = terminated | time_out

        handler_states = None
        if hasattr(env, 'handler') and env.handler is not None:
            handler_states = env.handler.get_states(mode="tensor")

        for i in range(num_eval_envs):
            if done_masks[i]:
                continue

            grasp_success = infos.get("grasp_success", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]
            lift_active = infos.get("lift_active", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]

            robot_name = scenario.robots[0].name
            joint_names = sorted(scenario.robots[0].actuators.keys())

            if handler_states is not None and hasattr(handler_states, 'robots') and robot_name in handler_states.robots:
                robot_state = handler_states.robots[robot_name]
                joint_positions = robot_state.joint_pos[i].cpu().numpy()
            else:
                robot_state = obs.robots[robot_name]
                joint_positions = robot_state.joint_pos[i].cpu().numpy()

            action_record = {
                "dof_pos_target": {name: float(pos) for name, pos in zip(joint_names, joint_positions)},
            }

            current_episode_actions[i].append(action_record)
            current_state = extract_state_dict(env, scenario, env_idx=i)
            current_episode_states[i].append(current_state)

            if grasp_success and lift_active and not in_lift_phase[i]:
                in_lift_phase[i] = True
                lift_start_state[i] = extract_state_dict(env, scenario, env_idx=i)
                lift_frame_count[i] = 1
                recording_traj[i] = True

                log.info(f"[Env {i}] Entered lift phase (grasp success and lift active)")

            elif in_lift_phase[i]:
                if lift_active and grasp_success:
                    lift_frame_count[i] += 1

                    if lift_frame_count[i] >= lift_stable_frames:
                        traj_data = {
                            "init_state": current_episode_init_state[i],
                            "actions": current_episode_actions[i],
                            "states": current_episode_states[i],
                        }

                        traj_data_serializable = tensor_to_list(traj_data)
                        state_data_serializable = tensor_to_list(lift_start_state[i])

                        collected_trajs.append(traj_data_serializable)
                        collected_states.append(state_data_serializable)

                        # Mark episode as successful (count at most once per episode)
                        if not success_in_episode[i]:
                            success_in_episode[i] = True
                            successful_episodes_count += 1

                        log.info(
                            f"[Env {i}] Collected trajectory {len(collected_trajs)} "
                            f"(lift maintained {lift_frame_count[i]} frames, total steps: {len(current_episode_actions[i])})"
                        )

                        lift_start_state[i] = None
                        lift_frame_count[i] = 0
                        in_lift_phase[i] = False
                        recording_traj[i] = False

                        if len(collected_trajs) >= target_count:
                            done_masks[i] = True
                else:
                    lift_frame_count[i] = 0
                    if not grasp_success:
                        in_lift_phase[i] = False
                        recording_traj[i] = False

        active_mask = ~done_masks
        current_returns = torch.where(active_mask, current_returns + rewards, current_returns)
        current_lengths = torch.where(active_mask, current_lengths + 1, current_lengths)

        newly_done = dones & ~done_masks
        if newly_done.any():
            for i in range(num_eval_envs):
                if newly_done[i]:
                    episodes_completed += 1

                    lift_start_state[i] = None
                    lift_frame_count[i] = 0
                    in_lift_phase[i] = False
                    recording_traj[i] = False
                    current_episode_actions[i] = []
                    current_episode_states[i] = []
                    current_episode_init_state[i] = None
                    current_returns[i] = 0
                    current_lengths[i] = 0
                    # reset per-episode success flag for next episode
                    success_in_episode[i] = False

            done_masks = torch.logical_or(done_masks, dones)

        if done_masks.all():
            done_masks.fill_(False)
            obs, info = env.reset()

            for i in range(num_eval_envs):
                lift_start_state[i] = None
                lift_frame_count[i] = 0
                in_lift_phase[i] = False
                recording_traj[i] = False
                current_episode_actions[i] = []
                current_episode_states[i] = []
                current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)
                # reset per-episode success flag after full reset
                success_in_episode[i] = False
        else:
            obs = next_obs

    # Treat each active env as an attempted episode, and count it as successful if it already
    active_envs = (~done_masks).sum().item() if 'done_masks' in locals() else 0
    successes_in_active = sum(
        1 for i in range(num_eval_envs)
        if 'done_masks' in locals() and not done_masks[i] and success_in_episode.get(i, False)
    )
    attempted_episodes = episodes_completed + active_envs
    total_successful_episodes = successful_episodes_count + successes_in_active

    if len(collected_trajs) > 0:
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        robot_name = scenario.robots[0].name
        trajs = {robot_name: collected_trajs}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_filename = f"{task_name}_{robot_name}_lift_{len(collected_trajs)}trajs_{timestamp}_v2.pkl"
        state_filename = f"{task_name}_{robot_name}_lift_states_{len(collected_states)}states_{timestamp}.pkl"

        traj_filepath = os.path.join(traj_dir, traj_filename)
        state_filepath = os.path.join(state_dir, state_filename)

        save_traj_file(trajs, traj_filepath)
        log.info(f"Trajectories saved to: {traj_filepath}")
        log.info(f"  - Trajectory count: {len(collected_trajs)}")
        log.info(f"  - Total steps: {sum(len(traj['actions']) for traj in collected_trajs)}")

        with open(state_filepath, "wb") as f:
            pickle.dump(collected_states, f)
        log.info(f"States saved to: {state_filepath}")
        log.info(f"  - State count: {len(collected_states)}")
    else:
        log.warning("No successful trajectories collected")
    # Success rate: fraction of attempted episodes (completed + in-progress when we stopped)
    denom = max(attempted_episodes, 1)
    success_rate = min(total_successful_episodes, denom) / denom
    stats = {
        "collected_count": len(collected_trajs),
        "target_count": target_count,
        "episodes_completed": attempted_episodes,
        "success_rate": success_rate,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='FastTD3 lift trajectory collection evaluation')
    parser.add_argument('--checkpoint', type=str, default='models/pick_place.approach_grasp_simple_1210000.pt',
                       help='Checkpoint file path')
    parser.add_argument('--target_count', type=int, default=100,
                       help='Target number of successful trajectories to collect (default: 100)')

    parser.add_argument('--device_rank', type=int, default=0,
                       help='GPU device rank')
    parser.add_argument('--num_envs', type=int, default=None,
                       help='Number of parallel environments (default: from checkpoint config)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode')

    parser.add_argument('--traj_dir', type=str, default='eval_trajs',
                       help='Trajectory save directory')
    parser.add_argument('--state_dir', type=str, default='eval_states',
                       help='State save directory')
    parser.add_argument('--lift_stable_frames', type=int, default=10,
                       help='Number of frames lift must be maintained (default: 10)')

    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    config = checkpoint.get("config", {})

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
        torch.cuda.set_device(args.device_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device_rank}")

    log.info(f"Using device: {device}")
    log.info(f"Checkpoint global step: {checkpoint.get('global_step', 'unknown')}")

    task_name = config.get("task")
    if not task_name:
        raise ValueError("Task name not found in checkpoint config")

    task_cls = get_task_class(task_name)
    num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 1)

    scenario = task_cls.scenario.update(
        robots=config.get("robots", ["franka"]),
        simulator=config.get("sim", "mujoco"),
        num_envs=num_envs,
        headless=args.headless,
        cameras=[],
    )

    env = task_cls(scenario, device=device)

    n_obs = env.num_obs
    n_act = env.num_actions

    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        device=device,
        init_scale=config.get("init_scale", 0.1),
        hidden_dim=config.get("actor_hidden_dim", 256),
    )

    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    amp_enabled = config.get("amp", False) and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    log.info(f"Starting lift trajectory collection...")
    log.info(f"  - Target count: {args.target_count}")
    log.info(f"  - Lift stable frames: {args.lift_stable_frames}")
    log.info(f"  - Trajectory dir: {args.traj_dir}")
    log.info(f"  - State dir: {args.state_dir}")

    stats = evaluate_lift_collection(
        env=env,
        actor=actor,
        obs_normalizer=obs_normalizer,
        target_count=args.target_count,
        device=device,
        scenario=scenario,
        task_name=task_name,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        traj_dir=args.traj_dir,
        state_dir=args.state_dir,
        lift_stable_frames=args.lift_stable_frames,
    )

    log.info("=" * 50)
    log.info("Evaluation results:")
    log.info(f"  Collected trajectories: {stats['collected_count']}")
    log.info(f"  Target count: {stats['target_count']}")
    log.info(f"  Episodes completed: {stats['episodes_completed']}")
    log.info(f"  Success rate: {stats['success_rate']:.2%}")
    log.info("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
