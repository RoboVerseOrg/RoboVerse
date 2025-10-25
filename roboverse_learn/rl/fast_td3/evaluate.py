from __future__ import annotations

import os
import sys
import argparse
from typing import Any
import yaml

import os
import sys

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
    """Extract state dictionary from handler states (similar to teleop_keyboard).

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


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def evaluate(
    env,
    actor,
    obs_normalizer,
    num_episodes: int,
    device: torch.device,
    scenario = None,
    task_name: str = "eval",
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    render: bool = True,
    video_path: str = None,
    render_each_episode: bool = True,
    save_traj: bool = True,
    save_states: bool = True,
    save_every_n_steps: int = 1,
    traj_dir: str = "eval_trajs",
) -> dict:
    """
    Evaluate the policy for a specified number of episodes.

    Args:
        env: The environment to evaluate on
        actor: The policy network
        obs_normalizer: Observation normalizer
        num_episodes: Number of episodes to run
        device: Device to run evaluation on
        scenario: Scenario configuration (required for trajectory saving)
        task_name: Task name for trajectory filename
        amp_enabled: Whether to use automatic mixed precision
        amp_device_type: Device type for AMP
        amp_dtype: Data type for AMP
        render: Whether to render and save video
        video_path: Path to save rendered video (base path for multiple videos)
        render_each_episode: If True, save a separate video for each episode
        save_traj: Whether to save trajectories
        save_states: Whether to save full states (not just actions)
        save_every_n_steps: Save every N steps (1=save all, 2=save every other step)
        traj_dir: Directory to save trajectories

    Returns:
        Dictionary with evaluation statistics
    """
    actor.eval()
    obs_normalizer.eval()

    num_eval_envs = env.num_envs
    episode_returns = []
    episode_lengths = []
    episode_successes = []

    # For single video mode
    frames = [] if (render and not render_each_episode) else None

    # For per-episode video mode
    episode_frames = {} if (render and render_each_episode) else None
    if render_each_episode:
        for i in range(num_eval_envs):
            episode_frames[i] = []

    # For trajectory saving
    all_episodes = {} if save_traj else None  # Dict: env_id -> list of episodes
    current_episode_actions = {}  # Dict: env_id -> current episode actions
    current_episode_states = {}  # Dict: env_id -> current episode states
    current_episode_init_state = {}  # Dict: env_id -> init state
    episode_step_count = {}  # Dict: env_id -> step count in current episode

    if save_traj:
        if scenario is None:
            raise ValueError("scenario must be provided when save_traj=True")
        for i in range(num_eval_envs):
            all_episodes[i] = []
            current_episode_actions[i] = []
            current_episode_states[i] = [] if save_states else None
            episode_step_count[i] = 0

    episodes_completed = 0
    episodes_per_env = torch.zeros(num_eval_envs, dtype=torch.long, device=device)  # Track episodes per env
    current_returns = torch.zeros(num_eval_envs, device=device)
    current_lengths = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
    finished_envs = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)  # Envs that reached num_episodes

    obs, info = env.reset()

    # Record initial states for trajectory saving
    if save_traj:
        for i in range(num_eval_envs):
            current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)

    if render and not render_each_episode:
        frames.append(env.render())
    elif render_each_episode:
        current_frame = env.render()
        for i in range(num_eval_envs):
            if not done_masks[i]:
                episode_frames[i].append(current_frame)

    max_steps = env.max_episode_steps * num_episodes

    for step in range(max_steps):
        # Only process envs that haven't finished all their episodes
        if finished_envs.all():
            break

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())
        dones = terminated | time_out

        # Record trajectory data (with downsampling)
        if save_traj:
            # Get states from handler for trajectory recording
            handler_states = None
            if hasattr(env, 'handler') and env.handler is not None:
                handler_states = env.handler.get_states(mode="tensor")

            for i in range(num_eval_envs):
                # Only record for envs that haven't finished all episodes
                if not finished_envs[i] and not done_masks[i] and (episode_step_count[i] % save_every_n_steps == 0):
                    # Get robot joint positions as actions from handler states
                    robot_name = scenario.robots[0].name
                    joint_names = sorted(scenario.robots[0].actuators.keys())

                    if handler_states is not None and hasattr(handler_states, 'robots') and robot_name in handler_states.robots:
                        # Use handler states (preferred)
                        robot_state = handler_states.robots[robot_name]
                        joint_positions = robot_state.joint_pos[i].cpu().numpy()
                    else:
                        # Fallback to obs if handler not available
                        robot_state = obs.robots[robot_name]
                        joint_positions = robot_state.joint_pos[i].cpu().numpy()

                    action_record = {
                        "dof_pos_target": {name: float(pos) for name, pos in zip(joint_names, joint_positions)},
                    }
                    current_episode_actions[i].append(action_record)

                    # Record state if requested
                    if save_states and current_episode_states[i] is not None:
                        # Extract state for this specific env using handler
                        current_state = extract_state_dict(env, scenario, env_idx=i)
                        current_episode_states[i].append(current_state)

                # Increment step count for active environments
                if not finished_envs[i] and not done_masks[i]:
                    episode_step_count[i] += 1

        # Render current frame
        if render:
            current_frame = env.render()
            if not render_each_episode:
                frames.append(current_frame)
            else:
                for i in range(num_eval_envs):
                    # Only render for envs that haven't finished all episodes
                    if not finished_envs[i] and not done_masks[i]:
                        episode_frames[i].append(current_frame)

        # Update episode statistics (only for envs still running)
        active_mask = ~done_masks & ~finished_envs
        current_returns = torch.where(active_mask, current_returns + rewards, current_returns)
        current_lengths = torch.where(active_mask, current_lengths + 1, current_lengths)

        # Check for newly completed episodes (only for envs that haven't finished all episodes)
        newly_done = dones & ~done_masks & ~finished_envs
        if newly_done.any():
            import imageio.v2 as iio

            for i in range(num_eval_envs):
                if newly_done[i]:
                    episode_returns.append(current_returns[i].item())
                    episode_lengths.append(current_lengths[i].item())

                    # Check for success if available in info
                    if "success" in infos:
                        episode_successes.append(infos["success"][i].item())

                    # Save individual episode video if enabled
                    if render_each_episode and episode_frames[i] and video_path:
                        # Use env_id and episode number for filename
                        base_dir = os.path.dirname(video_path)
                        base_name = os.path.splitext(os.path.basename(video_path))[0]
                        ext = os.path.splitext(video_path)[1] or '.mp4'
                        ep_video_path = os.path.join(base_dir, f"{base_name}_env{i:02d}_ep{episodes_per_env[i].item():02d}{ext}")

                        os.makedirs(base_dir, exist_ok=True)
                        iio.mimsave(ep_video_path, episode_frames[i], fps=30)
                        log.info(f"Env {i} Episode {episodes_per_env[i].item()}: Saved video to {ep_video_path} (return: {current_returns[i].item():.2f})")

                        # Clear frames for this env
                        episode_frames[i] = []

                    # Save trajectory for this episode if enabled
                    if save_traj and len(current_episode_actions[i]) > 0:
                        episode_data = {
                            "init_state": current_episode_init_state[i],
                            "actions": current_episode_actions[i],
                            "states": current_episode_states[i] if save_states else None,
                        }
                        all_episodes[i].append(episode_data)
                        log.info(f"Env {i} Episode {episodes_per_env[i].item()}: Saved trajectory ({len(current_episode_actions[i])} steps, return: {current_returns[i].item():.2f})")

                        # Reset trajectory tracking for this env
                        current_episode_actions[i] = []
                        if save_states:
                            current_episode_states[i] = []
                        episode_step_count[i] = 0

                    episodes_completed += 1
                    episodes_per_env[i] += 1

                    # Check if this env has finished all required episodes
                    if episodes_per_env[i] >= num_episodes:
                        finished_envs[i] = True
                        log.info(f"Env {i}: Completed all {num_episodes} episodes")

                    # Reset stats for this env
                    current_returns[i] = 0
                    current_lengths[i] = 0

            done_masks = torch.logical_or(done_masks, dones)

        # Stop if all envs have finished their episodes
        if finished_envs.all():
            break

        # Reset done_masks for envs that are still running (haven't finished all episodes)
        if (done_masks & ~finished_envs).any():
            done_masks.fill_(False)
            obs, info = env.reset()

            # Record initial states for new episodes if saving trajectories
            if save_traj:
                for i in range(num_eval_envs):
                    current_episode_init_state[i] = extract_state_dict(env, scenario, env_idx=i)

            # Add first frame for new episodes if rendering per episode
            if render_each_episode:
                current_frame = env.render()
                for i in range(num_eval_envs):
                    episode_frames[i].append(current_frame)
        else:
            obs = next_obs

    # Save single video if rendering all episodes together
    if render and not render_each_episode and frames and video_path:
        import imageio.v2 as iio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        iio.mimsave(video_path, frames, fps=30)
        log.info(f"Saved evaluation video to {video_path}")

    # Save all trajectories to file if enabled
    if save_traj and all_episodes:
        # Flatten all episodes from all envs into a single list
        all_episodes_flat = []
        for env_id in range(num_eval_envs):
            all_episodes_flat.extend(all_episodes[env_id])

        if len(all_episodes_flat) > 0:
            # Organize in v2 format
            robot_name = scenario.robots[0].name
            trajs = {robot_name: all_episodes_flat}

            # Create output directory
            os.makedirs(traj_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_name}_{robot_name}_eval_{timestamp}_v2.pkl"
            filepath = os.path.join(traj_dir, filename)

            # Save trajectory
            save_traj_file(trajs, filepath)
            log.info(f"All trajectories saved to {filepath}")
            log.info(f"  - Total episodes: {len(all_episodes_flat)}")
            log.info(f"  - Total steps: {sum(len(ep['actions']) for ep in all_episodes_flat)}")
        else:
            log.warning("No trajectories to save")

    # Compute statistics
    stats = {
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(episode_returns),
    }

    if episode_successes:
        stats["success_rate"] = np.mean(episode_successes)

    return stats


def main():
    parser = argparse.ArgumentParser(description='FastTD3 Evaluation')
    parser.add_argument('--checkpoint', type=str, default='roboverse_data/models/walk_1400.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='Number of episodes per environment (default: 1, each env saves one episode)')

    # Rendering arguments
    parser.add_argument('--render', type=int, default=1,
                       help='Render mode: 0=no render, 1=render each episode separately (default), 2=single combined video')
    parser.add_argument('--video_path', type=str, default='output/eval_rollout.mp4',
                       help='Path to save video (base name for multiple videos)')

    parser.add_argument('--device_rank', type=int, default=0,
                       help='GPU device rank')
    parser.add_argument('--num_envs', type=int, default=None,
                       help='Number of parallel environments (default: from checkpoint config)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode')

    # Trajectory saving arguments (default: enabled)
    parser.add_argument('--save_traj', type=int, default=1,
                       help='Save trajectories: 0=no, 1=yes (default)')
    parser.add_argument('--save_states', type=int, default=1,
                       help='Save full states: 0=no (actions only), 1=yes (default)')
    parser.add_argument('--save_every_n_steps', type=int, default=1,
                       help='Save every N steps (1=save all, 2=save every other step, etc.)')
    parser.add_argument('--traj_dir', type=str, default='eval_trajs',
                       help='Directory to save trajectories')

    args = parser.parse_args()

    # Convert render mode to booleans
    render = args.render > 0
    render_each_episode = args.render == 1
    save_traj = bool(args.save_traj)
    save_states = bool(args.save_states)

    # Load checkpoint
    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    # Get configuration from checkpoint
    config = checkpoint.get("config", {})

    # Override device based on availability
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
        torch.cuda.set_device(args.device_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device_rank}")

    log.info(f"Using device: {device}")
    log.info(f"Checkpoint global step: {checkpoint.get('global_step', 'unknown')}")

    # Get task configuration
    task_name = config.get("task")
    if not task_name:
        raise ValueError("Task name not found in checkpoint config")

    # Setup environment
    task_cls = get_task_class(task_name)
    num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 1)

    # Configure cameras for rendering if needed
    cameras = []
    if args.render:
        cameras = [
            PinholeCameraCfg(
                width=config.get("video_width", 1024),
                height=config.get("video_height", 1024),
                pos=(4.0, -4.0, 4.0),
                look_at=(0.0, 0.0, 0.0),
            )
        ]

    scenario = task_cls.scenario.update(
        robots=config.get("robots", ["franka"]),
        simulator=config.get("sim", "mujoco"),
        num_envs=num_envs,
        headless=args.headless or not args.render,
        cameras=cameras,
    )

    env = task_cls(scenario, device=device)

    # Get dimensions
    n_obs = env.num_obs
    n_act = env.num_actions

    # Create actor and normalizer
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        device=device,
        init_scale=config.get("init_scale", 0.1),
        hidden_dim=config.get("actor_hidden_dim", 256),
    )

    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    # Load weights
    actor.load_state_dict(checkpoint["actor_state_dict"])
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    # Setup AMP
    amp_enabled = config.get("amp", False) and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    # Run evaluation
    log.info(f"Evaluating for {args.num_episodes} episodes...")
    log.info(f"  - Render: {render}")
    log.info(f"  - Render each episode: {render_each_episode}")
    log.info(f"  - Save trajectories: {save_traj}")
    log.info(f"  - Save states: {save_states}")

    if render_each_episode and render:
        log.info(f"Saving separate video for each episode to {args.video_path}_epXXX.mp4")
    elif render:
        log.info(f"Saving single combined video to {args.video_path}")

    if save_traj:
        log.info(f"Saving trajectories to {args.traj_dir}/ (every {args.save_every_n_steps} steps)")

    stats = evaluate(
        env=env,
        actor=actor,
        obs_normalizer=obs_normalizer,
        num_episodes=args.num_episodes,
        device=device,
        scenario=scenario,
        task_name=task_name,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        render=render,
        video_path=args.video_path if render else None,
        render_each_episode=render_each_episode,
        save_traj=save_traj,
        save_states=save_states,
        save_every_n_steps=args.save_every_n_steps,
        traj_dir=args.traj_dir,
    )

    # Print results
    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"  Episodes: {stats['num_episodes']}")
    log.info(f"  Mean Return: {stats['mean_return']:.4f} ± {stats['std_return']:.4f}")
    log.info(f"  Mean Length: {stats['mean_length']:.4f} ± {stats['std_length']:.4f}")
    if "success_rate" in stats:
        log.info(f"  Success Rate: {stats['success_rate']:.2%}")
    log.info("=" * 50)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
