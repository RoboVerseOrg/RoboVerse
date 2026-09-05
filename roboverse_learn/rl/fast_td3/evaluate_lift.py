# Copyright (c) 2025 Younggyo Seo
# SPDX-License-Identifier: MIT
#
# Adapted from FastTD3 (https://github.com/younggyoseo/FastTD3).
# Changes: RoboVerse-specific evaluation script for collecting successful lift trajectories; reuses the
#   FastTD3 Actor/EmpiricalNormalization inference path from fttd3_module and the upstream environment-setup
#   preamble, and drives MetaSim tasks via get_task_class/handler state APIs.
# Full license: roboverse_learn/rl/fast_td3/LICENSE
"""Evaluation script for collecting successful lift trajectories.

Records state when first entering lift phase, saves traj and state after successful lift (maintained for 10 frames).
Loops until collecting target number of successful trajectories (default: 100).
"""

from __future__ import annotations

import argparse
import os
import pickle
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

from datetime import datetime

import torch
from loguru import logger as log
from torch.amp import autocast

from metasim.task.registry import get_task_class
from metasim.utils.demo_util import save_traj_file
from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization
from roboverse_learn.rl.fast_td3.trajectory_record import (
    commanded_action,
    initial_states,
    recorded_step,
    restart_done_envs,
)


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

    init_states = initial_states(env.handler)
    for i in range(num_eval_envs):
        current_episode_init_state[i] = init_states[i]

    max_steps_per_episode = env.max_episode_steps
    max_total_steps = max_steps_per_episode * 10000

    log.info(f"Starting lift trajectory collection, target: {target_count}")

    robot_name = env.robot.name  # the recorded robot (single-robot tasks)

    env.record_terminal_states = True  # keep the pre-reset state of auto-reset envs for the recorder

    for step in range(max_total_steps):
        if len(collected_trajs) >= target_count:
            log.info(f"Collected {len(collected_trajs)} successful trajectories, reached target {target_count}")
            break

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())
        dones = terminated | time_out

        # one conversion for the envs still collecting; an env auto-reset on this step is taken from the state its
        # episode ended in, so the terminal (action, state) pair and the lift bookkeeping of that step are kept
        active = [i for i in range(num_eval_envs) if not done_masks[i]]
        # a step that cannot be recorded is a configuration error that shows on the first step: it raises
        records = recorded_step(env.handler, env.handler.get_states(mode="tensor"), infos, active)
        record_of = dict(zip(active, records, strict=True))
        for i in active:
            grasp_success = infos.get("grasp_success", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]
            lift_active = infos.get("lift_active", torch.zeros(num_eval_envs, dtype=torch.bool, device=device))[i]

            current_episode_actions[i].append(commanded_action(record_of[i], robot_name=robot_name))
            current_state = record_of[i].state
            current_episode_states[i].append(current_state)

            if grasp_success and lift_active and not in_lift_phase[i]:
                in_lift_phase[i] = True
                lift_start_state[i] = current_state
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

                        collected_trajs.append(traj_data)  # already plain Python (recorded_step)
                        collected_states.append(lift_start_state[i])

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
            done_ids = newly_done.nonzero(as_tuple=False).squeeze(-1).tolist()
            # every done env (recorded above) starts its next episode from the state it was reset to
            fresh = restart_done_envs(env, done_ids, infos, next_obs, init_states=True)
            for i in done_ids:
                episodes_completed += 1

                lift_start_state[i] = None
                lift_frame_count[i] = 0
                in_lift_phase[i] = False
                recording_traj[i] = False
                current_episode_actions[i] = []
                current_episode_states[i] = []
                current_episode_init_state[i] = fresh[i]
                current_returns[i] = 0
                current_lengths[i] = 0
                # reset per-episode success flag for next episode
                success_in_episode[i] = False

        obs = next_obs

    # an env with an episode in flight (still collecting, and not just reset) is one attempted episode,
    # successful if it already succeeded
    in_flight = [i for i in range(num_eval_envs) if not done_masks[i] and current_lengths[i] > 0]
    active_envs = len(in_flight)
    successes_in_active = sum(1 for i in in_flight if success_in_episode.get(i, False))
    attempted_episodes = episodes_completed + active_envs
    total_successful_episodes = successful_episodes_count + successes_in_active

    if len(collected_trajs) > 0:
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="FastTD3 lift trajectory collection evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/pick_place.approach_grasp_simple_1210000.pt",
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=100,
        help="Target number of successful trajectories to collect (default: 100)",
    )

    parser.add_argument("--device_rank", type=int, default=0, help="GPU device rank")
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of parallel environments (default: from checkpoint config)"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    parser.add_argument("--traj_dir", type=str, default="eval_trajs", help="Trajectory save directory")
    parser.add_argument("--state_dir", type=str, default="eval_states", help="State save directory")
    parser.add_argument(
        "--lift_stable_frames", type=int, default=10, help="Number of frames lift must be maintained (default: 10)"
    )

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
    amp_device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    log.info("Starting lift trajectory collection...")
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
