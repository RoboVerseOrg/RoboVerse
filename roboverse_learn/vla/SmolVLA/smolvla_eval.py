#!/usr/bin/env python3
"""
SmolVLA Evaluation Script for RoboVerse

This script evaluates a fine-tuned SmolVLA model on RoboVerse tasks.
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
from collections import deque
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import metasim
import gymnasium as gym
from metasim.utils import configclass
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.randomization import DomainRandomizationManager, DRConfig
from roboverse_learn.il.runner.base_policy import BasePolicyCfg, ActionCfg, ObsCfg, EndEffectorCfg


@configclass
class SmolVLAPolicyCfg(BasePolicyCfg):
    """Configuration for SmolVLA policy."""
    name: str = "SmolVLAPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="ee",
        delta=1,
        action_dim=7,
        ee_cfg=EndEffectorCfg(rotation_rep="axis_angle", gripper_rep="strength"),
    )
    obs_config: ObsCfg = ObsCfg(obs_type="no_proprio", norm_image=False)


class SmolVLARunner:
    """Runner for SmolVLA model inference on RoboVerse."""

    def __init__(
        self,
        env,
        scenario,
        num_envs: int,
        checkpoint_path: str,
        task_name: str,
        device: str,
        robot_name: str,
        solver: str = "pyroki",
    ):
        self.env = env
        self.scenario = scenario
        self.num_envs = num_envs
        self.device = device
        self.task_name = task_name
        self.robot_name = robot_name
        self.solver = solver
        self.ee_body_name = self.scenario.robots[0].ee_body_name
        self.ee_body_idx = None

        # Get the base environment for accessing handler
        # GymEnvWrapper has a task_env attribute which contains the handler
        env_unwrapped = env
        while hasattr(env_unwrapped, 'unwrapped') and env_unwrapped != env_unwrapped.unwrapped:
            env_unwrapped = env_unwrapped.unwrapped
        # Access handler through task_env if it exists
        if hasattr(env_unwrapped, 'task_env'):
            self.env_base = env_unwrapped.task_env
        else:
            self.env_base = env_unwrapped

        self._init_policy(checkpoint_path=checkpoint_path)
        self._setup_ik()

    def _init_policy(self, checkpoint_path: str):
        """Initialize SmolVLA model."""
        self.model_path = checkpoint_path
        self.policy_cfg = SmolVLAPolicyCfg()

        print(f"Loading SmolVLA model from: {checkpoint_path}")

        # Handle LeRobot checkpoint structure
        pretrained_model_path = os.path.join(checkpoint_path, "pretrained_model")
        if os.path.exists(pretrained_model_path):
            checkpoint_path = pretrained_model_path

        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            self.model = SmolVLAPolicy.from_pretrained(checkpoint_path)
            self.model = self.model.to(self.device).eval()
            self.use_lerobot = True
        except Exception as e:
            try:
                from transformers import AutoModel, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device).eval()
                self.use_lerobot = False
            except Exception as e2:
                raise RuntimeError(f"Failed to load SmolVLA model from {checkpoint_path}: {e2}")

        # Load dataset statistics if available
        stats_path = os.path.join(self.model_path, "dataset_statistics.json")
        self.DATA_STAT = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

        self.obs = deque(maxlen=2)

    def _setup_ik(self):
        """Setup IK solver for end-effector control."""
        from metasim.utils.ik_solver import setup_ik_solver
        self.robot_cfg = self.scenario.robots[0]
        self.ik_solver = setup_ik_solver(self.robot_cfg, self.solver)

    def update_obs(self, current_obs):
        """Update observation buffer."""
        self.obs.append(current_obs)

    @torch.no_grad()
    def predict_action(self, observation=None):
        """
        Predict action from current observation.

        Returns:
            torch.Tensor: Action tensor of shape (B, 7) containing [dx, dy, dz, drx, dry, drz, gripper]
        """
        if observation is not None:
            self.update_obs(observation)
        if len(self.obs) == 0:
            raise ValueError("No observations available")

        latest_obs = self.obs[-1]

        # Extract RGB image from first camera
        first_cam = next(iter(latest_obs.cameras.values()))
        rgb_data = first_cam.rgb
        if rgb_data.dim() == 4:
            rgb_data = rgb_data[0]
        image = Image.fromarray(rgb_data.detach().cpu().numpy())

        # Get task description (unwrap to access task_env)
        task_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(task_env, 'task_env'):
            task_env = task_env.task_env
        instruction = getattr(task_env, "task_desc", self.task_name)

        if self.use_lerobot:
            # Prepare input batch for LeRobot model
            from torchvision.transforms import ToTensor
            import torch

            # Convert image to tensor and add batch dimension
            image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)

            # Get current robot state from joint positions
            rs = latest_obs.robots[self.robot_name]
            joint_pos_raw = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)

            # Flatten and prepare state tensor
            if joint_pos_raw.dim() > 1:
                state = joint_pos_raw.flatten()
            else:
                state = joint_pos_raw

            state_tensor = state.float().unsqueeze(0).to(self.device)

            # Tokenize instruction
            tokenized = self.model.model.vlm_with_expert.processor.tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            batch = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "observation.language.tokens": tokenized["input_ids"].to(self.device),
                "observation.language.attention_mask": tokenized["attention_mask"].bool().to(self.device),
            }

            action = self.model.select_action(batch).squeeze(0).cpu().numpy()
        else:
            prompt = f"What action should the robot take to {instruction}?"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype)
                     for k, v in inputs.items()}
            outputs = self.model(**inputs)
            action = outputs.action if hasattr(outputs, 'action') else outputs[0]

        # Convert to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        # Ensure correct shape
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return action

    def ee_control_actions(self, obs):
        """
        Convert VLA action to end-effector control and solve IK.

        Args:
            obs: Observation from environment

        Returns:
            List of action dictionaries for each environment
        """
        from pytorch3d import transforms

        # Get action from VLA
        with torch.no_grad():
            action = self.predict_action(obs)

        num_envs = action.shape[0]
        rs = obs.robots[self.robot_name]

        # Handle joint reordering (use base env to access handler)
        reorder_idx = self.env_base.handler.get_joint_reindex(self.robot_name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        joint_pos_raw = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)
        curr_robot_q = joint_pos_raw[:, inverse_reorder_idx].to(self.device).float()

        robot_ee_state = (rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state)).to(self.device).float()
        robot_root_state = (rs.root_state if isinstance(rs.root_state, torch.Tensor) else torch.tensor(rs.root_state)).to(self.device).float()

        # Get end-effector pose
        if self.ee_body_idx is None:
            self.ee_body_idx = rs.body_names.index(self.ee_body_name)
        ee_p_world = robot_ee_state[:, self.ee_body_idx, 0:3]
        ee_q_world = robot_ee_state[:, self.ee_body_idx, 3:7]

        # Transform to local frame
        robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
        inv_base_q = transforms.quaternion_invert(robot_quat)
        curr_ee_pos_local = transforms.quaternion_apply(inv_base_q, ee_p_world - robot_pos)
        curr_ee_quat_local = transforms.quaternion_multiply(inv_base_q, ee_q_world)

        # Apply action deltas
        ee_pos_delta = action[:num_envs, :3]
        ee_rot_delta = action[:num_envs, 3:6]
        gripper_open = action[:num_envs, 6]

        # Convert rotation delta to quaternion
        ee_quat_delta = transforms.matrix_to_quaternion(
            transforms.euler_angles_to_matrix(ee_rot_delta, "XYZ")
        )

        # Compute target pose
        ee_pos_target = curr_ee_pos_local + ee_pos_delta
        ee_quat_target = transforms.quaternion_multiply(curr_ee_quat_local, ee_quat_delta)

        # Solve IK
        q_solution, ik_succ = self.ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)

        # Process gripper command
        from metasim.utils.ik_solver import process_gripper_command
        gripper_widths = process_gripper_command(gripper_open, self.robot_cfg, self.device)

        # Compose final action
        actions = self.ik_solver.compose_joint_action(
            q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True
        )
        return actions

    def reset(self):
        """Reset the runner state."""
        self.obs.clear()


def evaluate_episode(
    env,
    runner: SmolVLARunner,
    max_steps: int,
    episode_num: int,
    output_dir: str,
    randomization_manager=None,
    demo_idx: int = 0,
    init_states=None,
) -> Dict[str, Any]:
    """Evaluate a single episode and save results."""

    # Apply domain randomization before reset
    if randomization_manager is not None:
        randomization_manager.apply_randomization(demo_idx=demo_idx, is_initial=(episode_num == 1))
        randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=0)
        randomization_manager.update_camera_look_at(env_id=0)
        randomization_manager.apply_camera_randomization()

    # Reset environment - must use task_env directly to pass states
    # (GymEnvWrapper.reset() doesn't support states parameter)
    if randomization_manager is not None and init_states is not None:
        from roboverse_learn.il.act.act_eval_runner import ensure_clean_state

        # Access task_env through wrapper
        env_unwrapped = env
        while hasattr(env_unwrapped, 'unwrapped') and env_unwrapped != env_unwrapped.unwrapped:
            env_unwrapped = env_unwrapped.unwrapped
        task_env = env_unwrapped.task_env if hasattr(env_unwrapped, 'task_env') else env_unwrapped

        # Reset to specific state
        obs, info = task_env.reset(states=[init_states[demo_idx]], env_ids=[0])

        # Ensure environment stabilizes after reset
        ensure_clean_state(task_env.handler, expected_state=init_states[demo_idx])

        # Update gym wrapper state to avoid "ResetNeeded" error
        wrapper = env
        while wrapper is not env_unwrapped:
            if hasattr(wrapper, '_has_reset'):
                wrapper._has_reset = True
            if hasattr(wrapper, '_episode_steps'):
                wrapper._episode_steps[0] = 0
            wrapper = wrapper.unwrapped if hasattr(wrapper, 'unwrapped') else wrapper.env
    else:
        obs, info = env.reset()

    stats = {
        "steps": 0,
        "success": False,
        "total_reward": 0.0,
        "start_time": time.time()
    }
    runner.reset()

    # Initialize video saver
    os.makedirs(output_dir, exist_ok=True)
    obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode_num:03d}.mp4")
    obs_saver.add(obs)

    for step in range(max_steps):
        actions = runner.ee_control_actions(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        stats["steps"] += 1

        # Handle both tensor and scalar rewards
        if isinstance(reward, torch.Tensor):
            stats["total_reward"] += float(reward.mean().item() if reward.numel() > 1 else reward.item())
        else:
            stats["total_reward"] += float(reward)

        obs_saver.add(obs)

        # Handle both boolean and tensor terminated/truncated
        is_terminated = terminated.any().item() if isinstance(terminated, torch.Tensor) else terminated
        is_truncated = truncated.any().item() if isinstance(truncated, torch.Tensor) else truncated

        if is_terminated or is_truncated:
            stats["success"] = True
            break

    obs_saver.save()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    return stats


def main():
    """Main evaluation function."""
    # Set multiprocessing start method for CUDA compatibility
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="SmolVLA Evaluation on RoboVerse")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SmolVLA checkpoint")
    parser.add_argument("--task", type=str, default="pick_butter", help="Task name")
    parser.add_argument("--robot", type=str, default="franka", help="Robot name")
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        choices=["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"],
        help="Simulator backend",
    )
    parser.add_argument("--solver", type=str, default="pyroki", choices=["curobo", "pyroki"], help="IK solver")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=250, help="Maximum steps per episode")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./smolvla_eval_output", help="Output directory for videos and results"
    )

    # Domain Randomization options
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"
    )
    parser.add_argument(
        "--scene_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"
    )
    parser.add_argument(
        "--randomization_seed",
        type=int,
        default=None,
        help="Seed for reproducible randomization. If None, uses random seed"
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"SmolVLA Evaluation")
    print(f"  Task: {args.task}")
    print(f"  Robot: {args.robot}")
    print(f"  Simulator: {args.sim}")
    print(f"  IK Solver: {args.solver}")
    print(f"  Device: {args.device}")
    print(f"  DR Level: {args.level}, Scene Mode: {args.scene_mode}, Seed: {args.randomization_seed}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Camera configuration
    if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
        dp_camera = True
    else:
        dp_camera = args.task != "close_box"

    is_libero_dataset = "libero_90" in args.task

    if is_libero_dataset:
        dp_pos = (2.0, 0.0, 2)
    elif dp_camera:
        dp_pos = (1.0, 0.0, 0.75)
    else:
        dp_pos = (1.5, 0.0, 1.5)

    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb"],
        width=256,
        height=256,
        pos=dp_pos,
        look_at=(0.0, 0.0, 0.0),
    )

    # Lighting setup
    render_mode = "raytracing"
    ceiling_main = 12000.0
    ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
        ),
    ]

    # Create environment
    env = gym.make(
        f"RoboVerse/{args.task}",
        robots=[args.robot],
        simulator=args.sim,
        headless=True,
        cameras=[camera],
        lights=lights,
        device=args.device,
    )

    # Access the actual task environment through wrapper
    env_unwrapped = env
    while hasattr(env_unwrapped, 'unwrapped') and env_unwrapped != env_unwrapped.unwrapped:
        env_unwrapped = env_unwrapped.unwrapped
    task_env = env_unwrapped.task_env if hasattr(env_unwrapped, 'task_env') else env_unwrapped

    # Load trajectories for domain randomization
    traj_filepath = task_env.traj_filepath
    robot_obj = get_robot(args.robot)
    init_states, _, _ = get_traj(traj_filepath, robot_obj, task_env.handler)

    # Initialize Domain Randomization
    randomization_manager = None
    if args.level > 0:
        from dataclasses import dataclass as dc
        @dc
        class SimpleRenderCfg:
            mode: str = render_mode

        randomization_manager = DomainRandomizationManager(
            config=DRConfig(
                level=args.level,
                scene_mode=args.scene_mode,
                randomization_seed=args.randomization_seed,
            ),
            scenario=env.unwrapped.scenario,
            handler=task_env.handler,
            init_states=init_states,
            render_cfg=SimpleRenderCfg(mode=render_mode)
        )
        print(f"Domain Randomization enabled: level={args.level}, scene_mode={args.scene_mode}, seed={args.randomization_seed}")

    # Create runner
    runner = SmolVLARunner(
        env=env,
        scenario=env.unwrapped.scenario,
        num_envs=1,
        checkpoint_path=args.model_path,
        task_name=args.task,
        device=args.device,
        robot_name=args.robot,
        solver=args.solver,
    )

    # Run evaluation
    start_time = time.time()
    eval_stats = {
        "total_episodes": 0,
        "total_successes": 0,
        "total_rewards": [],
        "episode_results": []
    }

    for ep in range(args.num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'=' * 50}")

        demo_idx = ep % len(init_states) if randomization_manager is not None else 0

        ep_res = evaluate_episode(
            env, runner, args.max_steps, ep + 1, args.output_dir,
            randomization_manager=randomization_manager,
            demo_idx=demo_idx,
            init_states=init_states if randomization_manager is not None else None
        )

        eval_stats["total_episodes"] += 1
        if ep_res["success"]:
            eval_stats["total_successes"] += 1
        eval_stats["total_rewards"].append(ep_res["total_reward"])
        eval_stats["episode_results"].append(ep_res)

        sr = eval_stats["total_successes"] / eval_stats["total_episodes"]
        print(f"  Steps: {ep_res['steps']}")
        print(f"  Success: {ep_res['success']}")
        print(f"  Reward: {ep_res['total_reward']:.2f}")
        print(f"  Success rate: {sr:.1%}")

    total_time = time.time() - start_time
    final_sr = eval_stats["total_successes"] / eval_stats["total_episodes"]
    final_avg_reward = float(np.mean(eval_stats["total_rewards"])) if eval_stats["total_rewards"] else 0.0

    print(f"\n{'=' * 50}")
    print("Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"Success Rate: {final_sr:.1%} ({eval_stats['total_successes']}/{eval_stats['total_episodes']})")
    print(f"Average Reward: {final_avg_reward:.2f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"DR Level: {args.level}, Scene Mode: {args.scene_mode}, Seed: {args.randomization_seed}")
    print(f"{'=' * 50}")

    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.output_dir, f"smolvla_eval_{args.task}_{ts}.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": vars(args),
            "eval_stats": eval_stats,
            "timestamp": ts,
            "dr_config": {"level": args.level, "scene_mode": args.scene_mode, "seed": args.randomization_seed}
        }, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Cleanup
    try:
        env.close()
    except Exception:
        pass

    return final_sr > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
