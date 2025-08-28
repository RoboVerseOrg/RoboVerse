#!/usr/bin/env python3
"""
OpenVLA Complete Evaluation Script

Complete evaluation script with OpenVLARunner class and VLAPolicyCfg configuration class.

Usage:
    python eval_openvla_simple.py \
        --model_path <CHECKPOINT_PATH> \
        --task <TASK_NAME> \
        --robot <ROBOT_NAME> \
        --sim <SIMULATOR> \
        --num_envs 1 \
        --num_episodes 10 \
        --max_steps 100
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.append(str(Path(__file__).parent.parent.parent))

from metasim.task.gym_registration import make_vec
from roboverse_learn.il.runner.base_policy import BasePolicyCfg, ActionCfg, ObsCfg, EndEffectorCfg
from metasim.utils import configclass


@configclass
class VLAPolicyCfg(BasePolicyCfg):
    name: str = "VLAPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="ee",
        delta=1,
        action_dim=7,
        ee_cfg=EndEffectorCfg(rotation_rep="axis_angle", gripper_rep="strength"),
    )
    obs_config: ObsCfg = ObsCfg(
        obs_type="no_proprio",
        norm_image=False,
    )


class OpenVLARunner():
    def __init__(self, env, scenario, num_envs: int, checkpoint_path: str, task_name: str, subset: str, device: str):
        self.env = env
        self.scenario = scenario
        self.num_envs = num_envs
        self.device = device
        self.task_name = task_name
        self._init_policy(checkpoint_path=checkpoint_path, task_name=task_name, subset=subset)

    def _init_policy(self, **kwargs):
        self.model_path = kwargs.get("checkpoint_path")
        self.task = kwargs.get("task_name")
        self.subset = kwargs.get("subset")

        self.policy_cfg = VLAPolicyCfg()
        self.policy_cfg.obs_config.obs_type = "no_proprio"

        self.dataset_info_path = os.path.join(self.model_path, "dataset_statistics.json")
        if os.path.exists(self.dataset_info_path):
            self.DATA_STAT = json.load(open(self.dataset_info_path))
        else:
            self.DATA_STAT = {}

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        if self.subset in self.model.norm_stats:
            self.model.norm_stats[self.task.lower()] = self.DATA_STAT.get(self.subset, {})

        self.model.eval()
        self.obs = deque(maxlen=2)

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def predict_action(self, observation=None):
        if observation is not None:
            self.update_obs(observation)

        if len(self.obs) == 0:
            raise ValueError("No observations available")

        latest_obs = self.obs[-1]

        if "rgb" in latest_obs:
            rgb_data = latest_obs["rgb"]
            if isinstance(rgb_data, torch.Tensor):
                if rgb_data.dim() == 4:
                    image = rgb_data[0].cpu().permute(1, 2, 0).numpy()
                else:
                    image = rgb_data.cpu().numpy()
            else:
                image = np.array(rgb_data)

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            image = Image.fromarray(image)
        else:
            raise ValueError("No rgb found in observation")

        try:
            instruction = self.env.task_language
        except AttributeError:
            instruction = self.task_name
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.task.lower(), do_sample=False)

        action = torch.tensor(action, dtype=torch.float32).to(self.device)

        if self.num_envs == 1 and action.dim() == 1:
            action = action.unsqueeze(0)

        return action

    def get_action(self, obs):
        action = self.predict_action(obs)
        return action

    def process_obs(self, obs):
        obs = obs.copy()

        if "rgb" in obs:
            if isinstance(obs["rgb"], torch.Tensor):
                obs["rgb"] = obs["rgb"].to(self.device)
            else:
                obs["rgb"] = torch.tensor(obs["rgb"], device=self.device)

        return obs

    def reset(self):
        self.obs.clear()


def evaluate_episode(env, runner: OpenVLARunner, max_steps: int) -> Dict[str, Any]:
    obs, info = env.reset()

    episode_stats = {
        "steps": 0,
        "success": False,
        "total_reward": 0.0,
        "start_time": time.time()
    }

    runner.reset()

    for step in range(max_steps):
        try:
            actions = runner.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(actions)

            episode_stats["steps"] += 1
            episode_stats["total_reward"] += float(reward.mean().item())

            if terminated.any() or truncated.any():
                episode_stats["success"] = True
                break

        except Exception as e:
            print(f"Step {step} error: {e}")
            break

    episode_stats["end_time"] = time.time()
    episode_stats["duration"] = episode_stats["end_time"] - episode_stats["start_time"]

    return episode_stats


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Evaluation Script")

    parser.add_argument("--model_path", type=str, default="/home/balen/murphy/ROSE/RoboVerse/roboverse_data/openvla-7b", help="OpenVLA checkpoint path")
    parser.add_argument("--task", type=str, default="stack_cube", help="Task name")
    parser.add_argument("--robot", type=str, default="franka", help="Robot name")
    parser.add_argument("--sim", type=str, default="mujoco",
                       choices=["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"],
                       help="Simulator")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Output directory")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"OpenVLA Evaluation: {args.task} | {args.robot} | {args.sim} | {args.device}")

    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        env = make_vec(
            f"RoboVerse/{args.task}",
            num_envs=args.num_envs,
            robots=[args.robot],
            simulator=args.sim,
            headless=False,
            cameras=[],
            device=args.device,
        )

        runner = OpenVLARunner(
            env=env,
            scenario=env.scenario,
            num_envs=args.num_envs,
            checkpoint_path=args.model_path,
            task_name=args.task,
            subset=args.task,
            device=args.device
        )

        start_time = time.time()

        eval_stats = {
            "total_episodes": 0,
            "total_successes": 0,
            "total_rewards": [],
            "episode_results": []
        }

        for episode in range(args.num_episodes):
            print(f"Episode {episode + 1}/{args.num_episodes}")

            episode_result = evaluate_episode(env, runner, args.max_steps)

            eval_stats["total_episodes"] += 1
            if episode_result["success"]:
                eval_stats["total_successes"] += 1

            eval_stats["total_rewards"].append(episode_result["total_reward"])
            eval_stats["episode_results"].append(episode_result)

            current_success_rate = eval_stats["total_successes"] / eval_stats["total_episodes"]
            print(f"  Success rate: {current_success_rate:.1%}")

        total_time = time.time() - start_time
        final_success_rate = eval_stats["total_successes"] / eval_stats["total_episodes"]
        final_avg_reward = np.mean(eval_stats["total_rewards"])

        print(f"\nEvaluation completed: {final_success_rate:.1%} | {final_avg_reward:.2f} | {total_time:.1f}s")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(args.output_dir, f"openvla_eval_{args.task}_{timestamp}.json")

            results = {
                "config": vars(args),
                "eval_stats": eval_stats,
                "timestamp": timestamp
            }

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        try:
            env.close()
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
