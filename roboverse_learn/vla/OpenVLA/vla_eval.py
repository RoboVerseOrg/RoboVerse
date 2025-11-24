#!/usr/bin/env python3
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
from scipy.spatial.transform import Rotation
sys.path.append(str(Path(__file__).parent.parent.parent))

# from metasim.task.gym_registration import make_vec
import metasim
from gymnasium import make_vec
from metasim.utils import configclass
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.obs_utils import ObsSaver
from roboverse_learn.il.dp.runner.base_policy import BasePolicyCfg, ActionCfg, ObsCfg, EndEffectorCfg


@configclass
class VLAPolicyCfg(BasePolicyCfg):
    name: str = "VLAPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="ee",
        delta=1,
        action_dim=7,
        ee_cfg=EndEffectorCfg(rotation_rep="axis_angle", gripper_rep="strength"),
    )
    obs_config: ObsCfg = ObsCfg(obs_type="no_proprio", norm_image=False)


class OpenVLARunner:
    def __init__(
        self,
        env,
        scenario,
        num_envs: int,
        checkpoint_path: str,
        task_name: str,
        subset: str,
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

        self._init_policy(checkpoint_path=checkpoint_path, task_name=task_name, subset=subset)
        self._setup_ik()

    # ---------------- Model ----------------
    def _init_policy(self, **kwargs):
        self.model_path = kwargs.get("checkpoint_path")
        self.task = kwargs.get("task_name")
        self.subset = kwargs.get("subset")


        self.policy_cfg = VLAPolicyCfg()
        self.policy_cfg.obs_config.obs_type = "no_proprio"

        stats_path = os.path.join(self.model_path, "dataset_statistics.json")
        self.DATA_STAT = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(self.device).eval()

        # Important: give norm stats to the model; unnorm_key used in predict_action
        self.model.norm_stats = self.DATA_STAT

        self.obs = deque(maxlen=2)

    # ---------------- IK ----------------
    def _setup_ik(self):
        from metasim.utils.ik_solver import setup_ik_solver
        self.robot_cfg = self.scenario.robots[0]
        self.ik_solver = setup_ik_solver(self.robot_cfg, self.solver)

    # ---------------- Per-step helpers ----------------
    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    @torch.no_grad()
    def predict_action(self, observation=None):
        """VLA forward: returns (B,7) in metric/radian units (dpos, drot, gripper)."""
        if observation is not None:
            self.update_obs(observation)
        if len(self.obs) == 0:
            raise ValueError("No observations available")


        latest_obs = self.obs[-1]
        # Take first camera
        first_cam = next(iter(latest_obs.cameras.values()))
        rgb_data = first_cam.rgb
        x = rgb_data[0].detach().cpu() if rgb_data.dim() == 4 else rgb_data.detach().cpu()
        image = x.numpy()
        image = Image.fromarray(image)

        # instruction = self.env.task_env.task_desc
        if hasattr(self.env.task_env, "task_desc"):
            instruction = self.env.task_env.task_desc
        else:
            # generate by task name
            task_desc = self.task_name.replace('_', ' ')
            instruction = task_desc[0].upper() + task_desc[1:]
        # instruction = self.env.task_env.task_desc
        # 'Pick up the butter and place it in the basket'  for pick butter tasks

        # Process inputs manually for OpenVLAForActionPrediction
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype)
                 for k, v in inputs.items()}

        # Use the model's predict_action method with input_ids
        with torch.no_grad():
            action = self.model.predict_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                unnorm_key="bridge_orig",
                do_sample=False
            )

        print(f"VLA model output (denormalized): pos={action[:3]}, rot={action[3:6]}, gripper={action[6]}")
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        return action.unsqueeze(0) if (self.num_envs == 1 and action.dim() == 1) else action

    # ---------------- Quaternion utilities (scipy-based) ----------------
    @staticmethod
    def quat_to_scipy(quat_torch):
        """Convert pytorch3d quaternion (w,x,y,z) to scipy format (x,y,z,w)."""
        # Input: (B, 4) tensor with (w, x, y, z)
        # Output: (B, 4) numpy array with (x, y, z, w)
        quat_np = quat_torch.cpu().numpy()
        return np.concatenate([quat_np[:, 1:], quat_np[:, 0:1]], axis=-1)

    @staticmethod
    def quat_from_scipy(quat_scipy, device):
        """Convert scipy quaternion (x,y,z,w) to pytorch3d format (w,x,y,z)."""
        # Input: (B, 4) numpy array with (x, y, z, w)
        # Output: (B, 4) tensor with (w, x, y, z)
        quat_torch = np.concatenate([quat_scipy[:, 3:], quat_scipy[:, :3]], axis=-1)
        return torch.from_numpy(quat_torch).to(device).float()

    # ---------------- EE control + IK ----------------
    def ee_control_actions(self, obs) -> list[dict]:
        """Δ-pose (local) -> target EE pose -> cuRobo IK -> joint targets."""
        # 1) VLA action
        with torch.no_grad():                     # only VLA forward is no-grad
            action = self.predict_action(obs)     # (B,7)
        num_envs = action.shape[0]

        # 2) Robot state (TensorState -> tensors)
        rs = obs.robots[self.robot_name]

        # IK solver expects original joint order, but state uses alphabetical order
        reorder_idx = self.env.task_env.handler.get_joint_reindex(self.robot_name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        joint_pos_raw = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)
        curr_robot_q = joint_pos_raw[:, inverse_reorder_idx].to(self.device).float()
        robot_ee_state = (rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state)).to(self.device).float()
        robot_root_state = (rs.root_state if isinstance(rs.root_state, torch.Tensor) else torch.tensor(rs.root_state)).to(self.device).float()

        if self.ee_body_idx is None:
            self.ee_body_idx = rs.body_names.index(self.ee_body_name)
        ee_p_world = robot_ee_state[:, self.ee_body_idx, 0:3]
        ee_q_world = robot_ee_state[:, self.ee_body_idx, 3:7]
        # print(f"EE position in world: {ee_p_world}")

        # Base pose
        robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
        # print(f"Robot position in world: {robot_pos}")

        # Local frame transform using scipy
        # Convert to scipy format and use Rotation for quaternion operations
        robot_quat_scipy = self.quat_to_scipy(robot_quat)
        ee_q_world_scipy = self.quat_to_scipy(ee_q_world)

        # Invert base quaternion
        inv_base_rot = Rotation.from_quat(robot_quat_scipy).inv()

        # Apply rotation to position vector
        ee_p_relative = (ee_p_world - robot_pos).cpu().numpy()
        curr_ee_pos_local_np = inv_base_rot.apply(ee_p_relative)
        curr_ee_pos_local = torch.from_numpy(curr_ee_pos_local_np).to(self.device).float()

        # Multiply quaternions: inv_base_q * ee_q_world
        curr_ee_rot_local = inv_base_rot * Rotation.from_quat(ee_q_world_scipy)
        curr_ee_quat_local = self.quat_from_scipy(curr_ee_rot_local.as_quat(), self.device)

        # 3) Apply deltas
        ee_pos_delta = action[:num_envs, :3]
        ee_rot_delta = action[:num_envs, 3:-1]

        # Convert euler angles to quaternion using scipy
        ee_rot_delta_np = ee_rot_delta.cpu().numpy()
        ee_quat_delta_rot = Rotation.from_euler('XYZ', ee_rot_delta_np)
        ee_quat_delta = self.quat_from_scipy(ee_quat_delta_rot.as_quat(), self.device)

        gripper_open = action[:num_envs, -1]
        ee_pos_target = curr_ee_pos_local + ee_pos_delta

        # Multiply quaternions: curr_ee_quat_local * ee_quat_delta
        curr_ee_quat_local_scipy = self.quat_to_scipy(curr_ee_quat_local)
        ee_quat_delta_scipy = self.quat_to_scipy(ee_quat_delta)
        ee_quat_target_rot = Rotation.from_quat(curr_ee_quat_local_scipy) * Rotation.from_quat(ee_quat_delta_scipy)
        ee_quat_target = self.quat_from_scipy(ee_quat_target_rot.as_quat(), self.device)


        # 4) IK (seed = current q)
        q_solution, ik_succ = self.ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q)
        if not ik_succ.all():
            print(f"WARNING: IK failed for {(~ik_succ).sum().item()}/{num_envs} environments")

        # 5) Gripper control
        from metasim.utils.ik_solver import process_gripper_command
        gripper_widths = process_gripper_command(gripper_open, self.robot_cfg, self.device)

        # Compose robot command
        actions = self.ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)
        return actions

    def reset(self):
        self.obs.clear()


def evaluate_episode(env, runner: OpenVLARunner, max_steps: int, episode_num: int, output_dir: str) -> Dict[str, Any]:
    obs, info = env.reset()
    stats = {"steps": 0, "success": False, "total_reward": 0.0, "start_time": time.time()}
    runner.reset()

    # Initialize obs saver for this episode
    os.makedirs(output_dir, exist_ok=True)
    obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode_num:03d}.mp4")
    obs_saver.add(obs)

    for step in range(max_steps):
        actions = runner.ee_control_actions(obs)  # use EE control + IK
        obs, reward, terminated, truncated, info = env.step(actions)
        stats["steps"] += 1
        stats["total_reward"] += float(reward.mean().item())

        # Save observation for video
        obs_saver.add(obs)

        # Check termination: only terminated=True means success, truncated=True means timeout (failure)
        is_terminated = terminated.any().item() if hasattr(terminated, "any") else bool(terminated)
        is_truncated = truncated.any().item() if hasattr(truncated, "any") else bool(truncated)

        if is_terminated:
            stats["success"] = True
            print(f"Task succeeded at step {step + 1}")
            break
        elif is_truncated:
            print(f"Task failed: timeout at step {step + 1}")
            break

    # Save the episode video
    obs_saver.save()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    return stats


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Evaluation (EE control + IK)")
    parser.add_argument("--model_path", type=str, default="openvla_runs/openvla-7b+roboverse_dataset+b16+lr-0.0005+lora-r32+dropout-0.0")
    parser.add_argument("--task", type=str, default="pick_butter")
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--sim", type=str, default="mujoco",
                        choices=["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"])
    parser.add_argument("--solver", type=str, default="pyroki", choices=["curobo", "pyroki"],
                        help="IK solver to use: curobo or pyroki")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    args = parser.parse_args()


    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"OpenVLA Eval: task={args.task} robot={args.robot} sim={args.sim} solver={args.solver} device={args.device}")

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
        headless=True,
        cameras=[PinholeCameraCfg(
            name="camera",
            data_types=["rgb"],
            width=256,
            height=256,
            pos=(1.5, 0.0, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )],
        device=args.device,
    )

    runner = OpenVLARunner(
        env=env,
        scenario=env.scenario,
        num_envs=args.num_envs,
        checkpoint_path=args.model_path,
        task_name=args.task,
        subset=args.task,
        device=args.device,
        robot_name=args.robot,
        solver=args.solver,
    )

    start_time = time.time()
    eval_stats = {"total_episodes": 0, "total_successes": 0, "total_rewards": [], "episode_results": []}

    for ep in range(args.num_episodes):
        print(f"Episode {ep + 1}/{args.num_episodes}")
        ep_res = evaluate_episode(env, runner, args.max_steps, ep + 1, args.output_dir)
        eval_stats["total_episodes"] += 1
        if ep_res["success"]:
            eval_stats["total_successes"] += 1
        eval_stats["total_rewards"].append(ep_res["total_reward"])
        eval_stats["episode_results"].append(ep_res)
        sr = eval_stats["total_successes"] / eval_stats["total_episodes"]
        print(f"  Success rate: {sr:.1%}")

    total_time = time.time() - start_time
    final_sr = eval_stats["total_successes"] / eval_stats["total_episodes"]
    final_avg_reward = float(np.mean(eval_stats["total_rewards"])) if len(eval_stats["total_rewards"]) else 0.0
    print(f"\nEvaluation completed: {final_sr:.1%} | {final_avg_reward:.2f} | {total_time:.1f}s")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"openvla_eval_{args.task}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "eval_stats": eval_stats, "timestamp": ts}, f, indent=2, ensure_ascii=False)

    try:
        env.close()
    except Exception:
        pass
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
