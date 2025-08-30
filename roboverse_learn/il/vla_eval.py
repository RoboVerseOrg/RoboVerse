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
from pytorch3d import transforms
from curobo.types.math import Pose

sys.path.append(str(Path(__file__).parent.parent.parent))

from metasim.task.gym_registration import make_vec
from metasim.utils import configclass
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.obs_utils import ObsSaver
from roboverse_learn.il.runner.base_policy import BasePolicyCfg, ActionCfg, ObsCfg, EndEffectorCfg
from metasim.utils.kinematics_utils import get_curobo_models


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
    ):
        self.env = env
        self.scenario = scenario
        self.num_envs = num_envs
        self.device = device
        self.task_name = task_name
        self.robot_name = robot_name
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
            attn_implementation="eager",
        ).to(self.device).eval()

        # Important: give norm stats to the model; unnorm_key used in predict_action
        self.model.norm_stats = self.DATA_STAT

        self.obs = deque(maxlen=2)

    # ---------------- IK ----------------
    def _setup_ik(self):
        self.robot_cfg = self.scenario.robots[0]
        *_, self.robot_ik = get_curobo_models(self.robot_cfg)
        self.joint_names = list(self.robot_cfg.joint_limits.keys())
        self.n_robot_dof = len(self.joint_names)
        self.curobo_n_dof = len(self.robot_ik.robot_config.cspace.joint_names)
        self.ee_n_dof = len(self.robot_cfg.gripper_open_q)

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
        print(f"RGB data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, min: {rgb_data.min()}, max: {rgb_data.max()}")

        if isinstance(rgb_data, torch.Tensor):
            x = rgb_data[0].detach().cpu() if rgb_data.dim() == 4 else rgb_data.detach().cpu()
            if x.ndim == 3 and x.shape[-1] in (3, 4):        # HWC
                image = x.numpy()
            elif x.ndim == 3 and x.shape[0] in (1, 3, 4):    # CHW -> HWC
                image = x.permute(1, 2, 0).numpy()
            else:
                raise ValueError(f"Unexpected RGB shape: {tuple(x.shape)}")
        else:
            image = np.array(rgb_data)

        image = Image.fromarray(image)
        instruction = self.env.env.task_language
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)


        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        print(f"Raw VLA model output: {action}")
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        return action.unsqueeze(0) if (self.num_envs == 1 and action.dim() == 1) else action

    # ---------------- EE control + IK ----------------
    def ee_control_actions(self, obs) -> list[dict]:
        """Δ-pose (local) -> target EE pose -> cuRobo IK -> joint targets."""
        # 1) VLA action
        with torch.no_grad():                     # only VLA forward is no-grad
            action = self.predict_action(obs)     # (B,7)
        num_envs = action.shape[0]

        # 2) Robot state (TensorState -> tensors)
        rs = obs.robots[self.robot_name]
        curr_robot_q = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).to(self.device).float()
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
        # Local frame transform
        inv_base_q = transforms.quaternion_invert(robot_quat)
        curr_ee_pos_local = transforms.quaternion_apply(inv_base_q, ee_p_world - robot_pos)
        curr_ee_quat_local = transforms.quaternion_multiply(inv_base_q, ee_q_world)

        # 3) Apply deltas
        ee_pos_delta = action[:num_envs, :3]
        ee_rot_delta = action[:num_envs, 3:-1]
        ee_quat_delta = transforms.matrix_to_quaternion(
            transforms.euler_angles_to_matrix(ee_rot_delta, "XYZ")
        )
        gripper_open = action[:num_envs, -1]
        # print(f"curr_ee_pos_locald: {curr_ee_pos_local}")
        ee_pos_target = curr_ee_pos_local + ee_pos_delta
        ee_quat_target = transforms.quaternion_multiply(curr_ee_quat_local, ee_quat_delta)

        # print(f"Delta position (local): {ee_pos_delta}")
        # print(f"Target position (world): {ee_pos_target}")

        # 4) IK (seed = current q)
        seed_config = curr_robot_q[:, :self.curobo_n_dof].unsqueeze(1).tile([1, self.robot_ik._num_seeds, 1])
        result = self.robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

        # 5) Gripper control
        gripper_pos = 1 - gripper_open.cpu()
        if gripper_pos.mean().item() > 0.5:
            gripper_widths = torch.tensor(self.robot_cfg.gripper_close_q).to(self.device)
        else:
            gripper_widths = torch.tensor(self.robot_cfg.gripper_open_q).to(self.device)

        # Compose robot command
        q = curr_robot_q.clone()
        ik_succ = result.success.squeeze(1)
        # print("IK success flags:", ik_succ)
        q[ik_succ, :self.curobo_n_dof] = result.solution[ik_succ, 0].clone()
        q[:, -self.ee_n_dof:] = gripper_widths

        q_use = q[:, :self.n_robot_dof]
        actions = [
            {self.robot_name: {"dof_pos_target": {jn: float(q_use[i, j]) for j, jn in enumerate(self.joint_names)}}}
            for i in range(num_envs)
        ]
        print(f"final actions {actions}")
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

        if (hasattr(terminated, "any") and terminated.any()) or (hasattr(truncated, "any") and truncated.any()):
            stats["success"] = True
            break

    # Save the episode video
    obs_saver.save()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    return stats


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Evaluation (EE control + cuRobo IK)")
    parser.add_argument("--model_path", type=str, default="/datasets/v2p/current/murphy/vla_model/openvla-7b")
    parser.add_argument("--task", type=str, default="pick_butter")
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--sim", type=str, default="mujoco",
                        choices=["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    args = parser.parse_args()


    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"OpenVLA Eval: task={args.task} robot={args.robot} sim={args.sim} device={args.device}")

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