from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gymnasium import make_vec
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.randomization import DomainRandomizationManager, DRConfig

from openpi_client import image_tools, websocket_client_policy

from roboverse_learn.il.runner.base_policy import ActionCfg, BasePolicyCfg, ObsCfg


@configclass
class PiPolicyCfg(BasePolicyCfg):
    name: str = "PiPolicy"
    action_config: ActionCfg = ActionCfg(
        action_type="joint_pos",
        action_dim=9,
        delta=0,
        temporal_agg=True,
    )
    obs_config: ObsCfg = ObsCfg(obs_type="joint_pos", norm_image=False)


class PiPolicyRunner:
    """Helper that queries a running π policy server and formats RoboVerse observations."""

    def __init__(
        self,
        env,
        scenario,
        num_envs: int,
        robot_name: str,
        solver: str,
        policy_host: str,
        policy_port: int,
        image_size: int = 224,
        gripper_threshold: float = 0.02,
        device: str = "cuda",
        actions_per_call: int | None = None,
    ):
        if num_envs != 1:
            raise ValueError("pi_eval currently supports num_envs == 1")

        self.env = env
        self.scenario = scenario
        self.robot_name = robot_name
        self.image_size = image_size
        self.gripper_threshold = gripper_threshold
        self.device = device
        self.actions_per_call = actions_per_call if actions_per_call and actions_per_call > 0 else None

        self.policy_cfg = PiPolicyCfg()
        self.client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)

        self.robot_cfg = self.scenario.robots[0]
        self.ik_solver = setup_ik_solver(self.robot_cfg, solver)
        self.reorder_idx = None
        self.inverse_reorder_idx = None

        self.cached_actions: np.ndarray | None = None
        self.cache_index: int = 0
        self.cache_remaining: int = 0

    # ------------------------------------------------------------------
    def _ensure_reindex(self) -> None:
        if self.reorder_idx is not None:
            return
        handler = self.env.task_env.handler
        self.reorder_idx = handler.get_joint_reindex(self.robot_name)
        self.inverse_reorder_idx = [self.reorder_idx.index(i) for i in range(len(self.reorder_idx))]

    def _extract_robot_state(self, obs) -> torch.Tensor:
        self._ensure_reindex()
        rs = obs.robots[self.robot_name]
        joint_pos = rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)
        joint_pos = joint_pos.to(torch.float32)
        return joint_pos

    def _build_state_vector(self, joint_pos_alpha: torch.Tensor) -> np.ndarray:
        return joint_pos_alpha[0].cpu().numpy().astype(np.float32)

    def _get_prompt(self) -> str:
        task_env = getattr(self.env, "task_env", None)
        if task_env is not None and getattr(task_env, "task_desc", None):
            return str(task_env.task_desc)
        return "Execute the RoboVerse task."

    def _compress_image(self, obs) -> np.ndarray:
        cam = next(iter(obs.cameras.values()))
        rgb = cam.rgb
        rgb_np = rgb[0].detach().cpu().numpy() if rgb.dim() == 4 else rgb.detach().cpu().numpy()
        resized = image_tools.resize_with_pad(rgb_np, self.image_size, self.image_size)
        return image_tools.convert_to_uint8(resized)

    def _build_policy_observation(self, obs) -> Dict[str, Any]:
        img_uint8 = self._compress_image(obs)
        curr_robot_q = self._extract_robot_state(obs)
        state_vec = self._build_state_vector(curr_robot_q)
        prompt = self._get_prompt()
        fake_wrist = np.zeros_like(img_uint8)
        return {
            "observation/image": img_uint8,
            "observation/wrist_image": fake_wrist,
            "observation/state": state_vec,
            "prompt": prompt,
        }

    def _decode_single_action(self, action: np.ndarray) -> list[dict]:
        action = action.astype(np.float32)

        finger_vals = torch.tensor(action[:2], device=self.device)
        gripper_binary = torch.tensor(
            [1.0 if float(finger_vals.mean()) > self.gripper_threshold else 0.0],
            device=self.device,
        )
        gripper_widths = process_gripper_command(gripper_binary, self.robot_cfg, self.device)

        arm_target = torch.tensor(action[2:], device=self.device).unsqueeze(0)
        joint_target = torch.cat([arm_target, gripper_widths], dim=-1)

        joint_names = list(self.robot_cfg.joint_limits.keys())
        assert len(joint_names) == joint_target.shape[1], \
            f"Joint count mismatch: {len(joint_names)} names vs {joint_target.shape[1]} targets"
        dof_pos_target = {
            joint_name: float(joint_target[0, i].item())
            for i, joint_name in enumerate(joint_names)
        }

        actions = [
            {
                self.robot_name: {
                    "dof_pos_target": dof_pos_target
                }
            }
        ]
        return actions


    def _request_action_chunk(self, policy_obs: Dict[str, Any]) -> None:
        response = self.client.infer(policy_obs)
        chunk = np.asarray(response["actions"], dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with ndim=2, got {chunk.shape}")

        self.cached_actions = chunk
        self.cache_index = 0
        total = len(chunk)
        self.cache_remaining = total if self.actions_per_call is None else min(self.actions_per_call, total)

    def infer_action(self, obs) -> list[dict]:
        current_q = self._extract_robot_state(obs)

        if (
            self.cached_actions is None
            or self.cache_remaining <= 0
            or self.cache_index >= len(self.cached_actions)
        ):
            policy_obs = self._build_policy_observation(obs)
            self._request_action_chunk(policy_obs)

        action_vec = self.cached_actions[self.cache_index]
        self.cache_index += 1
        self.cache_remaining -= 1
        return self._decode_single_action(action_vec)

    def reset(self) -> None:
        self.cached_actions = None
        self.cache_index = 0
        self.cache_remaining = 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def evaluate_episode(
    env,
    runner: PiPolicyRunner,
    max_steps: int,
    episode: int,
    output_dir: str,
    randomization_manager=None,
    demo_idx: int = 0,
    init_states=None,
) -> Dict[str, Any]:
    """Evaluate a single episode."""

    # Apply domain randomization before reset
    if randomization_manager is not None:
        randomization_manager.apply_randomization(demo_idx=demo_idx, is_initial=(episode == 1))
        randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=0)
        randomization_manager.update_camera_look_at(env_id=0)
        randomization_manager.apply_camera_randomization()

    # Reset environment
    if randomization_manager is not None and init_states is not None:
        from roboverse_learn.il.act.act_eval_runner import ensure_clean_state
        # Use task_env.reset() directly to pass states parameter
        obs, info = env.task_env.reset(states=[init_states[demo_idx]])
        ensure_clean_state(env.task_env.handler, expected_state=init_states[demo_idx])
        if hasattr(env, "_episode_steps"):
            env._episode_steps[0] = 0
    else:
        obs, info = env.reset()

    runner.reset()

    stats: Dict[str, Any] = {
        "steps": 0,
        "success": False,
        "total_reward": 0.0,
        "start_time": time.time(),
    }

    os.makedirs(output_dir, exist_ok=True)
    obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode:03d}.mp4")
    obs_saver.add(obs)

    for _ in range(max_steps):
        actions = runner.infer_action(obs)
        obs, reward, terminated, truncated, info = env.step(actions)

        stats["steps"] += 1
        stats["total_reward"] += float(reward.mean().item())
        obs_saver.add(obs)

        term = terminated.any().item() if hasattr(terminated, "any") else bool(terminated)
        trunc = truncated.any().item() if hasattr(truncated, "any") else bool(truncated)
        if term or trunc:
            stats["success"] = True
            break

    obs_saver.save()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate π policies via websocket server")
    parser.add_argument("--task", type=str, default="pick_butter")
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--sim", type=str, default="mujoco",
                        choices=["isaacgym", "isaacsim", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"])
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--solver", type=str, default="pyroki", choices=["curobo", "pyroki"],
                        help="IK backend used for composing joint commands")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./pi_eval_output")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gripper-threshold", type=float, default=0.02,
                        help="Threshold on finger joint values to treat the gripper as open")
    parser.add_argument("--actions-per-call", type=int, default=0,
                        help="Number of cached actions to use before requesting a new chunk (0 = consume entire chunk)")
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
    return parser.parse_args()


def main() -> bool:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Pi Evaluation")
    print(f"  Task: {args.task}")
    print(f"  Robot: {args.robot}")
    print(f"  Simulator: {args.sim}")
    print(f"  DR Level: {args.level}, Scene Mode: {args.scene_mode}, Seed: {args.randomization_seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
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

    env = make_vec(
        f"RoboVerse/{args.task}",
        num_envs=args.num_envs,
        robots=[args.robot],
        simulator=args.sim,
        headless=True,
        cameras=[camera],
        lights=lights,
        device=args.device,
    )

    runner = PiPolicyRunner(
        env=env,
        scenario=env.scenario,
        num_envs=args.num_envs,
        robot_name=args.robot,
        solver=args.solver,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        image_size=args.image_size,
        gripper_threshold=args.gripper_threshold,
        actions_per_call=args.actions_per_call,
        device=args.device,
    )

    # Load trajectories for DR
    traj_filepath = env.task_env.traj_filepath
    robot_obj = get_robot(args.robot)
    init_states, _, _ = get_traj(traj_filepath, robot_obj, env.task_env.handler)

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
            scenario=env.scenario,
            handler=env.task_env.handler,
            init_states=init_states,
            render_cfg=SimpleRenderCfg(mode=render_mode)
        )
        print(f"Domain Randomization enabled: level={args.level}, scene_mode={args.scene_mode}, seed={args.randomization_seed}")

    start_time = time.time()
    aggregate = {
        "total_episodes": 0,
        "total_successes": 0,
        "total_rewards": [],
        "episode_results": [],
    }

    for ep in range(args.num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'=' * 50}")

        demo_idx = ep % len(init_states) if randomization_manager is not None else 0

        result = evaluate_episode(
            env, runner, args.max_steps, ep + 1, args.output_dir,
            randomization_manager=randomization_manager,
            demo_idx=demo_idx,
            init_states=init_states if randomization_manager is not None else None
        )

        aggregate["total_episodes"] += 1
        aggregate["episode_results"].append(result)
        aggregate["total_rewards"].append(result["total_reward"])
        if result["success"]:
            aggregate["total_successes"] += 1

        sr = aggregate["total_successes"] / aggregate["total_episodes"]
        print(f"  Steps: {result['steps']}")
        print(f"  Success: {result['success']}")
        print(f"  Reward: {result['total_reward']:.2f}")
        print(f"  Success rate: {sr:.1%}")

    total_time = time.time() - start_time
    final_sr = aggregate["total_successes"] / max(1, aggregate["total_episodes"])
    final_reward = float(np.mean(aggregate["total_rewards"])) if aggregate["total_rewards"] else 0.0

    print(f"\n{'=' * 50}")
    print("Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"Success Rate: {final_sr:.1%} ({aggregate['total_successes']}/{aggregate['total_episodes']})")
    print(f"Average Reward: {final_reward:.2f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"DR Level: {args.level}, Scene Mode: {args.scene_mode}, Seed: {args.randomization_seed}")
    print(f"{'=' * 50}")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output_dir, f"pi_eval_{args.task}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "stats": aggregate,
            "timestamp": ts,
            "dr_config": {"level": args.level, "scene_mode": args.scene_mode, "seed": args.randomization_seed}
        }, f, indent=2)
    print(f"Saved results to {report_path}")

    try:
        env.close()
    except Exception:
        pass
    runner.close()
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
