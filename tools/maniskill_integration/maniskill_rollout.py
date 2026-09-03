"""Drive a ManiSkill gym env and record state / reward / frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ManiskillRollout:
    """Time-series captured from one ManiSkill rollout."""

    gym_id: str
    seed: int
    obs_mode: str
    control_mode: str
    actions: np.ndarray  # (T, A)
    rewards: np.ndarray  # (T,)
    success: np.ndarray  # (T+1,) bool — taken from info each step
    is_grasped: np.ndarray  # (T+1,) bool
    is_obj_placed: np.ndarray  # (T+1,) bool
    is_robot_static: np.ndarray  # (T+1,) bool
    qpos: np.ndarray  # (T+1, nq) robot qpos
    qvel: np.ndarray  # (T+1, nq) robot qvel
    cube_pos: np.ndarray  # (T+1, 3)
    cube_quat: np.ndarray  # (T+1, 4) wxyz
    goal_pos: np.ndarray  # (T+1, 3)
    tcp_pos: np.ndarray  # (T+1, 3) end-effector pos
    elapsed: np.ndarray  # (T+1,)
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 20.0

    def summary(self) -> dict:
        return {
            "gym_id": self.gym_id,
            "seed": self.seed,
            "T": int(self.actions.shape[0]),
            "nq": int(self.qpos.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "total_reward": float(self.rewards.sum()),
            "max_reward": float(self.rewards.max()),
            "any_success": bool(self.success.any()),
            "final_success": bool(self.success[-1]),
            "final_obj_placed": bool(self.is_obj_placed[-1]),
            "final_robot_static": bool(self.is_robot_static[-1]),
            "tcp_to_cube_final": float(np.linalg.norm(self.cube_pos[-1] - self.tcp_pos[-1])),
            "cube_to_goal_final": float(np.linalg.norm(self.cube_pos[-1] - self.goal_pos[-1])),
        }


def _detach(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _vec3(x) -> np.ndarray:
    arr = _detach(x).squeeze()
    return arr.reshape(3) if arr.size >= 3 else np.zeros(3)


def _vec4(x) -> np.ndarray:
    arr = _detach(x).squeeze()
    return arr.reshape(4) if arr.size >= 4 else np.array([1.0, 0.0, 0.0, 0.0])


def rollout_maniskill(
    gym_id: str,
    *,
    n_steps: int,
    seed: int = 0,
    control_mode: str = "pd_joint_delta_pos",
    obs_mode: str = "state",
    action_fn: Callable[[int, np.ndarray], np.ndarray] | None = None,
    render: bool = True,
    render_size: tuple[int, int] = (320, 240),
) -> ManiskillRollout:
    """Roll out one episode of a ManiSkill env and capture full state.

    Returns a `ManiskillRollout` ready for diffing against a MetaSim run.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers tasks

    env = gym.make(
        gym_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array" if render else None,
        max_episode_steps=n_steps,
        num_envs=1,
        sensor_configs={"width": render_size[0], "height": render_size[1]} if render else None,
        human_render_camera_configs={"width": render_size[0], "height": render_size[1]} if render else None,
    )
    obs, info = env.reset(seed=seed)
    unwrapped = env.unwrapped
    action_dim = int(env.action_space.shape[-1] if env.action_space.shape else 0)

    def _zero(t, q):
        return np.zeros(action_dim, dtype=np.float32)

    fn = action_fn or _zero

    def _primary_obj(u):
        # Tasks expose the main movable as one of these attrs depending on task.
        for name in ("cube", "obj", "cubeA"):
            v = getattr(u, name, None)
            if v is not None:
                return v
        return None

    def _goal_anchor(u):
        for name in ("goal_site", "goal_region"):
            v = getattr(u, name, None)
            if v is not None:
                return v
        # StackCube target is cubeB (place cubeA on cubeB)
        return getattr(u, "cubeB", None)

    obj = _primary_obj(unwrapped)
    goal = _goal_anchor(unwrapped)

    qpos = [_detach(unwrapped.agent.robot.qpos).reshape(-1).copy()]
    qvel = [_detach(unwrapped.agent.robot.qvel).reshape(-1).copy()]
    cube_pos = [_vec3(obj.pose.p) if obj is not None else np.zeros(3)]
    cube_quat = [_vec4(obj.pose.q) if obj is not None else np.array([1.0, 0.0, 0.0, 0.0])]
    goal_pos = [_vec3(goal.pose.p) if goal is not None else np.zeros(3)]
    tcp_pos = [_vec3(unwrapped.agent.tcp.pose.p)] if hasattr(unwrapped.agent, "tcp") else [np.zeros(3)]
    success_log = [bool(_detach(info.get("success", False)).any())] if isinstance(info, dict) else [False]
    grasp_log = [bool(_detach(info.get("is_grasped", False)).any())] if isinstance(info, dict) else [False]
    placed_log = [bool(_detach(info.get("is_obj_placed", False)).any())] if isinstance(info, dict) else [False]
    static_log = [bool(_detach(info.get("is_robot_static", False)).any())] if isinstance(info, dict) else [False]
    elapsed_log = [
        int(_detach(info.get("elapsed_steps", 0)).item() if isinstance(info.get("elapsed_steps"), object) else 0)
    ]
    frames: list[np.ndarray] = []
    if render:
        try:
            img = _detach(env.render()).squeeze()
            frames.append(img)
        except Exception:
            pass

    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for t in range(n_steps):
        a = fn(t, qpos[-1])
        a = np.asarray(a, dtype=np.float32).reshape(action_dim)
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(_detach(r).reshape(-1)[0]))
        qpos.append(_detach(unwrapped.agent.robot.qpos).reshape(-1).copy())
        qvel.append(_detach(unwrapped.agent.robot.qvel).reshape(-1).copy())
        cube_pos.append(_vec3(obj.pose.p) if obj is not None else np.zeros(3))
        cube_quat.append(_vec4(obj.pose.q) if obj is not None else np.array([1.0, 0.0, 0.0, 0.0]))
        goal_pos.append(_vec3(goal.pose.p) if goal is not None else np.zeros(3))
        tcp_pos.append(_vec3(unwrapped.agent.tcp.pose.p) if hasattr(unwrapped.agent, "tcp") else np.zeros(3))
        success_log.append(bool(_detach(info.get("success", False)).any()))
        grasp_log.append(bool(_detach(info.get("is_grasped", False)).any()))
        placed_log.append(bool(_detach(info.get("is_obj_placed", False)).any()))
        static_log.append(bool(_detach(info.get("is_robot_static", False)).any()))
        elapsed_log.append(int(_detach(info.get("elapsed_steps", t + 1)).reshape(-1)[0]))
        if render:
            try:
                img = _detach(env.render()).squeeze()
                frames.append(img)
            except Exception:
                pass
        if (term if isinstance(term, bool) else bool(_detach(term).any())) or (
            trunc if isinstance(trunc, bool) else bool(_detach(trunc).any())
        ):
            break

    env.close()

    return ManiskillRollout(
        gym_id=gym_id,
        seed=seed,
        obs_mode=obs_mode,
        control_mode=control_mode,
        actions=np.stack(actions) if actions else np.zeros((0, action_dim), dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        success=np.array(success_log),
        is_grasped=np.array(grasp_log),
        is_obj_placed=np.array(placed_log),
        is_robot_static=np.array(static_log),
        qpos=np.stack(qpos),
        qvel=np.stack(qvel),
        cube_pos=np.stack(cube_pos),
        cube_quat=np.stack(cube_quat),
        goal_pos=np.stack(goal_pos),
        tcp_pos=np.stack(tcp_pos),
        elapsed=np.array(elapsed_log),
        frames=frames,
        fps=20.0,
    )


def save_rollout(path: Path, r: ManiskillRollout, save_frames: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gym_id=r.gym_id,
        seed=r.seed,
        obs_mode=r.obs_mode,
        control_mode=r.control_mode,
        actions=r.actions,
        rewards=r.rewards,
        success=r.success,
        is_grasped=r.is_grasped,
        is_obj_placed=r.is_obj_placed,
        is_robot_static=r.is_robot_static,
        qpos=r.qpos,
        qvel=r.qvel,
        cube_pos=r.cube_pos,
        cube_quat=r.cube_quat,
        goal_pos=r.goal_pos,
        tcp_pos=r.tcp_pos,
        elapsed=r.elapsed,
    )
    if save_frames and r.frames:
        np.savez_compressed(path.with_suffix(".frames.npz"), frames=np.stack(r.frames))
