#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _motion_file_has_names(path: Path) -> bool:
    try:
        data = np.load(str(path))
    except Exception:
        return False
    return "joint_names" in data.files and "body_names" in data.files


def _load_g1_joint_and_body_names(repo_root: Path) -> tuple[list[str], list[str]]:
    urdf_path = repo_root / "roboverse_data/unitree_description/urdf/g1/main.urdf"
    root = ET.parse(str(urdf_path)).getroot()

    joint_names: list[str] = []
    parents: set[str] = set()
    children: set[str] = set()
    body_children: list[str] = []

    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        name = joint.attrib.get("name")
        if name:
            joint_names.append(str(name))

        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and "link" in parent.attrib:
            parents.add(str(parent.attrib["link"]))
        if child is not None and "link" in child.attrib:
            link = str(child.attrib["link"])
            children.add(link)
            body_children.append(link)

    base_candidates = list(parents - children)
    if len(base_candidates) != 1:
        raise RuntimeError(f"Failed to infer base link from URDF '{urdf_path}': candidates={base_candidates}")

    body_names = [base_candidates[0]] + body_children
    return joint_names, body_names


def _write_dummy_motion_file(
    path: Path, *, frames: int, joint_names: list[str], body_names: list[str], fps: float = 50.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_joints = len(joint_names)
    num_bodies = len(body_names)
    body_quat_w = np.zeros((frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    np.savez(
        str(path),
        fps=np.array(float(fps), dtype=np.float32),
        joint_names=np.asarray(joint_names, dtype=np.str_),
        body_names=np.asarray(body_names, dtype=np.str_),
        joint_pos=np.zeros((frames, num_joints), dtype=np.float32),
        joint_vel=np.zeros((frames, num_joints), dtype=np.float32),
        body_pos_w=np.zeros((frames, num_bodies, 3), dtype=np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((frames, num_bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, num_bodies, 3), dtype=np.float32),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for BeyondMimic motion-tracking-isaaclab on MetaSim handlers."
    )
    parser.add_argument("--sim", default="isaacgym", choices=["isaacsim", "isaacgym", "newton", "mujoco"])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--device", default="cpu", help="Torch device string passed to make_task_env (e.g., cpu, cuda:0)."
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motion-file", default="", help="Path to motion npz. If empty, a tiny dummy file is generated."
    )
    parser.add_argument(
        "--mujoco-gl",
        default="egl",
        choices=["egl", "osmesa", "glfw"],
        help="If --sim mujoco: set MUJOCO_GL (only if not already set).",
    )
    parser.add_argument(
        "--gui", action="store_true", help="Disable headless mode (not recommended in this environment)."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure repo root is importable when executing via `python scripts/...`.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Headless by default: this environment cannot reliably create GUI/OpenGL contexts.
    headless = not bool(args.gui)

    # If requested, make MuJoCo use an offscreen backend.
    if args.sim == "mujoco" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    # Import the task module so it registers `motion-tracking-isaaclab`.
    import roboverse_pack.tasks.beyondmimic.isaaclab.envs.tracking_rl_env as tracking_rl_env
    from metasim.task.factory import make_task_env

    # Motion file (create a tiny synthetic file if one isn't provided).
    motion_path = Path(args.motion_file) if args.motion_file else Path("/tmp/motion_tracking_dummy.npz")
    if motion_path.exists() and not _motion_file_has_names(motion_path):
        if args.motion_file:
            raise ValueError(
                f"Motion file '{motion_path}' is missing `joint_names`/`body_names`. "
                "Regenerate it with `roboverse_pack/tasks/beyondmimic/scripts/csv_to_npz.py`."
            )
        motion_path.unlink()

    if not motion_path.exists():
        if args.motion_file:
            raise FileNotFoundError(str(motion_path))
        joint_names, body_names = _load_g1_joint_and_body_names(repo_root)
        _write_dummy_motion_file(motion_path, frames=8, joint_names=joint_names, body_names=body_names, fps=50.0)

    # The task's cfg factory expects `args.train_cfg['seed']`, `args.device`, `args.motion_file`.
    task_args = SimpleNamespace(
        train_cfg={"seed": int(args.seed)}, device=str(args.device), motion_file=str(motion_path)
    )

    scenario = copy.deepcopy(tracking_rl_env.TrackingRLEnv.scenario)
    scenario.simulator = str(args.sim)
    scenario.num_envs = int(args.num_envs)
    scenario.headless = bool(headless)

    env = make_task_env("motion-tracking-isaaclab", scenario=scenario, args=task_args, device=str(args.device))

    obs, info = env.reset()
    print("reset_ok", {k: tuple(v.shape) for k, v in obs.items()})

    for i in range(int(args.steps)):
        act = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
        obs, rew, terminated, time_outs, info = env.step(act)
        done = bool(torch.any(terminated | time_outs).item())
        print(f"step={i} reward_mean={float(rew.mean().item()):.6f} done={done}")

    env.close()
    print("smoke_ok")


if __name__ == "__main__":
    main()
