#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


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
        description="Tiny-training smoke for BeyondMimic motion-tracking-isaaclab using the canonical RSL-RL pipeline."
    )
    parser.add_argument("--sim", default="isaacgym", choices=["isaacsim", "isaacgym", "newton", "mujoco"])
    parser.add_argument("--robot", default="g1_tracking")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="Torch device string (e.g., cpu, cuda:0).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--num-steps-per-env", type=int, default=4)
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
        "--out-dir",
        default="/tmp/roboverse_smoke",
        help="Output directory for logs/models (kept out of the repo workspace by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure repo root is importable when executing via `python scripts/...`.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Headless is required in this environment.
    headless = True

    if args.sim == "mujoco":
        # Multi-env MuJoCo uses ParallelSimWrapper (multiprocessing).
        # Optional queries (e.g., contact force history) are propagated to workers by the compat layer.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = args.mujoco_gl

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
        if args.robot != "g1_tracking":
            raise ValueError(
                f"Dummy motion generation only supports --robot g1_tracking, got '{args.robot}'. "
                "Pass --motion-file to use a real motion."
            )
        joint_names, body_names = _load_g1_joint_and_body_names(repo_root)
        _write_dummy_motion_file(motion_path, frames=8, joint_names=joint_names, body_names=body_names, fps=50.0)

    # Import the training entrypoint before importing configs (which import wandb).
    # In the IsaacGym environment, wandb may import PyTorch, and IsaacGym requires importing
    # its native modules before PyTorch.
    from roboverse_learn.rl.configs.rsl_rl.ppo_tracking import RslRlPPOTrackingConfig
    from roboverse_learn.rl.rsl_rl.ppo_tracking import train

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = str(Path(args.out_dir) / f"motion-tracking-isaaclab/{args.sim}/{timestamp}")

    use_cuda = str(args.device).startswith("cuda")
    cfg = RslRlPPOTrackingConfig(
        task="motion-tracking-isaaclab",
        robot=str(args.robot),
        sim=str(args.sim),
        num_envs=int(args.num_envs),
        headless=bool(headless),
        cuda=bool(use_cuda),
        device=str(args.device),
        seed=int(args.seed),
        use_wandb=False,
        motion_file=str(motion_path),
        max_iterations=int(args.max_iterations),
        num_steps_per_env=int(args.num_steps_per_env),
        model_dir=model_dir,
    )

    train(cfg)


if __name__ == "__main__":
    main()
