#!/usr/bin/env python3
"""Stage the bundled box_task assets into ``roboverse_data/`` and patch them
for the canonical naming the rest of RoboVerse expects.

Run once per checkout. The bundle (``box_task_replay_render_bundle_clean``)
is the upstream recording artifact, so we copy from there rather than
distribute the same files in two places.

What this script does:

1. Copies the OpenArm Wuji robot MJCF + mesh dirs into
   ``roboverse_data/robots/openarm_wuji/`` and patches the MJCF in place:
   - drops the IK-target ``mocap_left`` / ``mocap_right`` bodies (mujoco
     refuses to compile when they are nested under another body)
   - renames legacy hand joints / actuators ``{side}_hand_finger`` →
     ``{side}_finger`` to match ``OpenarmBimanualWujiCfg``
   - renames arm motor actuators ``{side}_joint{N}_ctrl`` →
     ``openarm_{side}_joint{N}`` so actuator names equal joint names
2. Copies the three object MJCFs alongside their USDs under
   ``roboverse_data/assets/box_task/local_pack_box/``.
3. Converts the legacy v2 trajectory pkl
   (``...task3_meshycup_openarm_wuji_..._v2.pkl``) into the canonical
   ``task3_openarm_bimanual_wuji_v2.pkl`` — same conversion the
   ``convert_box_task_legacy_traj.py`` script provides, invoked here so
   one command stages everything.

Usage::

    python scripts/prepare_box_task_assets.py \\
        --bundle ~/projects/RoboVerse/box_task_replay_render_bundle_clean
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path

OBJECT_NAMES = ("cardboard_box", "feast_soda_can", "feast_scented_candle")
LEGACY_TRAJ_NAME = "task3_meshycup_openarm_wuji_20260513_232823_0_v2.pkl"
CANONICAL_TRAJ_NAME = "task3_openarm_bimanual_wuji_v2.pkl"


def _stage_robot(bundle_root: Path, dest_root: Path) -> Path:
    """Copy openarm_wuji robot files + meshes; return the staged MJCF path."""
    src = bundle_root / "assets" / "source" / "box_task" / "assets"
    dst = dest_root / "robots" / "openarm_wuji"
    dst.mkdir(parents=True, exist_ok=True)

    for sub in ("openarm", "wuji_hand", "meshes"):
        src_sub = src / sub
        if src_sub.is_dir():
            shutil.copytree(src_sub, dst / sub, dirs_exist_ok=True)

    src_mjcf = src / "openarm_wuji.xml"
    dst_mjcf = dst / "openarm_wuji.xml"
    shutil.copy2(src_mjcf, dst_mjcf)
    return dst_mjcf


def _patch_robot_mjcf(mjcf_path: Path) -> None:
    """Strip mocap bodies; rename joints + actuators to cfg convention."""
    text = mjcf_path.read_text()

    # Drop the two single-line mocap bodies (visualization-only IK targets).
    # They must be world children in MJCF, which breaks when MetaSim
    # attaches the robot under another parent.
    text = re.sub(
        r'    <body name="mocap_(left|right)" mocap="true"[^/]*>\s*\n[^\n]*\n    </body>\n',
        "",
        text,
    )

    # ``{side}_hand_finger{i}_joint{j}`` → ``{side}_finger{i}_joint{j}``
    # (both the joint definitions and the position-actuator names refer to it).
    text = re.sub(r"(left|right)_hand_finger(\d)_joint(\d)", r"\1_finger\2_joint\3", text)

    # Arm motor names ``{side}_joint{N}_ctrl`` → ``openarm_{side}_joint{N}``
    # so actuator-name and joint-name agree (the cfg expects them to match).
    text = re.sub(
        r'<motor name="(left|right)_joint(\d)_ctrl"',
        lambda m: f'<motor name="openarm_{m.group(1)}_joint{m.group(2)}"',
        text,
    )

    mjcf_path.write_text(text)


def _stage_object_mjcfs(bundle_root: Path, dest_root: Path) -> list[Path]:
    """Copy each object's MJCF into local_pack_box/<name>/ alongside its USD."""
    src_dir = bundle_root / "assets" / "source" / "box_task" / "assets" / "scenes"
    out: list[Path] = []
    for name in OBJECT_NAMES:
        src = src_dir / name / f"{name}.xml"
        dst = dest_root / "assets" / "box_task" / "local_pack_box" / name / f"{name}.xml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
            out.append(dst)
    return out


def _convert_trajectory(bundle_root: Path, dest_root: Path, repo_root: Path) -> Path:
    """Run the v2 converter, returning the staged canonical pkl path."""
    src = bundle_root / "assets" / "traj" / LEGACY_TRAJ_NAME
    dst = dest_root / "trajs" / "box_task" / CANONICAL_TRAJ_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)

    converter_path = repo_root / "scripts" / "convert_box_task_legacy_traj.py"
    spec = importlib.util.spec_from_file_location("convert_box_task", converter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.convert(src, dst)
    return dst


def prepare(bundle_root: Path, repo_root: Path) -> None:
    dest_root = repo_root / "roboverse_data"
    robot_mjcf = _stage_robot(bundle_root, dest_root)
    _patch_robot_mjcf(robot_mjcf)
    obj_mjcfs = _stage_object_mjcfs(bundle_root, dest_root)
    traj_path = _convert_trajectory(bundle_root, dest_root, repo_root)

    print(f"robot MJCF:        {robot_mjcf}")
    for o in obj_mjcfs:
        print(f"object MJCF:       {o}")
    print(f"converted traj:    {traj_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to box_task_replay_render_bundle_clean (the upstream recording).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RoboVerse repo root (defaults to the repo that contains this script).",
    )
    args = parser.parse_args()
    prepare(args.bundle.resolve(), args.repo_root.resolve())


if __name__ == "__main__":
    main()
