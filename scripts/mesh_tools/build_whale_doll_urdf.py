#!/usr/bin/env python3
"""
从三维网格生成带凸分解碰撞体的 URDF。

默认输入为 `roboverse_pack/whale_doll.glb`，输出为同目录下的 `whale_doll.urdf`，
并在 `collision_meshes/` 子目录写出凸包 OBJ。

依赖：
  - trimesh (pip install trimesh)
  - numpy (pip install numpy)
  - coacd (pip install coacd)  若使用 `--mode convex`

示例：
  python scripts/mesh_tools/build_whale_doll_urdf.py \\
      --mesh roboverse_pack/whale_doll.glb \\
      --urdf roboverse_pack/whale_doll.urdf \\
      --mode convex
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from typing import Iterable, List

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - 直接提示用户安装
    print("numpy is required. Please install with `pip install numpy`.")
    sys.exit(1)

try:
    import trimesh
except ModuleNotFoundError:  # pragma: no cover
    print("trimesh is required. Please install with `pip install trimesh`.")
    sys.exit(1)

try:
    import coacd
except ModuleNotFoundError:  # pragma: no cover
    coacd = None


def _format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _format_inertia(matrix: np.ndarray) -> dict[str, str]:
    inertia = np.array(matrix, dtype=float)
    return {
        "ixx": f"{inertia[0, 0]:.8f}",
        "ixy": f"{inertia[0, 1]:.8f}",
        "ixz": f"{inertia[0, 2]:.8f}",
        "iyy": f"{inertia[1, 1]:.8f}",
        "iyz": f"{inertia[1, 2]:.8f}",
        "izz": f"{inertia[2, 2]:.8f}",
    }


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Mesh scene at {mesh_path} is empty.")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    return mesh


def compute_mass_properties(mesh: trimesh.Trimesh, target_mass: float) -> tuple[np.ndarray, np.ndarray]:
    props = mesh.mass_properties
    if props.volume <= 0 or not np.isfinite(props.mass):
        raise RuntimeError("Mesh mass properties unavailable; ensure the mesh is watertight.")
    scale = target_mass / props.mass
    inertia = np.array(props.inertia) * scale
    center = np.array(props.center_mass)
    return center, inertia


def convex_decompose(
    mesh: trimesh.Trimesh,
    *,
    threshold: float,
    max_convex_hulls: int,
    resolution: int,
    seed: int,
    preprocess_resolution: int,
    merge: bool,
    decimate: bool,
    max_ch_vertex: int,
) -> List[trimesh.Trimesh]:
    if coacd is None:
        raise ModuleNotFoundError("coacd is required for convex decomposition. Install with `pip install coacd`.")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hulls,
        preprocess_mode="auto",
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        merge=merge,
        decimate=decimate,
        max_ch_vertex=max_ch_vertex,
        seed=seed,
    )
    return [trimesh.Trimesh(vs, fs) for vs, fs in result]


def write_urdf(
    urdf_path: Path,
    *,
    robot_name: str,
    mass: float,
    inertia: np.ndarray,
    center_mass: np.ndarray,
    visual_mesh_rel: str,
    collision_mesh_rel_paths: List[str],
) -> None:
    robot = ET.Element("robot", name=robot_name)
    link = ET.SubElement(robot, "link", name=robot_name)

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_format_vec((0.0, 0.0, 0.0)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(inertial, "inertia", **_format_inertia(inertia))

    origin_xyz = _format_vec((-center_mass[0], -center_mass[1], -center_mass[2]))

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz=origin_xyz, rpy="0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=visual_mesh_rel)

    for rel_path in collision_mesh_rel_paths:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=origin_xyz, rpy="0 0 0")
        coll_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(coll_geom, "mesh", filename=rel_path)

    tree = ET.ElementTree(robot)
    try:
        ET.indent(tree, space="  ", level=0)
    except AttributeError:
        # Python < 3.9 兼容：无需缩进
        pass
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]  # 项目根目录
    default_mesh = default_root / "roboverse_pack" / "whale_doll.glb"
    default_urdf = default_mesh.with_suffix(".urdf")

    parser = argparse.ArgumentParser(description="Generate URDF for whale_doll mesh.")
    parser.add_argument("--mesh", type=Path, default=default_mesh, help="输入网格路径（glb/obj/stl 等）")
    parser.add_argument("--urdf", type=Path, default=default_urdf, help="输出 URDF 路径")
    parser.add_argument("--robot-name", type=str, default="whale_doll", help="URDF 中的 robot 名称")
    parser.add_argument("--mass", type=float, default=0.8, help="整体质量 (kg)")
    parser.add_argument(
        "--mode",
        choices=("convex", "triangle"),
        default="convex",
        help="碰撞体模式：convex 使用 CoACD 凸分解；triangle 使用原网格",
    )

    # CoACD 参数
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD 终止误差阈值 [0.01, 1]")
    parser.add_argument("--max-convex-hulls", type=int, default=10, help="最大凸包数量 (-1 表示不限制)")
    parser.add_argument("--resolution", type=int, default=2000, help="CoACD 表面采样分辨率")
    parser.add_argument("--preprocess-resolution", type=int, default=50, help="预处理分辨率")
    parser.add_argument("--no-merge", action="store_true", help="禁用 CoACD 的凸包合并")
    parser.add_argument("--decimate", action="store_true", help="启用凸包顶点简化")
    parser.add_argument("--max-ch-vertex", type=int, default=256, help="每个凸包最大顶点数 (decimate 有效)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")

    return parser.parse_args()


def main():
    args = parse_args()

    mesh_path = args.mesh.resolve()
    urdf_path = args.urdf.resolve()
    urdf_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path)
    center_mass, inertia = compute_mass_properties(mesh, args.mass)

    urdf_dir = urdf_path.parent
    visual_mesh_rel = os.path.relpath(mesh_path, urdf_dir)

    collision_paths: List[str]
    if args.mode == "triangle":
        collision_paths = [visual_mesh_rel.replace("\\", "/")]
    else:
        parts = convex_decompose(
            mesh,
            threshold=args.threshold,
            max_convex_hulls=args.max_convex_hulls,
            resolution=args.resolution,
            seed=args.seed,
            preprocess_resolution=args.preprocess_resolution,
            merge=not args.no_merge,
            decimate=args.decimate,
            max_ch_vertex=args.max_ch_vertex,
        )
        if not parts:
            raise RuntimeError("Convex decomposition returned no parts.")
        collision_dir = urdf_dir / "collision_meshes"
        collision_dir.mkdir(parents=True, exist_ok=True)
        collision_paths = []
        for idx, part in enumerate(parts):
            filename = f"{args.robot_name}_convex_{idx:02d}.obj"
            out_path = collision_dir / filename
            part.export(out_path)
            rel_path = os.path.relpath(out_path, urdf_dir).replace("\\", "/")
            collision_paths.append(rel_path)

    write_urdf(
        urdf_path,
        robot_name=args.robot_name,
        mass=args.mass,
        inertia=inertia,
        center_mass=center_mass,
        visual_mesh_rel=visual_mesh_rel.replace("\\", "/"),
        collision_mesh_rel_paths=collision_paths,
    )

    print(f"URDF 已生成：{urdf_path}")
    if args.mode == "convex":
        print("凸包碰撞体输出目录：", (urdf_dir / "collision_meshes"))


if __name__ == "__main__":
    main()
