from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENARM_ROOT = REPO_ROOT / "roboverse_data" / "robots" / "openarm_wuji"
ROBOT_DESCRIPTION_ROOT = REPO_ROOT / "roboverse_data" / "robots"
MJCF_PATH = OPENARM_ROOT / "openarm_wuji.xml"
URDF_PATH = OPENARM_ROOT / "openarm_wuji.urdf"
USD_PATH = OPENARM_ROOT / "openarm_wuji.usd"
LINK7_VISUAL_MESH = ROBOT_DESCRIPTION_ROOT / "openarm_description" / "meshes" / "arm" / "v10" / "visual" / "link7.stl"
LEFT_PALM_MESH = ROBOT_DESCRIPTION_ROOT / "wuji_hand_description" / "meshes" / "left" / "left_palm_link.STL"
RIGHT_PALM_MESH = ROBOT_DESCRIPTION_ROOT / "wuji_hand_description" / "meshes" / "right" / "right_palm_link.STL"

MAX_WRIST_PALM_VISUAL_OVERLAP_M = 0.001
MAX_WRIST_PALM_VISUAL_GAP_M = 0.001
WUJI_MOUNT_Z_M = 0.105


def _binary_stl_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size > len(data):
        raise ValueError(f"{path} has an invalid binary STL triangle count")

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    offset = 84
    for _ in range(triangle_count):
        vertices = struct.unpack_from("<9f", data, offset + 12)
        for index in range(0, 9, 3):
            for axis in range(3):
                value = vertices[index + axis]
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
        offset += 50
    return tuple(mins), tuple(maxs)


def _vector_attr(element: ET.Element, name: str) -> tuple[float, float, float]:
    return tuple(float(value) for value in element.attrib[name].split())  # type: ignore[return-value]


def _mjcf_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise AssertionError(f"Missing MJCF body {name!r}")


def _mjcf_geom(root: ET.Element, name: str) -> ET.Element:
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    raise AssertionError(f"Missing MJCF geom {name!r}")


def _urdf_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == name:
            return joint
    raise AssertionError(f"Missing URDF joint {name!r}")


def _mesh_paths_from_urdf(path: Path) -> list[Path]:
    root = ET.parse(path).getroot()
    return [
        (path.parent / mesh.attrib["filename"]).resolve() for mesh in root.iter("mesh") if mesh.attrib.get("filename")
    ]


def _mesh_paths_from_mjcf(path: Path) -> list[Path]:
    root = ET.parse(path).getroot()
    return [(path.parent / mesh.attrib["file"]).resolve() for mesh in root.iter("mesh") if mesh.attrib.get("file")]


@pytest.mark.parametrize(
    ("asset_file", "mesh_path_reader"),
    (
        (URDF_PATH, _mesh_paths_from_urdf),
        (MJCF_PATH, _mesh_paths_from_mjcf),
    ),
)
def test_openarm_wuji_mesh_paths_resolve_from_robot_descriptions(
    asset_file: Path, mesh_path_reader: Callable[[Path], list[Path]]
) -> None:
    mesh_paths = mesh_path_reader(asset_file)

    assert mesh_paths
    assert all(path.is_file() for path in mesh_paths)
    assert all(ROBOT_DESCRIPTION_ROOT in path.parents for path in mesh_paths)


def test_openarm_wuji_cfg_defaults_use_renamed_asset_directory() -> None:
    old_module_path = REPO_ROOT / "roboverse_pack" / "robots" / "openarm_bimanual_wuji_cfg.py"
    new_module_path = REPO_ROOT / "roboverse_pack" / "robots" / "openarm_wuji_cfg.py"
    assert not old_module_path.exists()
    assert new_module_path.is_file()

    from roboverse_pack.robots.openarm_wuji_cfg import OpenarmBimanualWujiCfg

    cfg = OpenarmBimanualWujiCfg()

    expected_paths = {
        "urdf_path": OPENARM_ROOT / "openarm_wuji.urdf",
        "mjcf_path": OPENARM_ROOT / "openarm_wuji.xml",
        "usd_path": OPENARM_ROOT / "openarm_wuji.usd",
    }
    for attr, expected_path in expected_paths.items():
        actual_path = (REPO_ROOT / getattr(cfg, attr)).resolve()
        assert actual_path == expected_path
        assert actual_path.is_file()


@pytest.mark.parametrize(
    ("side", "palm_mesh"),
    (
        ("left", LEFT_PALM_MESH),
        ("right", RIGHT_PALM_MESH),
    ),
)
def test_openarm_wuji_palm_mount_sits_flush_with_wrist_visual(side: str, palm_mesh: Path) -> None:
    mjcf_root = ET.parse(MJCF_PATH).getroot()
    urdf_root = ET.parse(URDF_PATH).getroot()

    link7_mesh_bounds = _binary_stl_bounds(LINK7_VISUAL_MESH)
    palm_mesh_bounds = _binary_stl_bounds(palm_mesh)

    link7_visual = _mjcf_geom(mjcf_root, f"openarm_{side}_link7_visual_0")
    link7_visual_pos = _vector_attr(link7_visual, "pos")
    link7_visual_scale_z = 0.001
    link7_tip_z = link7_mesh_bounds[1][2] * link7_visual_scale_z + link7_visual_pos[2]

    mjcf_mount = _mjcf_body(mjcf_root, f"{side}_hand_palm_link")
    mjcf_mount_z = _vector_attr(mjcf_mount, "pos")[2]

    urdf_mount = _urdf_joint(urdf_root, f"openarm_{side}_wuji_mount")
    urdf_mount_origin = urdf_mount.find("origin")
    assert urdf_mount_origin is not None
    assert _vector_attr(urdf_mount_origin, "xyz")[2] == pytest.approx(mjcf_mount_z)
    assert mjcf_mount_z == pytest.approx(WUJI_MOUNT_Z_M)

    palm_base_z = mjcf_mount_z + palm_mesh_bounds[0][2]
    overlap = max(0.0, link7_tip_z - palm_base_z)
    gap = max(0.0, palm_base_z - link7_tip_z)

    assert overlap <= MAX_WRIST_PALM_VISUAL_OVERLAP_M
    assert gap <= MAX_WRIST_PALM_VISUAL_GAP_M


def test_openarm_wuji_usd_mount_matches_mjcf_and_urdf() -> None:
    Usd = pytest.importorskip("pxr.Usd")

    stage = Usd.Stage.Open(USD_PATH.as_posix())
    assert stage is not None

    for side in ("left", "right"):
        joint = stage.GetPrimAtPath(f"/openarm_bimanual_wuji/joints/openarm_{side}_wuji_mount")
        assert joint
        assert tuple(joint.GetAttribute("physics:localPos0").Get()) == pytest.approx((0.0, 0.0, WUJI_MOUNT_Z_M))

        link7 = stage.GetPrimAtPath(f"/openarm_bimanual_wuji/openarm_{side}_link7")
        palm = stage.GetPrimAtPath(f"/openarm_bimanual_wuji/{side}_palm_link")
        assert link7
        assert palm
        link7_z = link7.GetAttribute("xformOp:translate").Get()[2]
        palm_z = palm.GetAttribute("xformOp:translate").Get()[2]
        assert link7_z - palm_z == pytest.approx(WUJI_MOUNT_Z_M)
