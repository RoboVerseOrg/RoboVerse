from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

OPENARM_ASSET_ROOT = Path(os.environ.get("ROBOVERSE_OPENARM_ASSET_ROOT", "roboverse_data/robots/openarm_wuji"))
OPENARM_URDF_PATH = OPENARM_ASSET_ROOT / "openarm_wuji.urdf"
OPENARM_MJCF_PATH = OPENARM_ASSET_ROOT / "openarm_wuji.xml"
OPENARM_USD_PATH = OPENARM_ASSET_ROOT / "openarm_wuji.usd"

OPENARM_JOINT_NAMES: tuple[str, ...] = tuple(
    [f"openarm_left_joint{i}" for i in range(1, 8)]
    + [f"openarm_right_joint{i}" for i in range(1, 8)]
    + [f"left_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
    + [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
)

LEFT_ARM_DEFAULT_Q = (-0.4, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0)
RIGHT_ARM_DEFAULT_Q = (0.4, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _JointMetadata:
    lower: float
    upper: float
    effort: float
    velocity: float


def _is_hand_joint(joint_name: str) -> bool:
    return joint_name.startswith(("left_finger", "right_finger"))


def _fallback_joint_metadata() -> dict[str, _JointMetadata]:
    metadata: dict[str, _JointMetadata] = {}
    for joint_name in OPENARM_JOINT_NAMES:
        if _is_hand_joint(joint_name):
            metadata[joint_name] = _JointMetadata(lower=-0.5, upper=1.6, effort=1.0, velocity=10.0)
        else:
            metadata[joint_name] = _JointMetadata(lower=-3.5, upper=3.5, effort=40.0, velocity=8.0)
    return metadata


def _parse_urdf_joint_metadata(urdf_path: Path) -> dict[str, _JointMetadata]:
    if not urdf_path.is_file():
        return _fallback_joint_metadata()

    parsed: dict[str, _JointMetadata] = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        if joint_name not in OPENARM_JOINT_NAMES or joint.attrib.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        parsed[joint_name] = _JointMetadata(
            lower=float(limit.attrib["lower"]),
            upper=float(limit.attrib["upper"]),
            effort=float(limit.attrib.get("effort", 1.0)),
            velocity=float(limit.attrib.get("velocity", 1.0)),
        )

    fallback = _fallback_joint_metadata()
    return {joint_name: parsed.get(joint_name, fallback[joint_name]) for joint_name in OPENARM_JOINT_NAMES}


def _safe_default(metadata: _JointMetadata) -> float:
    if metadata.lower <= 0.0 <= metadata.upper:
        return 0.0
    return max(metadata.lower + 1.0e-4, min(metadata.upper - 1.0e-4, metadata.lower))


def _default_joint_positions(metadata: dict[str, _JointMetadata]) -> dict[str, float]:
    defaults = {joint_name: _safe_default(joint_metadata) for joint_name, joint_metadata in metadata.items()}
    for index, value in enumerate(LEFT_ARM_DEFAULT_Q, start=1):
        defaults[f"openarm_left_joint{index}"] = value
    for index, value in enumerate(RIGHT_ARM_DEFAULT_Q, start=1):
        defaults[f"openarm_right_joint{index}"] = value
    return {joint_name: defaults[joint_name] for joint_name in OPENARM_JOINT_NAMES}


def _actuator_cfg(joint_name: str, metadata: _JointMetadata) -> BaseActuatorCfg:
    if _is_hand_joint(joint_name):
        return BaseActuatorCfg(
            effort_limit_sim=3.0,
            velocity_limit=metadata.velocity,
            velocity_limit_sim=metadata.velocity,
            stiffness=20.0,
            damping=1.5,
        )
    return BaseActuatorCfg(
        effort_limit_sim=120.0,
        velocity_limit=metadata.velocity,
        velocity_limit_sim=metadata.velocity,
        stiffness=220.0,
        damping=40.0,
    )


_JOINT_METADATA = _parse_urdf_joint_metadata(OPENARM_URDF_PATH)


@configclass
class OpenarmBimanualWujiCfg(RobotCfg):
    """OpenArm bimanual robot with left and right Wuji dexterous hands."""

    name: str = "openarm_bimanual_wuji"
    num_joints: int = 54
    fix_base_link: bool = True
    urdf_path: str = OPENARM_URDF_PATH.as_posix()
    mjcf_path: str = OPENARM_MJCF_PATH.as_posix()
    usd_path: str = OPENARM_USD_PATH.as_posix()
    enabled_gravity: bool = False
    enabled_self_collisions: bool = True
    default_position: tuple[float, float, float] = (0.0, -1.1, -0.18)
    default_orientation: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
    joint_limits: dict[str, tuple[float, float]] = {
        joint_name: (metadata.lower, metadata.upper) for joint_name, metadata in _JOINT_METADATA.items()
    }
    default_joint_positions: dict[str, float] = _default_joint_positions(_JOINT_METADATA)
    control_type: dict[str, str] = {joint_name: "position" for joint_name in OPENARM_JOINT_NAMES}
    actuators: dict[str, BaseActuatorCfg] = {
        joint_name: _actuator_cfg(joint_name, metadata) for joint_name, metadata in _JOINT_METADATA.items()
    }
    joint_names: list[str] = list(OPENARM_JOINT_NAMES)


__all__ = ["OPENARM_JOINT_NAMES", "OpenarmBimanualWujiCfg"]
