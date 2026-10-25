"""Sharpa Wave: a 22-DoF five-fingered tactile hand (Menagerie ``sharpa_wave`` MJCF), left and right."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass
from metasim.utils.hf_util import register_upstream_mirror

MENAGERIE_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
"""MuJoCo Menagerie revision the configs below were written against (``sharpa_wave`` CHANGELOG 2026-08-03)."""

# The asset is published by Menagerie; until it is mirrored into the roboverse_data dataset a checkout fetches it
# from the pinned upstream revision into the same ``roboverse_data`` location.
register_upstream_mirror(
    "robots/sharpa_wave/mjcf",
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_COMMIT}/sharpa_wave",
)

# Per-joint data of the Menagerie model, keyed by the joint name without its ``left_`` / ``right_`` prefix:
# (lower limit, upper limit, kp, kv, actuator force range). kp is the asset's ``gainprm``; kv is the damping the
# asset's ``dampratio`` compiles to (MuJoCo >= 3.1.6), written out so Newton drives the hand with the same
# servo law; the force range is the joint's ``actuatorfrcrange``. The right hand is a mirror of the left with the
# same joint coordinates, so one table serves both.
_JOINTS: dict[str, tuple[float, float, float, float, float]] = {
    "thumb_CMC_FE": (-0.1745, 1.9199, 6.95, 0.332, 3.3),
    "thumb_CMC_AA": (-0.3491, 0.3491, 13.2, 0.468, 3.3),
    "thumb_MCP_FE": (-0.5236, 1.3963, 4.76, 0.143, 1.864),
    "thumb_MCP_AA": (-0.3491, 0.3491, 6.62, 0.169, 1.864),
    "thumb_IP": (0.0, 1.7453, 0.9, 0.035, 0.638),
    "index_MCP_FE": (-0.174533, 1.5708, 4.76, 0.149, 1.864),
    "index_MCP_AA": (-0.3491, 0.3491, 6.62, 0.176, 1.864),
    "index_PIP": (0.0, 1.7453, 0.9, 0.035, 0.638),
    "index_DIP": (0.0, 1.3963, 0.9, 0.009, 0.189369),
    "middle_MCP_FE": (-0.174533, 1.5708, 4.76, 0.149, 1.864),
    "middle_MCP_AA": (-0.3491, 0.3491, 6.62, 0.176, 1.864),
    "middle_PIP": (0.0, 1.7453, 0.9, 0.035, 0.638),
    "middle_DIP": (0.0, 1.3963, 0.9, 0.009, 0.189369),
    "ring_MCP_FE": (-0.174533, 1.5708, 4.76, 0.149, 1.864),
    "ring_MCP_AA": (-0.3491, 0.3491, 6.62, 0.176, 1.864),
    "ring_PIP": (0.0, 1.7453, 0.9, 0.035, 0.638),
    "ring_DIP": (0.0, 1.3963, 0.9, 0.009, 0.189369),
    "pinky_CMC": (0.0, 0.2618, 1.38, 0.029, 0.5285),
    "pinky_MCP_FE": (-0.174533, 1.5708, 4.76, 0.149, 1.864),
    "pinky_MCP_AA": (-0.3491, 0.3491, 6.62, 0.176, 1.864),
    "pinky_PIP": (0.0, 1.7453, 0.9, 0.035, 0.638),
    "pinky_DIP": (0.0, 1.3963, 0.9, 0.009, 0.189369),
}

FINGERTIP_SITES = ("thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "pinky_fingertip")
"""Fingertip site names (prefix them with the hand's side), the tactile pads' centres."""


def _actuators(side: str) -> dict[str, BaseActuatorCfg]:
    return {
        f"{side}_{joint}": BaseActuatorCfg(stiffness=kp, damping=kv, effort_limit_sim=force)
        for joint, (_, _, kp, kv, force) in _JOINTS.items()
    }


def _joint_limits(side: str) -> dict[str, tuple[float, float]]:
    return {f"{side}_{joint}": (lo, hi) for joint, (lo, hi, _, _, _) in _JOINTS.items()}


@configclass
class SharpaWaveRightCfg(RobotCfg):
    """Sharpa Wave right hand: thumb 5 DoF, index / middle / ring 4 DoF each, pinky 5 DoF, one servo per joint.

    Asset: ``roboverse_data/robots/sharpa_wave/mjcf/right_hand.xml`` (MuJoCo Menagerie ``sharpa_wave``,
    Apache-2.0, derived from the manufacturer's URDF; ``README.md``, ``LICENSE`` and ``UPSTREAM.txt`` ship
    next to it). The palm is the root body ``right_hand_C_MC`` with a ``wrist_site`` at its base for mounting
    on an arm; the tactile pads are the ``*_fingertip`` sites. Joints are zero with the hand open, which is the
    default pose. Needs MuJoCo >= 3.1.6.
    """

    name: str = "sharpa_wave_right"
    num_joints: int = 22
    fix_base_link: bool = True

    mjcf_path: str = "roboverse_data/robots/sharpa_wave/mjcf/right_hand.xml"
    # MJCF is the only description; Newton and Isaac Gym read it directly.
    file_type: dict[str, str] = {**RobotCfg.file_type, "newton": "mjcf", "isaacgym": "mjcf"}

    enabled_gravity: bool = True
    # The asset excludes the contact pairs that would jam a finger; the rest of the hand may self-collide.
    enabled_self_collisions: Literal["mujoco_default"] | bool = "mujoco_default"

    actuators: dict[str, BaseActuatorCfg] = _actuators("right")
    joint_limits: dict[str, tuple[float, float]] = _joint_limits("right")
    default_joint_positions: dict[str, float] = {f"right_{joint}": 0.0 for joint in _JOINTS}
    control_type: dict[str, Literal["position", "effort"]] = {f"right_{joint}": "position" for joint in _JOINTS}

    ee_body_name: str = "right_hand_C_MC"


@configclass
class SharpaWaveLeftCfg(RobotCfg):
    """Sharpa Wave left hand; see :class:`SharpaWaveRightCfg`."""

    name: str = "sharpa_wave_left"
    num_joints: int = 22
    fix_base_link: bool = True

    mjcf_path: str = "roboverse_data/robots/sharpa_wave/mjcf/left_hand.xml"
    # MJCF is the only description; Newton and Isaac Gym read it directly.
    file_type: dict[str, str] = {**RobotCfg.file_type, "newton": "mjcf", "isaacgym": "mjcf"}

    enabled_gravity: bool = True
    enabled_self_collisions: Literal["mujoco_default"] | bool = "mujoco_default"

    actuators: dict[str, BaseActuatorCfg] = _actuators("left")
    joint_limits: dict[str, tuple[float, float]] = _joint_limits("left")
    default_joint_positions: dict[str, float] = {f"left_{joint}": 0.0 for joint in _JOINTS}
    control_type: dict[str, Literal["position", "effort"]] = {f"left_{joint}": "position" for joint in _JOINTS}

    ee_body_name: str = "left_hand_C_MC"
