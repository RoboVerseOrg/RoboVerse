"""I2RT YAM: a 6-DoF arm with a parallel-jaw gripper (Menagerie ``i2rt_yam`` MJCF)."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

_ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# Servo gains and force limits as the Menagerie MJCF authors them (``dm4340`` / ``dm4310`` motor classes,
# joint4 overridden, ``finger`` class), so every backend drives the robot the way the asset does.
_GAINS: dict[str, tuple[float, float, float | None]] = {  # joint -> (kp, kv, force limit)
    "joint1": (40.0, 2.5, 28.0),
    "joint2": (40.0, 2.5, 28.0),
    "joint3": (40.0, 2.5, 28.0),
    "joint4": (20.0, 0.5, 10.0),
    "joint5": (10.0, 1.0, 10.0),
    "joint6": (10.0, 1.0, 10.0),
    "left_finger": (100.0, 10.0, None),
}

GRIPPER_OPEN = 0.037524
"""``left_finger`` travel at which the jaws are fully open (the joint's upper limit)."""
GRIPPER_CLOSED = 0.0
"""``left_finger`` travel at which the jaws meet (the MJCF's home keyframe)."""


@configclass
class YamCfg(RobotCfg):
    """I2RT YAM arm with its parallel-jaw gripper.

    Asset: ``roboverse_data/robots/Yet_Another_Manipulator_YAM/mjcf`` (MuJoCo Menagerie ``i2rt_yam``, MIT,
    derived from the manufacturer's URDF; ``README.md`` and ``LICENSE`` ship next to the XML). The gripper
    has two slide joints coupled by an MJCF ``<equality>``: ``left_finger`` is driven and ``right_finger``
    mirrors it, so ``right_finger`` is a *passive* joint, present in the state and action vectors but
    never written as a target. The home pose is the asset's ``home`` keyframe (``joint2 = joint3 = 60°``,
    arm raised) with the gripper open. Gravity stays on: the asset compensates it per link
    (``gravcomp="1"``).

    The task-side ``mjlab_yam`` config (``MjlabYamCfg``) drives mjlab's bare copy of the same arm for the
    mjlab lift-cube ports; this is the general-purpose embodiment.
    """

    name: str = "yam"
    num_joints: int = 8
    fix_base_link: bool = True

    mjcf_path: str = "roboverse_data/robots/Yet_Another_Manipulator_YAM/mjcf/yam.xml"
    # MJCF is the only description; Newton and Isaac Gym read it directly.
    file_type: dict[str, str] = {**RobotCfg.file_type, "newton": "mjcf", "isaacgym": "mjcf"}

    enabled_gravity: bool = True
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        joint: BaseActuatorCfg(stiffness=kp, damping=kv, effort_limit_sim=limit, is_ee=joint == "left_finger")
        for joint, (kp, kv, limit) in _GAINS.items()
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.61799, 3.05433),
        "joint2": (0.0, 3.66519),
        "joint3": (0.0, 3.66519),
        "joint4": (-1.5708, 1.5708),
        "joint5": (-1.5708, 1.5708),
        "joint6": (-2.0944, 2.0944),
        "left_finger": (-0.00205, GRIPPER_OPEN),
        "right_finger": (-GRIPPER_OPEN, 0.00205),  # passive: mirrors left_finger through the equality constraint
    }

    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": 1.047,
        "joint3": 1.047,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
        "left_finger": GRIPPER_OPEN,
        "right_finger": -GRIPPER_OPEN,
    }

    control_type: dict[str, Literal["position", "effort"]] = {joint: "position" for joint in _GAINS}

    ee_body_name: str = "link_6"
    ee_joint_names: list[str] = ["left_finger"]
    gripper_open_q: list[float] = [GRIPPER_OPEN]
    gripper_close_q: list[float] = [GRIPPER_CLOSED]

    # The grasp point (the asset's ``grasp_site``) expressed in the ``link_6`` frame.
    curobo_tcp_rel_pos: tuple[float, float, float] = (0.0, 0.0, 0.1347)
    curobo_tcp_rel_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
