from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class FrankaCfg(RobotCfg):
    """Cfg for the Franka Emika Panda robot.

    Args:
        RobotCfg (_type_): _description_
    """

    name: str = "franka"
    num_joints: int = 9
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/franka/usd/franka_v2.usd"
    mjcf_path: str = "roboverse_data/robots/franka/mjcf/panda.xml"
    # mjx_panda.xml was shipped as a WIP stub (missing the entire <asset>
    # mesh block, structurally malformed XML, no <worldbody> open tag).
    # Until a proper MJX-tuned MJCF lands, point MJX at the same fully-
    # populated ``panda.xml`` MuJoCo uses — it parses cleanly through
    # ``mjx_helper.compile`` and unblocks every -k mjx integration test.
    mjx_mjcf_path: str = "roboverse_data/robots/franka/mjcf/panda.xml"
    # urdf_path: str = "roboverse_data/robots/franka/urdf/panda.urdf"  # work for pybullet and sapien
    urdf_path: str = "roboverse_data/robots/franka/urdf/franka_panda.urdf"  # work for isaacgym
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    # NOTE 2026-05-26: ``effort_limit_sim`` deliberately not set here.
    # Setting it to Franka's published torque limits (87 N·m on joints 1-4,
    # 12 N·m on 5-7) closes the cross-backend asymmetry that ``task #6``
    # documented (MuJoCo inherits MJCF ``forcerange=±40``, URDF backends
    # inherit different values), BUT also changes motion at large errors
    # enough to trigger ``test_self_collision[mujoco-1]`` failures —
    # which means it can't land without coordinated test + downstream
    # policy review. The handler now warns when this asymmetry is in
    # play (see ``mujoco.py``/``newton.py`` actuator-override path), so
    # the gap is no longer silent. Closing task #6 requires the spec
    # decision, not just adding the field.
    actuators: dict[str, BaseActuatorCfg] = {
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=250, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
        "panda_finger_joint1": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=0.2, is_ee=True),
        "panda_finger_joint2": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=0.2, is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
        "panda_finger_joint1": (0.0, 0.04),  # 0.0 close, 0.04 open
        "panda_finger_joint2": (0.0, 0.04),  # 0.0 close, 0.04 open
    }

    ee_body_name: str = "panda_hand"
    ee_joint_names: list[str] = ["panda_finger_joint1", "panda_finger_joint2"]

    default_joint_positions: dict[str, float] = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    control_type: dict[str, Literal["position", "effort"]] = {
        "panda_joint1": "position",
        "panda_joint2": "position",
        "panda_joint3": "position",
        "panda_joint4": "position",
        "panda_joint5": "position",
        "panda_joint6": "position",
        "panda_joint7": "position",
        "panda_finger_joint1": "position",
        "panda_finger_joint2": "position",
    }

    # TODO: Make it more elegant
    gripper_open_q = [0.04, 0.04]
    gripper_close_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "franka.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
