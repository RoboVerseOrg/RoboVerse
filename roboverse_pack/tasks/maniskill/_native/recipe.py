"""ManiSkill-faithful scene/physics building blocks for the shipped native tasks.

Encapsulates the four things a tabletop ManiSkill task needs to reproduce native dynamics through the
sapien3 handler: the panda robot cfg (ManiSkill ``panda_v2.urdf`` + its drive gains), the
kinematic table-workspace box, the ground altitude, and the ManiSkill ``SimConfig`` mapped onto the
opt-in ``SimParamCfg`` recipe knobs.
"""

from __future__ import annotations

import os

from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.simulator_params import SimParamCfg

# --- ManiSkill TableSceneBuilder geometry (box variant) ---
TABLE_HALF = (2.418 / 2, 1.209 / 2, 0.9196429 / 2)
TABLE_HEIGHT = 0.9196429
# A π/2 yaw quaternion (w, x, y, z); the box top sits at z = 0.
_TABLE_QUAT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)

# ManiSkill Panda drive gains (PDJointPosController: stiffness 1000, damping 100, force_limit 100).
_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
# Panda keyframe rest pose (ManiSkill Panda.keyframes["rest"], pre-noise).
_REST_QPOS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.7853981633974483,  # -π/4
    "panda_joint3": 0.0,
    "panda_joint4": -2.356194490192345,  # -3π/4
    "panda_joint5": 0.0,
    "panda_joint6": 1.5707963267948966,  # π/2
    "panda_joint7": 0.7853981633974483,  # π/4
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


def panda_urdf_path() -> str:
    """Resolve ManiSkill's ``panda_v2.urdf``: a vendored copy if present, else the installed package.

    Vendoring to ``roboverse_data/robots/maniskill_panda/`` (HF-backed, like the other integrations)
    is the follow-up that makes this fully clone-deletable; until then we fall back to the installed
    ``mani_skill`` asset so the task runs in the maniskill env.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    vendored = os.path.join(here, "roboverse_data", "robots", "maniskill_panda", "panda_v2.urdf")
    if os.path.exists(vendored):
        return vendored
    import mani_skill  # local import — only needed for the fallback path

    return os.path.join(os.path.dirname(mani_skill.__file__), "assets", "robots", "panda", "panda_v2.urdf")


# ManiSkill mounts the panda at the table edge (TableSceneBuilder); a few tasks rotate it.
PANDA_BASE_DEFAULT = ((-0.615, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def panda_stick_urdf_path() -> str:
    """Resolve ManiSkill's ``panda_stick.urdf`` (arm + fixed stick, no gripper): vendored else pkg."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    vendored = os.path.join(here, "roboverse_data", "robots", "maniskill_panda", "panda_stick.urdf")
    if os.path.exists(vendored):
        return vendored
    import mani_skill

    return os.path.join(os.path.dirname(mani_skill.__file__), "assets", "robots", "panda", "panda_stick.urdf")


def maniskill_panda_stick_cfg(
    name: str = "panda",
    base_position: tuple[float, float, float] = PANDA_BASE_DEFAULT[0],
    base_orientation: tuple[float, float, float, float] = PANDA_BASE_DEFAULT[1],
    rest_qpos: dict | None = None,
) -> RobotCfg:
    """ManiSkill PandaStick (7-dof arm, fixed stick end-effector, NO gripper)."""
    actuators = {j: BaseActuatorCfg(stiffness=1000.0, damping=100.0, effort_limit_sim=100.0) for j in _ARM_JOINTS}
    default = rest_qpos if rest_qpos is not None else {j: _REST_QPOS[j] for j in _ARM_JOINTS}
    robot = RobotCfg(name=name, actuators=actuators, fix_base_link=True, default_joint_positions=dict(default))
    robot.urdf_path = panda_stick_urdf_path()
    robot.default_position = tuple(base_position)
    robot.default_orientation = tuple(base_orientation)
    return robot


def maniskill_panda_cfg(
    name: str = "panda",
    base_position: tuple[float, float, float] = PANDA_BASE_DEFAULT[0],
    base_orientation: tuple[float, float, float, float] = PANDA_BASE_DEFAULT[1],
) -> RobotCfg:
    """The ManiSkill Panda as a RobotCfg with ManiSkill's exact drive gains + mount pose."""
    actuators = {
        j: BaseActuatorCfg(stiffness=1000.0, damping=100.0, effort_limit_sim=100.0)
        for j in (_ARM_JOINTS + _FINGER_JOINTS)
    }
    robot = RobotCfg(
        name=name,
        actuators=actuators,
        fix_base_link=True,
        default_joint_positions=dict(_REST_QPOS),
    )
    robot.urdf_path = panda_urdf_path()
    robot.default_position = tuple(base_position)
    robot.default_orientation = tuple(base_orientation)
    return robot


def table_workspace_cfg(name: str = "table-workspace") -> PrimitiveCubeCfg:
    """ManiSkill's kinematic table box; its top surface sits at z = 0."""
    return PrimitiveCubeCfg(
        name=name,
        size=[2 * TABLE_HALF[0], 2 * TABLE_HALF[1], 2 * TABLE_HALF[2]],
        mass=1.0,  # kinematic → mass is irrelevant
        color=[0.4, 0.3, 0.2],
        fix_base_link=True,
        default_position=(-0.12, 0.0, -TABLE_HEIGHT + TABLE_HALF[2]),
        default_orientation=_TABLE_QUAT,
    )


def maniskill_sim_params(sim_freq: int = 100, control_freq: int = 20) -> SimParamCfg:
    """ManiSkill ``SimConfig`` (default tabletop) mapped onto the opt-in SimParamCfg recipe knobs."""
    return SimParamCfg(
        dt=1.0 / sim_freq,
        contact_offset=0.02,
        rest_offset=0.0,
        num_position_iterations=15,
        num_velocity_iterations=1,
        sapien_apply_global_physx=True,
        sapien_disable_robot_gravity=True,
        sapien_drive_force_mode="force",
        sapien_ground_altitude=-TABLE_HEIGHT,
        sapien_sleep_threshold=0.005,
        sapien_bounce_threshold=2.0,
        sapien_enable_pcm=True,
        sapien_enable_tgs=True,
        sapien_enable_ccd=False,
        sapien_enable_enhanced_determinism=False,
        sapien_enable_friction_every_iteration=True,
        sapien_default_static_friction=0.3,
        sapien_default_dynamic_friction=0.3,
        sapien_default_restitution=0.0,
    )


# decimation = sim_freq // control_freq (100 // 20 = 5).
DECIMATION = 5
