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


def _maniskill_panda_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    return os.path.join(here, "roboverse_data", "robots", "maniskill_panda")


def _resolve_panda_asset(urdf_basename: str) -> str:
    """Resolve a vendored ManiSkill panda URDF, downloading the asset folder from HF if missing.

    Prefers the local ``roboverse_data/robots/maniskill_panda/`` copy; if absent, fetches it
    token-free from the public ``RoboVerseOrg/roboverse_data`` dataset (snapshot_download, the same
    HF-backed pattern as the other integrations) so the clone needs no ``mani_skill`` install; only
    then falls back to the installed ``mani_skill`` package.
    """
    vendored = os.path.join(_maniskill_panda_dir(), urdf_basename)
    if os.path.exists(vendored):
        return vendored
    try:
        from huggingface_hub import snapshot_download

        here = os.path.dirname(_maniskill_panda_dir())  # .../roboverse_data/robots
        root = os.path.dirname(os.path.dirname(here))  # repo root
        snapshot_download(
            repo_id="RoboVerseOrg/roboverse_data",
            repo_type="dataset",
            allow_patterns="robots/maniskill_panda/*",
            local_dir=os.path.join(root, "roboverse_data"),
        )
        if os.path.exists(vendored):
            return vendored
    except Exception:
        pass
    import mani_skill  # last-resort fallback — requires a mani_skill install

    return os.path.join(os.path.dirname(mani_skill.__file__), "assets", "robots", "panda", urdf_basename)


def panda_urdf_path() -> str:
    """ManiSkill ``panda_v2.urdf`` (vendored / HF / pkg)."""
    return _resolve_panda_asset("panda_v2.urdf")


# ManiSkill mounts the panda at the table edge (TableSceneBuilder); a few tasks rotate it.
PANDA_BASE_DEFAULT = ((-0.615, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def panda_stick_urdf_path() -> str:
    """ManiSkill ``panda_stick.urdf`` (arm + fixed stick, no gripper) — vendored / HF / pkg."""
    return _resolve_panda_asset("panda_stick.urdf")


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


def panda_v3_urdf_path() -> str:
    """ManiSkill ``panda_v3.urdf`` (panda_wristcam) — vendored / HF / pkg."""
    return _resolve_panda_asset("panda_v3.urdf")


def maniskill_panda_wristcam_cfg(name, base_position, base_orientation, rest_qpos=None) -> RobotCfg:
    """ManiSkill PandaWristCam (panda_v3.urdf: 7 arm + 2 fingers + wrist camera link)."""
    actuators = {
        j: BaseActuatorCfg(stiffness=1000.0, damping=100.0, effort_limit_sim=100.0)
        for j in (_ARM_JOINTS + _FINGER_JOINTS)
    }
    default = rest_qpos if rest_qpos is not None else dict(_REST_QPOS)
    robot = RobotCfg(name=name, actuators=actuators, fix_base_link=True, default_joint_positions=dict(default))
    robot.urdf_path = panda_v3_urdf_path()
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


# ManiSkill Panda gripper contact material (Panda.urdf_config._materials["gripper"]). ManiSkill
# applies this *in code*, not in the URDF, so a plain URDF load leaves the fingers at the scene
# default (0.3) and grasps slip — the cube drifts ~0.01 m during lift and success flips near the
# 0.025 m goal threshold. Replicating it pulls the contact-phase cube trajectory back to the PhysX
# CPU/CUDA noise floor (~2e-4).
GRIPPER_FRICTION = 2.0
GRIPPER_PATCH_RADIUS = 0.1
GRIPPER_FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")


def apply_maniskill_gripper_friction(
    robot,
    finger_links: tuple[str, ...] = GRIPPER_FINGER_LINKS,
    static_friction: float = GRIPPER_FRICTION,
    dynamic_friction: float = GRIPPER_FRICTION,
    restitution: float = 0.0,
    patch_radius: float = GRIPPER_PATCH_RADIUS,
    min_patch_radius: float = GRIPPER_PATCH_RADIUS,
) -> int:
    """Set the high-friction grasp material on a loaded panda articulation's finger links.

    Mirrors ManiSkill's ``Panda.urdf_config`` gripper material (``static_friction = dynamic_friction
    = 2.0``, ``restitution = 0``) plus the contact-patch radii (``0.1``). A fresh ``PhysxMaterial``
    is assigned per collision shape so a shared default material is never mutated. No-op for robots
    without these finger links (e.g. ``panda_stick``).

    Args:
        robot: a loaded SAPIEN articulation (raw ``PhysxArticulation`` or a mani_skill link struct).
        finger_links: link names to receive the gripper material.
        static_friction: per-shape static friction.
        dynamic_friction: per-shape dynamic friction.
        restitution: per-shape restitution.
        patch_radius: contact patch radius (skipped if the SAPIEN build lacks the setter).
        min_patch_radius: minimum contact patch radius.

    Returns:
        The number of collision shapes updated.
    """
    import sapien.physx as physx

    applied = 0
    for link in robot.get_links():
        name = link.get_name() if hasattr(link, "get_name") else link.name
        if name not in finger_links:
            continue
        raw = link._objs[0] if hasattr(link, "_objs") else link
        shapes = getattr(raw, "collision_shapes", None) or raw.get_collision_shapes()
        for cs in shapes:
            cs.set_physical_material(physx.PhysxMaterial(static_friction, dynamic_friction, restitution))
            if hasattr(cs, "set_patch_radius"):
                cs.set_patch_radius(patch_radius)
            if hasattr(cs, "set_min_patch_radius"):
                cs.set_min_patch_radius(min_patch_radius)
            applied += 1
    return applied


def apply_object_friction(
    obj,
    static_friction: float,
    dynamic_friction: float,
    restitution: float = 0.0,
    patch_radius: float | None = None,
) -> int:
    """Set a uniform contact material on every collision shape of a loaded rigid-body object.

    Mirrors a ManiSkill task that builds an object with a custom ``PhysxMaterial`` *in code* (e.g.
    PushT's Tee at friction 3.0, ``push_t.py``) rather than in the asset — without it the sapien3
    load leaves the object at the scene-default 0.3, so a long-horizon push slides the object off the
    demonstrated trajectory and the contact-phase pixel diff balloons (PushT 6.5 → ~0.1 / 255).

    Args:
        obj: a loaded SAPIEN entity (``task.handler.object_ids[name]``).
        static_friction: per-shape static friction.
        dynamic_friction: per-shape dynamic friction.
        restitution: per-shape restitution.
        patch_radius: contact patch radius; left at the SAPIEN default when ``None``.

    Returns:
        The number of collision shapes updated.
    """
    import sapien.physx as physx

    comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent) or obj.find_component_by_type(
        physx.PhysxRigidStaticComponent
    )
    if comp is None:
        return 0
    applied = 0
    for cs in comp.get_collision_shapes():
        cs.set_physical_material(physx.PhysxMaterial(static_friction, dynamic_friction, restitution))
        if patch_radius is not None and hasattr(cs, "set_patch_radius"):
            cs.set_patch_radius(patch_radius)
            if hasattr(cs, "set_min_patch_radius"):
                cs.set_min_patch_radius(patch_radius)
        applied += 1
    return applied


# decimation = sim_freq // control_freq (100 // 20 = 5).
DECIMATION = 5
