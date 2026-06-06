"""Native-ManiSkill ↔ SAPIEN3 1:1 reproduction recipe.

This module encodes the *exact* SAPIEN 3 configuration that reproduces a native
ManiSkill (``sim_backend="physx_cpu"``) rollout bit-for-bit on object state and
to float32 solver-roundoff (~1e-4) on robot qpos.  It is the engineering ground
truth behind the MetaSim-native ManiSkill tasks and the parity harness.

Five things have to match ManiSkill, none of which the generic SAPIEN defaults
give you (discovered empirically — see ``parity_native.py`` for the measured
deltas):

1. **Robot links must have gravity disabled.**  ManiSkill sets
   ``link.disable_gravity = True`` on every robot link rather than feeding a
   gravity-compensation torque.  This is the single biggest factor — without it
   the arm sags and qpos diverges by ~3e-1 over 30 steps; with it, ~1e-4.
2. **The full PhysX config is set globally before the scene is created**
   (``physx.set_shape_config`` / ``set_body_config`` / ``set_scene_config`` /
   ``set_default_material``), mirroring ``BaseEnv._set_scene_config``.
3. **Joint drives use** ``set_drive_property(stiffness, damping, force_limit,
   mode="force")`` — the force limit and mode matter.
4. **The table is a kinematic box** (not a mesh), with the ground plane far
   below it; getting the cube-on-table contact exactly right is what turns the
   object delta from ~1e-5 into a true 0.
5. **The controller is ManiSkill's** ``pd_joint_*`` math (clip→scale→delta/abs,
   gripper mimic), run at ``decimation = sim_freq // control_freq``.

The capture/replica helpers read the scene straight out of a live ManiSkill env
so the recipe generalises across tasks without hand-coding each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ManiSkill's standard table-scene geometry (TableSceneBuilder, box variant).
TABLE_HALF = (2.418 / 2, 1.209 / 2, 0.9196429 / 2)
TABLE_INITIAL_P = (-0.12, 0.0, -0.9196429)
TABLE_LOCAL_OFFSET_P = (0.0, 0.0, 0.9196429 / 2)
TABLE_HEIGHT = 0.9196429  # ground plane sits at ``-TABLE_HEIGHT``


def clip_and_scale(action: np.ndarray, low: float, high: float) -> np.ndarray:
    """ManiSkill ``gym_utils.clip_and_scale_action`` (float32, matches torch)."""
    a = np.clip(np.float32(action), np.float32(-1), np.float32(1))
    return (np.float32(0.5) * np.float32(high + low) + np.float32(0.5) * np.float32(high - low) * a).astype(np.float32)


@dataclass
class ShapeSpec:
    """A single rigid collision shape pulled off a native actor."""

    kind: str  # "box" | "sphere" | "capsule" | "convex"
    pose_p: np.ndarray  # local shape pose (relative to the actor)
    pose_q: np.ndarray
    half_size: np.ndarray | None = None  # box
    radius: float | None = None  # sphere / capsule
    half_length: float | None = None  # capsule
    vertices: np.ndarray | None = None  # convex
    static_friction: float = 0.3
    dynamic_friction: float = 0.3
    restitution: float = 0.0


@dataclass
class ActorSpec:
    """A non-robot actor (dynamic or kinematic) to rebuild in the replica."""

    name: str
    pose_p: np.ndarray
    pose_q: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    mass: float
    is_kinematic: bool
    shapes: list[ShapeSpec] = field(default_factory=list)


@dataclass
class RobotSpec:
    urdf_path: str
    root_p: np.ndarray
    root_q: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    joint_names: list[str]
    stiffness: np.ndarray
    damping: np.ndarray
    force_limit: np.ndarray


@dataclass
class SimSpec:
    sim_freq: int
    control_freq: int
    gravity: np.ndarray
    bounce_threshold: float
    sleep_threshold: float
    contact_offset: float
    rest_offset: float
    solver_position_iterations: int
    solver_velocity_iterations: int
    enable_pcm: bool
    enable_tgs: bool
    enable_ccd: bool
    enable_enhanced_determinism: bool
    enable_friction_every_iteration: bool
    cpu_workers: int
    mat_static_friction: float
    mat_dynamic_friction: float
    mat_restitution: float

    @property
    def decimation(self) -> int:
        return self.sim_freq // self.control_freq


@dataclass
class SceneCapture:
    sim: SimSpec
    robot: RobotSpec
    actors: list[ActorSpec]
    has_table: bool = True


def _shape_from_native(shape) -> ShapeSpec | None:
    """Translate a SAPIEN ``PhysxCollisionShape`` into a ``ShapeSpec``."""
    import sapien.physx as physx

    pose = shape.get_local_pose()
    p, q = np.asarray(pose.p), np.asarray(pose.q)
    mat = shape.get_physical_material()
    kw = dict(
        pose_p=p,
        pose_q=q,
        static_friction=float(mat.static_friction),
        dynamic_friction=float(mat.dynamic_friction),
        restitution=float(mat.restitution),
    )
    if isinstance(shape, physx.PhysxCollisionShapeBox):
        return ShapeSpec("box", half_size=np.asarray(shape.half_size), **kw)
    if isinstance(shape, physx.PhysxCollisionShapeSphere):
        return ShapeSpec("sphere", radius=float(shape.radius), **kw)
    if isinstance(shape, physx.PhysxCollisionShapeCapsule):
        return ShapeSpec("capsule", radius=float(shape.radius), half_length=float(shape.half_length), **kw)
    if isinstance(shape, physx.PhysxCollisionShapeConvexMesh):
        return ShapeSpec("convex", vertices=np.asarray(shape.vertices) * np.asarray(shape.scale), **kw)
    # Plane / nonconvex (ground, table-mesh) are handled separately.
    return None


def capture_native(env, actor_filter=None) -> SceneCapture:
    """Read the full simulator state out of a freshly-reset ManiSkill env."""
    import sapien.physx as physx

    u = env.unwrapped
    scfg = u.sim_config
    sc = scfg.scene_config
    dm = scfg.default_materials_config
    sim = SimSpec(
        sim_freq=scfg.sim_freq,
        control_freq=scfg.control_freq,
        gravity=np.asarray(sc.gravity, dtype=np.float64),
        bounce_threshold=sc.bounce_threshold,
        sleep_threshold=sc.sleep_threshold,
        contact_offset=sc.contact_offset,
        rest_offset=sc.rest_offset,
        solver_position_iterations=sc.solver_position_iterations,
        solver_velocity_iterations=sc.solver_velocity_iterations,
        enable_pcm=sc.enable_pcm,
        enable_tgs=sc.enable_tgs,
        enable_ccd=sc.enable_ccd,
        enable_enhanced_determinism=sc.enable_enhanced_determinism,
        enable_friction_every_iteration=sc.enable_friction_every_iteration,
        cpu_workers=sc.cpu_workers,
        mat_static_friction=dm.static_friction,
        mat_dynamic_friction=dm.dynamic_friction,
        mat_restitution=dm.restitution,
    )

    robot = u.agent.robot
    art = robot._objs[0]
    ajoints = art.get_active_joints()
    rspec = RobotSpec(
        urdf_path=u.agent.urdf_path,
        root_p=robot.get_pose().raw_pose.cpu().numpy().ravel()[:3].copy(),
        root_q=robot.get_pose().raw_pose.cpu().numpy().ravel()[3:].copy(),
        qpos=robot.get_qpos().cpu().numpy().ravel().astype(np.float64).copy(),
        qvel=robot.get_qvel().cpu().numpy().ravel().astype(np.float64).copy(),
        joint_names=[j.get_name() for j in ajoints],
        stiffness=np.array([float(np.asarray(j.stiffness).ravel()[0]) for j in ajoints]),
        damping=np.array([float(np.asarray(j.damping).ravel()[0]) for j in ajoints]),
        force_limit=np.array([float(np.asarray(j.force_limit).ravel()[0]) for j in ajoints]),
    )

    actors: list[ActorSpec] = []
    has_table = False
    for name, actor in u.scene.actors.items():
        if name in ("ground",):
            continue
        if name == "table-workspace":
            has_table = True
            continue
        if actor_filter is not None and not actor_filter(name):
            continue
        ent = actor._objs[0]
        body = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
        is_kin = False
        if body is None:
            body = ent.find_component_by_type(physx.PhysxRigidStaticComponent)
        else:
            is_kin = bool(body.kinematic)
        if body is None:
            continue
        shapes = [s for s in (_shape_from_native(sh) for sh in body.get_collision_shapes()) if s is not None]
        if not shapes:
            continue
        rp = actor.pose.raw_pose.cpu().numpy().ravel()
        lin = getattr(body, "linear_velocity", np.zeros(3))
        ang = getattr(body, "angular_velocity", np.zeros(3))
        actors.append(
            ActorSpec(
                name=name,
                pose_p=rp[:3].copy(),
                pose_q=rp[3:].copy(),
                lin_vel=np.asarray(lin, dtype=np.float64).ravel()[:3].copy(),
                ang_vel=np.asarray(ang, dtype=np.float64).ravel()[:3].copy(),
                mass=float(body.mass) if hasattr(body, "mass") else 0.0,
                is_kinematic=is_kin,
                shapes=shapes,
            )
        )
    return SceneCapture(sim=sim, robot=rspec, actors=actors, has_table=has_table)


def apply_global_physx(sim: SimSpec) -> None:
    """Set the global SAPIEN PhysX config exactly like ``BaseEnv._set_scene_config``."""
    import sapien.physx as physx

    physx.set_shape_config(contact_offset=sim.contact_offset, rest_offset=sim.rest_offset)
    physx.set_body_config(
        solver_position_iterations=sim.solver_position_iterations,
        solver_velocity_iterations=sim.solver_velocity_iterations,
        sleep_threshold=sim.sleep_threshold,
    )
    physx.set_scene_config(
        gravity=sim.gravity,
        bounce_threshold=sim.bounce_threshold,
        enable_pcm=sim.enable_pcm,
        enable_tgs=sim.enable_tgs,
        enable_ccd=sim.enable_ccd,
        enable_enhanced_determinism=sim.enable_enhanced_determinism,
        enable_friction_every_iteration=sim.enable_friction_every_iteration,
        cpu_workers=sim.cpu_workers,
    )
    physx.set_default_material(sim.mat_static_friction, sim.mat_dynamic_friction, sim.mat_restitution)


def build_replica(cap: SceneCapture):
    """Build a clean SAPIEN3 scene reproducing the captured ManiSkill scene."""
    import sapien
    import sapien.physx as physx
    from transforms3d.euler import euler2quat

    apply_global_physx(cap.sim)
    scene = sapien.Scene([physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
    scene.set_timestep(1.0 / cap.sim.sim_freq)

    if cap.has_table:
        scene.add_ground(-TABLE_HEIGHT)
        mat0 = physx.PhysxMaterial(cap.sim.mat_static_friction, cap.sim.mat_dynamic_friction, cap.sim.mat_restitution)
        tb = scene.create_actor_builder()
        tb.add_box_collision(pose=sapien.Pose(p=list(TABLE_LOCAL_OFFSET_P)), half_size=list(TABLE_HALF), material=mat0)
        table = tb.build_kinematic(name="table-workspace")
        table.set_pose(sapien.Pose(p=list(TABLE_INITIAL_P), q=euler2quat(0, 0, np.pi / 2)))
    else:
        scene.add_ground(0.0)

    # Robot.
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(cap.robot.urdf_path)
    robot.set_root_pose(sapien.Pose(cap.robot.root_p, cap.robot.root_q))
    robot.set_qpos(cap.robot.qpos.astype(np.float32))
    robot.set_qvel(cap.robot.qvel.astype(np.float32))
    ajoints = robot.get_active_joints()
    for k, j in enumerate(ajoints):
        j.set_drive_property(
            stiffness=float(cap.robot.stiffness[k]),
            damping=float(cap.robot.damping[k]),
            force_limit=float(cap.robot.force_limit[k]),
            mode="force",
        )
    for link in robot.get_links():
        link.disable_gravity = True

    # Actors.
    actor_objs = {}
    for spec in cap.actors:
        b = scene.create_actor_builder()
        for sh in spec.shapes:
            mat = physx.PhysxMaterial(sh.static_friction, sh.dynamic_friction, sh.restitution)
            pose = sapien.Pose(sh.pose_p, sh.pose_q)
            if sh.kind == "box":
                b.add_box_collision(pose=pose, half_size=list(sh.half_size), material=mat)
            elif sh.kind == "sphere":
                b.add_sphere_collision(pose=pose, radius=sh.radius, material=mat)
            elif sh.kind == "capsule":
                b.add_capsule_collision(pose=pose, radius=sh.radius, half_length=sh.half_length, material=mat)
            elif sh.kind == "convex":
                # Rebuild a convex hull from the captured (scaled) vertices when
                # the builder supports it; otherwise fall back to its bbox.
                _add = getattr(b, "add_convex_collision_from_vertices", None)
                if _add is not None:
                    _add(vertices=sh.vertices, pose=pose, material=mat)
                else:
                    hs = (sh.vertices.max(0) - sh.vertices.min(0)) / 2
                    b.add_box_collision(pose=pose, half_size=list(hs), material=mat)
        if spec.is_kinematic:
            obj = b.build_kinematic(name=spec.name)
        else:
            obj = b.build(name=spec.name)
        obj.set_pose(sapien.Pose(spec.pose_p, spec.pose_q))
        # Match the captured mass exactly.
        for comp in obj.get_components():
            if isinstance(comp, physx.PhysxRigidDynamicComponent) and spec.mass > 0:
                comp.mass = spec.mass
                if np.any(spec.lin_vel) or np.any(spec.ang_vel):
                    comp.set_linear_velocity(spec.lin_vel)
                    comp.set_angular_velocity(spec.ang_vel)
        actor_objs[spec.name] = obj

    return scene, robot, actor_objs
