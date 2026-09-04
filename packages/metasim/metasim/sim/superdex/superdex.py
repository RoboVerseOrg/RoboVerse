"""SuperDex (Meta "Mochi" engine) backend for MetaSim.

`SuperDex <https://github.com/facebookresearch/project_superdex>`_ is a contact-first, fully
implicit rigid/articulated/soft-body engine with a Python API (``superdex.physics`` +
``superdex.robotics``). This handler maps the :class:`~metasim.sim.base.BaseSimHandler` contract
onto it:

* **Scene** — one ``superdex.physics.Scene`` per handler (single environment; ``num_envs > 1``
  is served by :class:`~metasim.sim.parallel.ParallelSimWrapper`, as for MuJoCo/PyBullet).
* **Assets** — robots and articulated objects are loaded from URDF through
  ``superdex.robotics.load_bot_prefab_from_urdf_file`` after :func:`~metasim.sim.superdex._assets.bake_urdf`
  has replaced every collision geometry by a watertight hull (SuperDex bakes SDF colliders and
  ignores URDF primitives). Primitive and mesh rigid objects become closed triangle-mesh actors.
* **Control** — ``RobotCfg.actuators`` stiffness/damping/effort limits become the per-joint gains
  of SuperDex's built-in *implicit* pose controller, so stiff PD targets stay stable at the
  10-25 ms steps the engine is designed for. Velocity targets use the same controller; effort
  targets are applied as external DoF forces.
* **State** — root/body poses come from the actor and link transforms (SuperDex quaternions are
  ``xyzw``; MetaSim's are ``wxyz``), joint state from the articulated pose/velocity arrays.
* **Cameras** — SuperDex has no headless renderer, so RGB + depth are produced by
  :class:`~metasim.sim.superdex._renderer.OffscreenRenderer` (pyrender/EGL) from the physics link
  transforms. Segmentation and mounted cameras are not implemented and are rejected at launch.

Known limitations (all reported loudly rather than silently ignored): CPU only; no GUI viewer;
Python >= 3.12 (the ``superdex-*`` wheels); no instance segmentation; per-object friction /
restitution from ``BaseRigidObjCfg`` are not yet forwarded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger as log

from metasim.scenario.objects import (
    ArticulationObjCfg,
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.sim.superdex import _assets
from metasim.types import CameraState, CompatActionInput, DictStateBatch, ObjectState, RobotState, TensorState
from metasim.utils.state import adapt_actions_to_dict

try:
    import superdex.physics as sdp
    import superdex.robotics as sdr
except ImportError as _exc:  # pragma: no cover - surfaced by get_sim_handler_class with an install hint
    raise ImportError(
        "SuperDex is not installed. It ships Python >= 3.12 wheels: "
        "python -m pip install superdex-physics superdex-robotics  (or metasim[superdex])"
    ) from _exc

DEFAULT_DT = 1.0 / 1000.0
"""Physics step ``decimation`` counts when ``ScenarioCfg.sim_params.dt`` is None.

Same default as the MuJoCo backend (``mjcf_model.option.timestep = 0.001``), so an env step
(``dt * decimation``) spans the same simulated time on both backends and drop / tracking
trajectories line up step for step. It is *not* the step the solver takes: see ``DEFAULT_SOLVER_DT``.
"""

DEFAULT_SOLVER_DT = 5.0 / 1000.0
"""Step the SuperDex solver actually takes when ``sim_params.superdex_solver_dt`` is None.

SuperDex integrates fully implicitly and is stable at 10-25 ms; stepping it at 1 ms only burns time
(its maintainers' guidance). ``_simulate`` therefore covers the env step (``decimation * dt``) in
``round(env_step / solver_dt)`` solver steps of ``env_step / n`` each, so the simulated time per env
step is exactly what ``dt`` and ``decimation`` say on every backend: 15 x 1 ms becomes 3 x 5 ms,
15 x 5 ms stays 15 x 5 ms, 4 x 1 ms becomes 1 x 4 ms (the step is rounded to divide the env step,
so it is 5 ms only when the env step is a multiple of 5 ms). Set ``superdex_solver_dt=0.001`` to
recover the old 1 ms solver stepping, e.g. for a bit-for-bit comparison with an older recording.
"""

DEFAULT_FRICTION = 1.0
"""Coulomb friction used when a cfg sets none: MuJoCo's geom default, so contacts match that backend
(SuperDex's own default is 0.5)."""

JOINT_LIMIT_STIFFNESS = 5.0e6
JOINT_LIMIT_DAMPING = 10.0
"""URDF limits are hard constraints in MuJoCo/PhysX; SuperDex models them as penalties (its URDF loader
sets stiffness 100, which the pose controller pushes through). These gains make the range effectively
hard at the default step (measured on the Franka fingers: 100 -> 29 mm overshoot past the range,
5e4 -> 1.2 mm, 5e6 -> 0.0 mm)."""

_ONE_DOF_JOINTS = {"REVOLUTE", "PRISMATIC", "CYCLE"}
_PRIMITIVE_CFGS = (PrimitiveCubeCfg, PrimitiveSphereCfg, PrimitiveCylinderCfg)


def _friction(cfg) -> float:
    """Coulomb friction for a cfg: ``static_friction`` if set, else the MuJoCo-compatible default."""
    value = getattr(cfg, "static_friction", None)
    return float(value) if value is not None else DEFAULT_FRICTION


def _joint_type_name(joint_type) -> str:
    return str(joint_type).split(".")[-1].upper()


def _joint_dofs(joint_type) -> int:
    name = _joint_type_name(joint_type)
    if name in _ONE_DOF_JOINTS:
        return 1
    if name == "SPHERICAL":
        return 3
    if name == "FREE":
        return 6
    return 0  # HARD


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _transform_from_wxyz(pos, quat_wxyz):
    q = _np(quat_wxyz)
    return sdp.TransformRT(rotation=[q[1], q[2], q[3], q[0]], translation=_np(pos).tolist())


def _wxyz_from_transform(tf) -> tuple[np.ndarray, np.ndarray]:
    rot = tf.rotation
    quat = np.array([rot[3], rot[0], rot[1], rot[2]], dtype=np.float64)  # xyzw -> wxyz
    return np.asarray(tf.translation, dtype=np.float64), quat


def _matrix_from_transform(tf) -> np.ndarray:
    pos, q = _wxyz_from_transform(tf)
    w, x, y, z = q
    mat = np.eye(4)
    mat[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    mat[:3, 3] = pos
    return mat


@dataclass
class _Articulation:
    """Book-keeping for one articulated actor (robot or ArticulationObjCfg)."""

    cfg: BaseObjCfg
    bot: object
    actor: object
    prefab: object
    baked: _assets.BakedUrdf
    link_names: list[str]
    joint_names: list[str]
    """1-DoF joint names in actor DoF order (SuperDex joint order minus the root and fixed joints)."""
    joint_dof_index: dict[str, int]
    base_dofs: int
    num_dofs: int
    controlled: bool = False
    target_pose: np.ndarray | None = None
    target_vel: np.ndarray | None = None
    effort: np.ndarray | None = None
    kp: np.ndarray | None = None
    """Per actor DoF PD gains / clamp (pd control mode); zero for uncontrolled DoFs."""
    kd: np.ndarray | None = None
    effort_limit: np.ndarray | None = None
    last_tau: np.ndarray | None = None
    link_actors: list = field(default_factory=list)
    contact_queries: list = field(default_factory=list)


@dataclass
class _Rigid:
    cfg: BaseObjCfg
    actor: object
    is_static: bool


class SuperdexHandler(BaseSimHandler):
    """MetaSim handler backed by SuperDex Physics (see module docstring)."""

    set_states_refreshes = True  # ``_set_states`` pushes the poses into the renderer

    _set_states_input_type = "dict"

    def __init__(self, scenario: ScenarioCfg, optional_queries=None):
        super().__init__(scenario, optional_queries)
        self._scene = None
        self._robotics_ctx = None
        self._articulations: dict[str, _Articulation] = {}
        self._rigids: dict[str, _Rigid] = {}
        self._renderer = None
        physics_dt = scenario.sim_params.dt if scenario.sim_params.dt is not None else DEFAULT_DT
        env_step = float(physics_dt) * int(self.decimation)
        if not (env_step > 0.0):
            raise ValueError(
                f"[superdex] dt * decimation = {physics_dt} * {self.decimation} must be > 0 (the scene would never advance)"
            )
        solver_dt = getattr(scenario.sim_params, "superdex_solver_dt", None)
        solver_dt = float(solver_dt) if solver_dt is not None else DEFAULT_SOLVER_DT
        if not (solver_dt > 0.0):
            raise ValueError(f"[superdex] sim_params.superdex_solver_dt={solver_dt!r} must be > 0")
        self._substeps = max(1, round(env_step / solver_dt))
        self._dt = env_step / self._substeps  # solver step; divides the env step exactly
        log.debug(
            f"[superdex] env step {env_step * 1e3:.3g} ms = {self._substeps} solver step(s) of {self._dt * 1e3:.3g} ms "
            f"(dt={physics_dt}, decimation={self.decimation}, solver_dt={solver_dt})"
        )
        self._num_threads = int(getattr(scenario.sim_params, "num_threads", 0) or 0)
        self._cache_dir = _assets.default_cache_dir()
        self._control_mode = getattr(scenario.sim_params, "superdex_control_mode", "pd")
        if self._control_mode not in ("pd", "implicit"):
            raise ValueError(
                f"[superdex] unknown superdex_control_mode {self._control_mode!r} (expected 'pd' or 'implicit')"
            )
        self._sim_time = 0.0
        self._contact_queries_fresh = False

    # ------------------------------------------------------------------ lifecycle
    def launch(self) -> None:
        if not self.headless:
            log.warning("[superdex] SuperDex has no interactive viewer; running headless (cameras still render).")
        self._validate_cameras()
        if not sdp.is_initialized():
            sdp.initialize(num_worker_threads=self._num_threads)
        self._scene = sdp.create_scene(f"metasim-{os.getpid()}-{id(self)}")
        self._scene.set_gravity([float(g) for g in self.scenario.gravity])
        self._robotics_ctx = sdr.create_context()

        if self.cameras:
            from metasim.sim.superdex._renderer import OffscreenRenderer

            self._renderer = OffscreenRenderer()

        if self.scenario.add_default_ground:
            plane = sdp.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
            self._scene.create_rigid_actor(name="ground", shape=plane, is_static=True)
            if self._renderer is not None:
                self._renderer.add_ground()

        for obj in self.objects:
            self._add_object(obj)
        for robot in self.robots:
            self._add_articulation(robot, is_robot=True)

        self._push_render_poses()
        super().launch()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._scene is not None:
            for art in self._articulations.values():
                try:
                    sdr.destroy_bot(self._scene, art.bot)
                except Exception as exc:  # engine teardown must not mask the caller's exception
                    log.debug(f"[superdex] destroy_bot({art.cfg.name}) during close: {exc}")
            self._articulations.clear()
            self._rigids.clear()
            sdp.destroy_scene(self._scene)
            self._scene = None
        self._robotics_ctx = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def scene(self):
        """The underlying ``superdex.physics.Scene`` (for advanced users / tests)."""
        return self._scene

    # ------------------------------------------------------------------ scene construction
    def _validate_cameras(self) -> None:
        for cam in self.cameras:
            unsupported = [dt for dt in cam.data_types if dt not in ("rgb", "depth")]
            if unsupported:
                raise NotImplementedError(
                    f"[superdex] camera '{cam.name}' requests {unsupported}; the SuperDex backend renders rgb/depth only"
                )
            if cam.mount_to is not None:
                if cam.mount_to not in {o.name for o in self.objects} | {r.name for r in self.robots}:
                    raise ValueError(f"[superdex] camera '{cam.name}' is mounted to unknown object '{cam.mount_to}'")
                if cam.mount_link is None or cam.mount_pos is None or cam.mount_quat is None:
                    raise ValueError(
                        f"[superdex] camera '{cam.name}': mount_to needs mount_link, mount_pos and mount_quat"
                    )

    def _add_object(self, cfg: BaseObjCfg) -> None:
        if isinstance(cfg, ArticulationObjCfg):
            self._add_articulation(cfg, is_robot=False)
        elif isinstance(cfg, _PRIMITIVE_CFGS):
            mesh = _assets.primitive_trimesh(cfg)
            color = tuple(float(c) for c in cfg.color[:3])
            self._add_rigid_mesh(cfg, mesh, mass=float(cfg.mass), density=None, visuals=[(mesh, np.eye(4), color)])
        elif isinstance(cfg, RigidObjCfg):
            self._add_rigid_from_file(cfg)
        else:
            raise NotImplementedError(f"[superdex] object type {type(cfg).__name__} ('{cfg.name}') is not supported")

    def _add_rigid_mesh(self, cfg, mesh, *, mass, density, visuals) -> None:
        coords, conn = _assets.mesh_to_arrays(_assets.watertight_hull(mesh))
        shape = sdp.create_tri_mesh_shape(coordinates=coords, connectivity=conn)
        is_static = bool(cfg.fix_base_link)
        kwargs = {}
        if not is_static:
            if mass is not None:
                kwargs["mass"] = float(mass)
            elif density is not None:
                kwargs["density"] = float(density)
        contact = sdp.ContactParams()
        contact.coulomb_friction_coefficient = _friction(cfg)
        position = self._lift_out_of_ground(cfg, mesh) if not is_static else tuple(cfg.default_position)
        actor = self._scene.create_rigid_actor(
            name=cfg.name,
            shape=shape,
            is_static=is_static,
            world_from_local=_transform_from_wxyz(position, cfg.default_orientation),
            has_gravity=bool(cfg.enabled_gravity),
            contact=contact,
            **kwargs,
        )
        if actor is None:
            raise RuntimeError(f"[superdex] failed to create rigid actor '{cfg.name}'")
        self._rigids[cfg.name] = _Rigid(cfg=cfg, actor=actor, is_static=is_static)
        if self._renderer is not None:
            self._renderer.add_body(cfg.name, cfg.name, visuals)

    GROUND_HEIGHT = 0.0  # the default ground plane is z = 0 (``create_plane_shape(distance=0.0)``)
    GROUND_CLEARANCE = 1e-3  # m; a spawned body's hull is lifted to at least this far above the ground

    def _lift_out_of_ground(self, cfg, mesh) -> tuple[float, float, float]:
        """Spawn position for a dynamic rigid body, lifted so its collision hull clears the ground.

        SuperDex resolves an initial interpenetration with the ground SDF as an impulse: a hull that
        starts 10 cm inside the plane leaves at ~90 m/s on the first step (measured with a scaled
        URDF whose origin sits at the mesh centre while the scenario height was authored for an MJCF
        with a bottom-centred origin). MuJoCo tolerates the same overlap. Lifting the body is a
        deterministic, loudly logged correction; a penetrating spawn is never silently accepted.
        """
        if not self.scenario.add_default_ground:
            return tuple(cfg.default_position)
        pos = np.asarray(cfg.default_position, dtype=np.float64)
        rot = _matrix_from_transform(_transform_from_wxyz(cfg.default_position, cfg.default_orientation))[:3, :3]
        lowest = float((np.asarray(mesh.vertices, dtype=np.float64) @ rot.T)[:, 2].min()) + pos[2]
        ground_z = self.GROUND_HEIGHT
        if lowest >= ground_z + self.GROUND_CLEARANCE:
            return tuple(cfg.default_position)
        lift = ground_z + self.GROUND_CLEARANCE - lowest
        log.warning(
            f"[superdex] '{cfg.name}' would spawn {lowest - ground_z:.4f} m into the ground (hull bottom below "
            f"z={ground_z}); lifting it by {lift:.4f} m to avoid the depenetration impulse. Check the asset's "
            "origin (URDF vs MJCF) or default_position."
        )
        return (float(pos[0]), float(pos[1]), float(pos[2] + lift))

    def _add_rigid_from_file(self, cfg: RigidObjCfg) -> None:
        import trimesh

        path = cfg.urdf_path or cfg.mesh_path
        if path is None:
            raise ValueError(f"[superdex] rigid object '{cfg.name}' needs a urdf_path or mesh_path")
        scale = cfg.scale if isinstance(cfg.scale, (tuple, list)) else (cfg.scale,) * 3
        visuals = []
        if path.lower().endswith(".urdf"):
            baked = _assets.bake_urdf(path, scale=tuple(float(s) for s in scale), cache_dir=self._cache_dir)
            if len(baked.link_names) > 1:
                log.warning(
                    f"[superdex] rigid object '{cfg.name}' URDF has {len(baked.link_names)} links; "
                    "loading it as ONE rigid body (use ArticulationObjCfg for jointed assets)"
                )
            hull_meshes = []
            for link in baked.link_names:
                for hull_path, link_from_geom in baked.collisions.get(link, []):
                    hull = trimesh.load(hull_path, force="mesh")
                    hull.apply_transform(link_from_geom)
                    hull_meshes.append(hull)
                for vis in baked.visuals.get(link, []):
                    visuals.append((vis.mesh, vis.link_from_geom, vis.color))
            if not hull_meshes:
                raise ValueError(f"[superdex] rigid object '{cfg.name}': URDF has no collision geometry")
            mesh = trimesh.util.concatenate(hull_meshes) if len(hull_meshes) > 1 else hull_meshes[0]
            mass = baked.link_masses.get(baked.link_names[0]) if baked.link_masses else None
        else:
            mesh = trimesh.load(path, force="mesh")
            mesh.apply_scale(np.asarray(scale, dtype=np.float64))
            visuals.append((mesh, np.eye(4), None))
            mass = None
        self._add_rigid_mesh(cfg, mesh, mass=mass, density=cfg.mesh_density, visuals=visuals)

    def _add_articulation(self, cfg: ArticulationObjCfg, *, is_robot: bool) -> None:
        if cfg.urdf_path is None:
            raise ValueError(f"[superdex] '{cfg.name}' needs a urdf_path (SuperDex loads URDF only)")
        scale = cfg.scale if isinstance(cfg.scale, (tuple, list)) else (cfg.scale,) * 3
        baked = _assets.bake_urdf(cfg.urdf_path, scale=tuple(float(s) for s in scale), cache_dir=self._cache_dir)
        prefab = sdr.load_bot_prefab_from_urdf_file(baked.path)
        prefab.name = cfg.name
        fix_base = bool(cfg.fix_base_link)
        prefab.joints[0].type = sdp.ArticulatedJointType.HARD if fix_base else sdp.ArticulatedJointType.FREE
        self._apply_link_and_joint_params(cfg, prefab, baked)

        link_names = [link.name for link in prefab.links]
        joint_names: list[str] = []
        joint_dof_index: dict[str, int] = {}
        n_joint_dofs = 0
        # SuperDex joint i is the parent joint of link i; index 0 is the injected root joint.
        for i in range(1, len(prefab.joints)):
            joint = prefab.joints[i]
            ndof = _joint_dofs(joint.type)
            if ndof == 1:
                joint_names.append(joint.name)
                joint_dof_index[joint.name] = n_joint_dofs  # offset by base_dofs below
            elif ndof > 1:
                raise NotImplementedError(
                    f"[superdex] '{cfg.name}': joint '{joint.name}' is {_joint_type_name(joint.type)}; "
                    "only 1-DoF joints map onto MetaSim's scalar joint state"
                )
            n_joint_dofs += ndof

        if not fix_base:
            massless = [
                prefab.links[i].name
                for i in range(len(prefab.links))
                if prefab.links[i].mass is None and prefab.links[i].density is None
            ]
            if massless and not (baked.link_inertials or baked.link_masses):
                pass  # engine derives every mass from hull volume x default density: consistent, allowed
            elif massless:
                raise ValueError(
                    f"[superdex] '{cfg.name}' has a free base but links {massless} carry no <inertial> mass "
                    "(the solver goes singular and the body never moves); add inertials to the URDF or set fix_base_link=True"
                )
        bot = sdr.create_bot(self._scene, prefab, self._robotics_ctx)
        if bot is None:
            raise RuntimeError(f"[superdex] create_bot failed for '{cfg.name}' ({baked.path})")
        if getattr(cfg, "enabled_self_collisions", True) is False:
            # Convex-hull colliders of neighbouring links overlap, so without this the arm fights its
            # own contact penalties (a Franka needed > 12 N m just to hold joint 6 still).
            self._scene.enable_layer_contact_symmetric(f"metasim/{cfg.name}", f"metasim/{cfg.name}", enable=False)
        actor = bot.get_articulated_actor()
        num_dofs = int(actor.get_num_dofs())
        base_dofs = num_dofs - n_joint_dofs
        if base_dofs not in (0, 6):
            raise RuntimeError(
                f"[superdex] '{cfg.name}': actor has {num_dofs} DoFs but the URDF joints account for {n_joint_dofs}"
            )
        joint_dof_index = {name: idx + base_dofs for name, idx in joint_dof_index.items()}

        art = _Articulation(
            cfg=cfg,
            bot=bot,
            actor=actor,
            prefab=prefab,
            baked=baked,
            link_names=link_names,
            joint_names=joint_names,
            joint_dof_index=joint_dof_index,
            base_dofs=base_dofs,
            num_dofs=num_dofs,
        )
        art.link_actors = [self._scene.get_actor(h) for h in actor.get_nested_link_actors()]
        self._articulations[cfg.name] = art

        actor.set_root_transform(_transform_from_wxyz(cfg.default_position, cfg.default_orientation))
        pose = np.zeros(num_dofs, dtype=np.float64)
        defaults = getattr(cfg, "default_joint_positions", None) or {}
        for name, value in defaults.items():
            if name in joint_dof_index:
                pose[joint_dof_index[name]] = float(value)
        actor.set_articulated_pose_from_joints(pose)
        actor.set_articulated_joint_velocities(np.zeros(num_dofs))

        if is_robot:
            self._attach_pose_controller(art, pose)

        if self._renderer is not None:
            for link in link_names:
                geoms = [(v.mesh, v.link_from_geom, v.color) for v in baked.visuals.get(link, [])]
                if geoms:
                    self._renderer.add_body(cfg.name, link, geoms)

    def _apply_link_and_joint_params(self, cfg: ArticulationObjCfg, prefab, baked: _assets.BakedUrdf) -> None:
        """Align per-link / per-joint physics with what the other backends read from the same URDF.

        * mass, centre of mass and inertia tensor from ``<inertial>`` (SuperDex's URDF loader leaves them
          unset and would derive mass from hull volume x default density);
        * Coulomb friction from ``static_friction`` or the MuJoCo-compatible default;
        * ``enabled_gravity=False`` -> no gravity on any link (the MuJoCo backend's gravity compensation);
        * stiff joint-limit penalties so URDF ranges behave like hard limits.
        """
        links = prefab.links
        friction = _friction(cfg)
        # One contact layer per articulation so self-collision can be switched off as a whole
        # (``enabled_self_collisions=False`` excludes every same-robot contact on the MuJoCo backend).
        self_layer = f"metasim/{cfg.name}"
        inertials = baked.link_inertials
        if not inertials and getattr(cfg, "mjcf_path", None) and os.path.isfile(cfg.mjcf_path):
            inertials = _assets.mjcf_inertials(cfg.mjcf_path)
            if inertials:
                log.info(
                    f"[superdex] '{cfg.name}': URDF has no <inertial>; using the explicit MJCF inertials of "
                    f"{len(inertials)} bodies from {cfg.mjcf_path} so mass distribution matches the MuJoCo backend"
                )
        for i in range(len(links)):
            link = links[i]
            inertial = inertials.get(link.name)
            if inertial is not None:
                link.mass = float(inertial.mass)
                link.density = None
                link.center_of_mass = inertial.com.tolist()
                t = inertial.inertia
                link.moment_of_inertia = [t[0, 0], t[0, 1], t[0, 2], t[1, 1], t[1, 2], t[2, 2]]
            contact = link.contact
            contact.coulomb_friction_coefficient = friction
            link.contact = contact
            link.layer = self_layer
            if not cfg.enabled_gravity:
                link.has_gravity = False
            links[i] = link
        prefab.links = links
        # Joint damping / Coulomb friction: URDF <dynamics>, else the explicit MJCF values the MuJoCo
        # backend uses, else zero (MuJoCo's default). SuperDex's URDF loader would otherwise inject a
        # viscous friction of 10 N m s/rad on every joint, which alone halves a 40 N m step response.
        dynamics = baked.joint_dynamics
        if not dynamics and getattr(cfg, "mjcf_path", None) and os.path.isfile(cfg.mjcf_path):
            dynamics = _assets.mjcf_joint_dynamics(cfg.mjcf_path)
        actuators = getattr(cfg, "actuators", None) or {}
        joints = prefab.joints
        for i in range(1, len(joints)):
            joint = joints[i]
            if _joint_dofs(joint.type) != 1:
                continue
            joint.limit_stiffness = JOINT_LIMIT_STIFFNESS
            joint.limit_damping = JOINT_LIMIT_DAMPING
            damping, coulomb = dynamics.get(joint.name, (0.0, 0.0))
            act = actuators.get(joint.name)
            if act is not None and getattr(act, "frictionloss", None) is not None:
                coulomb = float(act.frictionloss)  # cfg override, as on MuJoCo
            fric = joint.friction
            fric.viscous = float(damping)
            fric.coulomb = float(coulomb)
            joint.friction = fric
            if act is not None and getattr(act, "armature", None) is not None:
                joint.inertia = float(act.armature)
            joints[i] = joint
        prefab.joints = joints

    def _attach_pose_controller(self, art: _Articulation, initial_pose: np.ndarray) -> None:
        """Turn ``RobotCfg.actuators`` gains into the robot's joint controller.

        ``pd`` mode (default) keeps per-DoF ``kp`` / ``kd`` / effort clamps and applies
        ``clip(kp (q* - q) - kd (qd - qd*), +-limit)`` as an external DoF force every substep — the same
        law and clamp as the MuJoCo backend's actuators. ``implicit`` mode hands the gains to SuperDex's
        native constraint-based pose controller (its ``saturation`` is not an effort clamp in N m; the
        controller converges in a few ms regardless of the limit).
        """
        cfg: RobotCfg = art.cfg  # type: ignore[assignment]
        actuators = cfg.actuators or {}
        n_links = len(art.link_names)
        kp = np.zeros(art.num_dofs)
        kd = np.zeros(art.num_dofs)
        limit = np.full(art.num_dofs, np.inf)
        mjcf_limits: dict[str, float] = {}
        if getattr(cfg, "mjcf_path", None) and os.path.isfile(cfg.mjcf_path):
            mjcf_limits = _assets.mjcf_actuator_limits(cfg.mjcf_path)
        missing: list[str] = []
        for i in range(1, n_links):
            joint = art.prefab.joints[i]
            if _joint_dofs(joint.type) != 1:
                continue
            act = actuators.get(joint.name)
            if act is None:
                missing.append(joint.name)
                continue
            if not act.fully_actuated:
                continue
            dof = art.joint_dof_index[joint.name]
            kp[dof] = float(act.stiffness) if act.stiffness is not None else 0.0
            kd[dof] = float(act.damping) if act.damping is not None else 0.0
            # Effort clamp precedence mirrors the MuJoCo backend: an explicit ``effort_limit_sim`` wins;
            # otherwise the asset-authored clamp -- the MJCF actuator ``forcerange`` when the cfg also
            # ships an MJCF (that is what the MuJoCo backend clamps with), else the URDF ``<limit effort>``
            # (SuperDex parses it into the prefab but does not enforce it); otherwise unbounded.
            if act.effort_limit_sim is not None:
                limit[dof] = float(act.effort_limit_sim)
            elif joint.name in mjcf_limits:
                limit[dof] = mjcf_limits[joint.name]
            elif float(joint.effort_limit) > 0:
                limit[dof] = float(joint.effort_limit)
        if missing:
            log.warning(
                f"[superdex] robot '{cfg.name}': joints {missing} have no entry in RobotCfg.actuators and stay passive"
            )
        art.kp, art.kd, art.effort_limit = kp, kd, limit
        art.controlled = True
        art.target_pose = initial_pose.copy()
        # Both modes use SuperDex's implicit pose controller for the spring/damper (an explicit PD with
        # kd ~ 1e4 is unstable at any usable substep). ``pd`` mode additionally enforces the effort clamp
        # by shaping the target every substep (see ``_apply_pd_torques``); ``implicit`` mode passes the
        # clamp as the controller's own ``saturation`` (not an N m clamp in practice).
        params = sdp.PoseControllerParams(num_links=n_links)
        tracking = params.joint_tracking
        for i in range(1, n_links):
            joint = art.prefab.joints[i]
            if _joint_dofs(joint.type) != 1 or kp[art.joint_dof_index[joint.name]] == 0.0:
                continue
            dof = art.joint_dof_index[joint.name]
            p = tracking[i]
            p.stiffness, p.damping = float(kp[dof]), float(kd[dof])
            p.saturation = float(limit[dof]) if (self._control_mode == "implicit" and np.isfinite(limit[dof])) else -1.0
            tracking[i] = p
        params.joint_tracking = tracking
        art.actor.add_articulated_pose_controller(params)
        art.actor.set_articulated_target_pose(art.target_pose)

    def _apply_pd_torques(self, art: _Articulation) -> None:
        """One substep of effort-clamped PD (pd control mode) plus any effort targets.

        The MuJoCo backend applies ``tau = clip(kp e - kd qd, +-L)`` (``e = q* - q``). Applying that
        explicitly is unstable here (kd ~ 1e4 at a 1 ms solver step; coarser solver steps make the clamp reshaping below coarser too), so the spring/damper stays inside
        SuperDex's implicit controller and the clamp is reproduced by shaping what the controller
        tracks each substep:

        * unsaturated (``|kp e| <= L``): target ``q*``, target velocity 0 — plain PD;
        * saturated: the spring excursion is bounded to ``L / kp`` so the spring force is ``+-L``, and
          the damper is referenced to ``v_ref = (kp e - L sign(e)) / kd`` — the steady velocity a
          clamped PD settles at — so it does not fight the motion the clamp allows.
        """
        pose, vel = self._joint_arrays(art)
        active = art.kp > 0
        kp = np.where(active, art.kp, 1.0)
        kd = np.where(art.kd > 0, art.kd, 1.0)
        err = art.target_pose - pose
        spring = art.kp * err
        saturated = active & (np.abs(spring) > art.effort_limit)
        excursion = np.where(saturated, art.effort_limit / kp, np.inf)
        q_eff = art.target_pose.copy()
        q_eff[active] = pose[active] + np.clip(err, -excursion, excursion)[active]
        v_ref = np.zeros_like(pose)
        v_ref[saturated] = ((spring - art.effort_limit * np.sign(err)) / kd)[saturated]
        if art.target_vel is not None:
            v_ref = v_ref + art.target_vel
        if art.base_dofs:
            q_eff[: art.base_dofs] = 0.0
            v_ref[: art.base_dofs] = 0.0
        art.actor.set_articulated_target_pose(q_eff)
        art.actor.set_articulated_target_velocity(v_ref)
        art.last_tau = np.clip(spring - art.kd * (vel - v_ref), -art.effort_limit, art.effort_limit)
        if art.effort is not None:
            idx = np.arange(art.base_dofs, art.num_dofs, dtype=np.int32)
            art.actor.set_external_forces_on_dofs(dof_indices=idx, force_values=art.effort.astype(np.float32))

    # ------------------------------------------------------------------ simulation
    def _simulate(self) -> None:
        for _ in range(self._substeps):
            for art in self._articulations.values():
                if art.controlled and self._control_mode == "pd":
                    self._apply_pd_torques(art)
                elif art.effort is not None:
                    idx = np.arange(art.base_dofs, art.num_dofs, dtype=np.int32)
                    art.actor.set_external_forces_on_dofs(dof_indices=idx, force_values=art.effort.astype(np.float32))
            self._scene.step(self._dt)
            self._sim_time += self._dt
        self._contact_queries_fresh = True
        self._push_render_poses()

    def refresh_render(self) -> None:
        self._push_render_poses()

    # ------------------------------------------------------------------ actions
    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        self._actions_cache = actions
        per_robot = adapt_actions_to_dict(self, actions)
        for name, payload in per_robot.items():
            art = self._articulations.get(name)
            if art is None:
                raise KeyError(f"[superdex] action for unknown robot '{name}'")
            if not art.controlled:
                raise RuntimeError(f"[superdex] '{name}' has no controller (it is not a RobotCfg)")
            pos_targets = payload.get("dof_pos_target")
            if pos_targets:
                for jn, value in pos_targets.items():
                    idx = art.joint_dof_index.get(jn)
                    if idx is None:
                        raise KeyError(f"[superdex] robot '{name}' has no joint '{jn}'")
                    art.target_pose[idx] = float(value)
                if self._control_mode == "implicit":
                    art.actor.set_articulated_target_pose(art.target_pose)
            vel_targets = payload.get("dof_vel_target")
            if vel_targets:
                # Best effort, like the MuJoCo backend: the pose controller keeps pulling towards
                # ``target_pose``; the velocity target only feeds its damping term.
                vel = np.zeros(art.num_dofs, dtype=np.float64)
                for jn, value in vel_targets.items():
                    idx = art.joint_dof_index.get(jn)
                    if idx is None:
                        raise KeyError(f"[superdex] robot '{name}' has no joint '{jn}' (dof_vel_target)")
                    vel[idx] = float(value)
                if self._control_mode == "implicit":
                    art.actor.set_articulated_target_velocity(vel)
                art.target_vel = vel
            else:
                art.target_vel = None
            effort_targets = payload.get("dof_effort_target")
            if effort_targets:
                effort = np.zeros(art.num_dofs - art.base_dofs, dtype=np.float64)
                for jn, value in effort_targets.items():
                    idx = art.joint_dof_index.get(jn)
                    if idx is None:
                        raise KeyError(f"[superdex] robot '{name}' has no joint '{jn}' (dof_effort_target)")
                    effort[idx - art.base_dofs] = float(value)
                art.effort = effort
            elif art.effort is not None:
                art.effort = None

    # ------------------------------------------------------------------ state I/O
    def _rigid_root_state(self, rigid: _Rigid) -> np.ndarray:
        pos, quat = _wxyz_from_transform(rigid.actor.get_root_transform())
        if rigid.is_static:
            lin = ang = np.zeros(3)
        else:
            lin = np.asarray(rigid.actor.get_linear_velocity(), dtype=np.float64)
            ang = np.asarray(rigid.actor.get_angular_velocity(), dtype=np.float64)
        return np.concatenate([pos, quat, lin, ang])

    def _link_states(self, art: _Articulation) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return (root_state (13,), per-link states in SuperDex link order)."""
        transforms = sdp.DynamicArrayTransformRT(len(art.link_names))
        art.actor.get_articulated_link_transforms(transforms)
        link_states = []
        for i, link_actor in enumerate(art.link_actors):
            pos, quat = _wxyz_from_transform(transforms[i])
            try:
                lin = np.asarray(link_actor.get_linear_velocity(), dtype=np.float64)
                ang = np.asarray(link_actor.get_angular_velocity(), dtype=np.float64)
            except Exception as exc:  # a welded root link carries no velocity component
                log.debug(f"[superdex] no velocity on link {art.link_names[i]!r}: {exc}")
                lin = ang = np.zeros(3)
            link_states.append(np.concatenate([pos, quat, lin, ang]))
        # ``get_root_transform`` is the *articulation root frame*; with a FREE base joint the base motion
        # lives in the first six DoFs, so the world pose of link 0 is the only correct root pose.
        return link_states[0].copy(), link_states

    def _joint_arrays(self, art: _Articulation) -> tuple[np.ndarray, np.ndarray]:
        pose = sdp.DynamicArrayReal(art.num_dofs)
        vel = sdp.DynamicArrayReal(art.num_dofs)
        art.actor.get_articulated_pose(pose)
        art.actor.get_articulated_joint_velocities(vel)
        return np.asarray(pose, dtype=np.float64), np.asarray(vel, dtype=np.float64)

    def _articulation_state(self, art: _Articulation, cls):
        root_state, link_states = self._link_states(art)
        body_names = self._get_body_names(art.cfg.name, sort=True)
        body_order = [art.link_names.index(n) for n in body_names]
        body_state = np.stack([link_states[i] for i in body_order]) if body_order else np.zeros((0, 13))
        pose, vel = self._joint_arrays(art)
        joint_names = self._get_joint_names(art.cfg.name, sort=True)
        idx = [art.joint_dof_index[n] for n in joint_names]
        kwargs = dict(
            root_state=torch.from_numpy(root_state).float().unsqueeze(0),
            body_names=body_names,
            body_state=torch.from_numpy(body_state).float().unsqueeze(0),
            joint_pos=torch.from_numpy(pose[idx]).float().unsqueeze(0),
            joint_vel=torch.from_numpy(vel[idx]).float().unsqueeze(0),
        )
        if cls is RobotState:
            kwargs["joint_pos_target"] = (
                torch.from_numpy(art.target_pose[idx]).float().unsqueeze(0) if art.target_pose is not None else None
            )
            kwargs["joint_vel_target"] = (
                torch.from_numpy(art.target_vel[idx]).float().unsqueeze(0) if art.target_vel is not None else None
            )
            effort = None
            if art.controlled and self._control_mode == "pd" and art.last_tau is not None:
                effort = torch.from_numpy(art.last_tau[idx]).float().unsqueeze(0)
            elif art.controlled:
                try:
                    force = np.asarray(art.actor.get_articulated_controller_force(), dtype=np.float64)
                    if force.shape[0] == art.num_dofs:
                        effort = torch.from_numpy(force[idx]).float().unsqueeze(0)
                except Exception as exc:  # controller force readback is best-effort
                    log.debug(f"[superdex] controller force readback unavailable for '{art.cfg.name}': {exc}")
                    effort = None
            kwargs["joint_effort_target"] = effort
        return cls(**kwargs)

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        object_states: dict[str, ObjectState] = {}
        for obj in self.objects:
            if obj.name in self._articulations:
                object_states[obj.name] = self._articulation_state(self._articulations[obj.name], ObjectState)
            else:
                root = self._rigid_root_state(self._rigids[obj.name])
                object_states[obj.name] = ObjectState(root_state=torch.from_numpy(root).float().unsqueeze(0))
        robot_states: dict[str, RobotState] = {}
        for robot in self.robots:
            robot_states[robot.name] = self._articulation_state(self._articulations[robot.name], RobotState)
        camera_states: dict[str, CameraState] = {}
        if self._renderer is not None:
            self._push_render_poses()
            for cam in self.cameras:
                rgb, depth = self._renderer.render(cam, pose=self._mounted_camera_pose(cam))
                camera_states[cam.name] = CameraState(
                    rgb=torch.from_numpy(rgb).unsqueeze(0) if "rgb" in cam.data_types else None,
                    depth=torch.from_numpy(depth).unsqueeze(0) if "depth" in cam.data_types else None,
                )
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras={})

    def _set_states(self, states: DictStateBatch, env_ids: list[int] | None = None) -> None:
        if len(states) != 1:
            raise ValueError(f"[superdex] single-env handler got {len(states)} env states")
        objects, robots = states[0].get("objects", {}), states[0].get("robots", {})
        clash = set(objects) & set(robots)
        if clash:
            raise KeyError(f"[superdex] set_states: names used for both an object and a robot: {sorted(clash)}")
        for name, obj_state in {**objects, **robots}.items():
            if name in self._rigids:
                self._set_rigid_state(self._rigids[name], obj_state)
            elif name in self._articulations:
                self._set_articulation_state(self._articulations[name], obj_state)
            else:
                raise KeyError(f"[superdex] set_states: unknown object '{name}'")
        self._push_render_poses()

    def _set_rigid_state(self, rigid: _Rigid, obj_state) -> None:
        if "pos" in obj_state or "rot" in obj_state:
            cur_pos, cur_quat = _wxyz_from_transform(rigid.actor.get_root_transform())
            pos = obj_state.get("pos", cur_pos)
            rot = obj_state.get("rot", cur_quat)
            rigid.actor.set_root_transform(_transform_from_wxyz(pos, rot))
        if not rigid.is_static:
            # Restore recorded velocities so a state round-trips; missing keys mean rest.
            lin = np.asarray(obj_state.get("vel", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
            ang = np.asarray(obj_state.get("ang_vel", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
            rigid.actor.set_velocity(lin, ang)

    def _set_articulation_state(self, art: _Articulation, obj_state) -> None:
        # Re-anchor the articulation root at the current world pose of link 0 (they differ once a free
        # base has moved), then overwrite with the requested pos/rot. Base DoFs are zeroed below.
        cur_pos, cur_quat = self._link_states(art)[0][:3], self._link_states(art)[0][3:7]
        pos = obj_state.get("pos", cur_pos)
        rot = obj_state.get("rot", cur_quat)
        art.actor.set_root_transform(_transform_from_wxyz(pos, rot))
        pose, _ = self._joint_arrays(art)
        dof_pos = obj_state.get("dof_pos") or {}
        for jn, value in dof_pos.items():
            idx = art.joint_dof_index.get(jn)
            if idx is None:
                raise KeyError(f"[superdex] set_states: '{art.cfg.name}' has no joint '{jn}'")
            pose[idx] = float(_np(value))
        if art.base_dofs:
            pose[: art.base_dofs] = 0.0  # base offset is carried by the root transform
        art.actor.set_articulated_pose_from_joints(pose)
        vel = np.zeros(art.num_dofs)
        for jn, value in (obj_state.get("dof_vel") or {}).items():
            idx = art.joint_dof_index.get(jn)
            if idx is None:
                raise KeyError(f"[superdex] set_states: '{art.cfg.name}' has no joint '{jn}' (dof_vel)")
            vel[idx] = float(_np(value))
        art.actor.set_articulated_joint_velocities(vel)
        if art.controlled:
            art.target_pose = pose.copy()
            if self._control_mode == "implicit":
                art.actor.set_articulated_target_pose(art.target_pose)

    # ------------------------------------------------------------------ names
    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        art = self._articulations.get(obj_name)
        if art is None:
            return []
        return sorted(art.joint_names) if sort else list(art.joint_names)

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        art = self._articulations.get(obj_name)
        if art is None:
            return [obj_name] if obj_name in self._rigids else []
        return sorted(art.link_names) if sort else list(art.link_names)

    # ------------------------------------------------------------------ rendering helpers
    def _mounted_camera_pose(self, cam) -> np.ndarray | None:
        """World pose (4x4) of a camera mounted on a link, or None for a world-fixed camera.

        Same convention as the MuJoCo backend: ``mount_pos``/``mount_quat`` (wxyz) are the camera frame
        in the link frame, and the camera looks down its -Z axis with +Y up (MuJoCo == OpenGL).
        """
        if cam.mount_to is None:
            return None
        art = self._articulations.get(cam.mount_to)
        link_name = str(cam.mount_link).split("/")[-1]
        if art is not None:
            if link_name not in art.link_names:
                raise KeyError(f"[superdex] camera '{cam.name}': '{cam.mount_to}' has no link '{link_name}'")
            transforms = sdp.DynamicArrayTransformRT(len(art.link_names))
            art.actor.get_articulated_link_transforms(transforms)
            world_from_link = _matrix_from_transform(transforms[art.link_names.index(link_name)])
        else:
            world_from_link = _matrix_from_transform(self._rigids[cam.mount_to].actor.get_root_transform())
        return world_from_link @ _matrix_from_transform(_transform_from_wxyz(cam.mount_pos, cam.mount_quat))

    def get_contact_forces(self) -> torch.Tensor:
        """Net contact force per body of the first robot, ``(num_bodies, 3)`` in ``get_body_names`` order.

        Backs :class:`metasim.queries.contact_force.ContactForces`. SuperDex 1.0 exposes contact
        queries only on actors that carry contact sample points: standalone mesh rigid actors do, but
        articulation links and the implicit ground plane do not. The force on a robot link is therefore
        assembled from the *other side*: every dynamic rigid object registers a TOTAL_CONTACT_FORCE
        query and reports the force it receives from each link (``get_contact_force_from_actor_world``),
        which is negated onto the link. Contacts with the ground plane, static objects and the
        robot's own links are **not** observable this way; a one-time warning says so.
        """
        if not self.robots:
            return torch.zeros((0, 3))
        art = self._articulations[self.robots[0].name]
        if not art.contact_queries:
            dynamic = [r for r in self._rigids.values() if not r.is_static]
            for rigid in dynamic:
                art.contact_queries.append((rigid, rigid.actor.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)))
            self._contact_queries_fresh = False
            log.warning(
                f"[superdex] ContactForces for '{art.cfg.name}': measured through the {len(dynamic)} dynamic rigid "
                "object(s) it touches; ground-plane, static-object and self contacts are not observable "
                "(SuperDex 1.0 has no contact sample points on articulation links)."
            )
        order = [art.link_names.index(n) for n in self._get_body_names(art.cfg.name, sort=True)]
        forces = np.zeros((len(order), 3), dtype=np.float64)
        if not self._contact_queries_fresh:
            # query results only exist after the next simulation step
            return torch.from_numpy(forces).float()
        for rigid, _handle in art.contact_queries:
            for row, i in enumerate(order):
                f_on_obj = np.asarray(
                    rigid.actor.get_contact_force_from_actor_world(art.link_actors[i]), dtype=np.float64
                )
                forces[row] -= f_on_obj
        return torch.from_numpy(forces).float()

    def _push_render_poses(self) -> None:
        if self._renderer is None:
            return
        for name, rigid in self._rigids.items():
            self._renderer.set_body_pose(name, name, _matrix_from_transform(rigid.actor.get_root_transform()))
        for name, art in self._articulations.items():
            transforms = sdp.DynamicArrayTransformRT(len(art.link_names))
            art.actor.get_articulated_link_transforms(transforms)
            for i, link in enumerate(art.link_names):
                self._renderer.set_body_pose(name, link, _matrix_from_transform(transforms[i]))
