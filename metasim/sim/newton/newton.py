from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger as log

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

import math
from collections import defaultdict

import newton
import numpy as np
import scipy.spatial.transform as tr
import warp as wp
from newton import Contacts
from newton._src.sim.articulation import eval_fk
from newton._src.sim.joints import JointType
from newton.sensors import SensorContact, SensorTiledCamera, populate_contacts
from newton.solvers import SolverMuJoCo
from newton.viewer import ViewerGL

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler
from metasim.types import Action
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, state_tensor_to_nested


def wp2torch(arr: wp.array, dtype=torch.float32) -> torch.Tensor:
    """Convert Warp array to PyTorch tensor on the same device."""
    if arr is None:
        return None
    # Use DLPack for zero-copy conversion when possible
    return torch.from_dlpack(arr).to(dtype)


def torch2wp(tensor: torch.Tensor, dtype=None) -> wp.array:
    """Convert PyTorch tensor to Warp array."""
    if tensor is None:
        return None
    if dtype is None:
        dtype = wp.float32
    return wp.from_torch(tensor, dtype=dtype)


class NewtonHandler(BaseSimHandler):
    """Newton physics simulator handler using MuJoCo Warp solver.

    This handler implements the MetaSim BaseSimHandler interface for the Newton
    physics engine. It uses Newton's world grouping feature for parallel environments
    and the MuJoCo Warp solver for high-fidelity articulated body dynamics.
    """

    def __init__(
        self,
        scenario: ScenarioCfg,
        optional_queries: dict[str, BaseQueryType] | None = None,
    ):
        super().__init__(scenario, optional_queries)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Newton model and state
        self._model: newton.Model | None = None
        self._state_0: newton.State | None = None
        self._state_1: newton.State | None = None
        self._control: newton.Control | None = None
        self._solver: SolverMuJoCo | None = None
        self._viewer = None
        self._sim_time = 0.0
        self._use_mujoco_contacts = False

        # Tiled camera sensors (decoupled from ViewerGL window size)
        self._camera_groups: list[dict] = []
        self._use_tiled_camera = False
        self._shape_color_overrides: dict[int, np.ndarray] = {}

        # Contact sensor for contact force queries
        self._contacts: Contacts | None = None
        self._contact_sensor: SensorContact | None = None

        # Caches for efficient lookups
        self._joint_name_to_id: dict[str, dict[str, int]] = {}
        self._body_name_to_id: dict[str, dict[str, int]] = {}
        self._robot_joint_ids: dict[str, list[int]] = {}
        self._robot_body_ids: dict[str, list[int]] = {}
        self._body_children: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._body_child_to_joint: dict[int, dict[int, int]] = defaultdict(dict)
        self._obj_body_indices: dict[int, dict[str, list[int]]] = defaultdict(dict)
        self._obj_joint_indices: dict[int, dict[str, list[int]]] = defaultdict(dict)
        self._obj_joint_name_cache: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)

        # Gravity handling
        self._gravity_disabled_body_ids: dict[int, list[int]] = defaultdict(list)
        self._gravity_compensation_enabled = False
        self._gravity_vec = None

        # Actions cache
        self._actions_cache: list[Action] | torch.Tensor | np.ndarray = []

        # Decimation for substeps
        if scenario.decimation is not None:
            self.decimation = scenario.decimation
        else:
            self.decimation = 1

        log.info(f"NewtonHandler initialized for {self.num_envs} environments")

    def launch(self) -> None:
        """Initialize Newton model, allocate states, and create solver."""
        log.info("Launching Newton simulation...")

        # Build the Newton model from scenario configuration
        self._build_model()

        # Build name-to-ID caches for joint/body lookups
        self._build_name_caches()
        # Precompute per-env index caches for fast state extraction
        self._build_state_index_caches()

        # Apply foot friction overrides based on ground friction
        self._apply_foot_friction_overrides()

        # Apply global gravity from scenario
        self._apply_gravity_settings()

        # Build gravity compensation map for robots with gravity disabled
        self._build_gravity_compensation()

        # Apply actuator gains and limits before creating the solver/control
        self._apply_actuator_settings()

        # Create states (double-buffering for solver) and control
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # Apply default root poses and joint positions from scenario
        self._apply_default_state()

        # Create MuJoCo solver
        sim_params = self.scenario.sim_params
        use_mujoco_contacts = getattr(sim_params, "newton_use_mujoco_contacts", None)
        if use_mujoco_contacts is None:
            use_mujoco_contacts = False

        if use_mujoco_contacts:
            self._solver = SolverMuJoCo(
                self._model,
                njmax=sim_params.njmax,
                nconmax=sim_params.nconmax,
                use_mujoco_contacts=use_mujoco_contacts,
            )
        else:  # A weird bug that if passes use_mujoco_contacts=False the contact number would be super large causing bugs
            self._solver = SolverMuJoCo(
                self._model,
                njmax=sim_params.njmax,
                nconmax=sim_params.nconmax,
            )
        self._use_mujoco_contacts = bool(use_mujoco_contacts)

        # Initialize Contacts object for contact force queries
        nconmax = self._resolve_contact_capacity(sim_params.nconmax)
        self._contacts = Contacts(
            rigid_contact_max=nconmax,
            soft_contact_max=0,
            device=self._device,
        )
        self._ensure_contact_buffers()

        # Initialize Viewer for GUI rendering (optional)
        self._viewer = None
        headless = getattr(self.scenario, "headless", True)
        if not headless:
            if self.scenario.cameras:
                max_w = max(c.width for c in self.scenario.cameras)
                max_h = max(c.height for c in self.scenario.cameras)
            else:
                max_w, max_h = 1280, 720

            self._viewer = ViewerGL(width=max_w, height=max_h, headless=False)
            self._viewer.set_model(self._model)
            spacing = self.scenario.env_spacing
            self._viewer.set_world_offsets((spacing, spacing, 0.0))
            if self._viewer.ui is None or not self._viewer.ui.is_available:
                log.warning("Newton Viewer UI is unavailable. Install `imgui-bundle` to enable the left panel.")

        # Initialize tiled camera sensors (decoupled from ViewerGL window size)
        self._init_tiled_cameras()

        log.info(f"Newton launched with {self.num_envs} worlds, solver={self._solver.__class__.__name__}")

        return super().launch()

    def _apply_default_state(self) -> None:
        """Apply scenario default poses and joint positions to the model/state/control."""
        if self._model is None or self._state_0 is None:
            return

        default_states: list[dict] = []
        for _ in range(self.num_envs):
            env_state = {"robots": {}, "objects": {}}

            for robot in self.robots:
                robot_state = {}
                if getattr(robot, "default_position", None) is not None:
                    robot_state["pos"] = robot.default_position
                if getattr(robot, "default_orientation", None) is not None:
                    robot_state["rot"] = robot.default_orientation
                if getattr(robot, "default_joint_positions", None):
                    robot_state["dof_pos"] = robot.default_joint_positions
                if robot_state:
                    env_state["robots"][robot.name] = robot_state

            for obj in self.objects:
                obj_state = {}
                if getattr(obj, "default_position", None) is not None:
                    obj_state["pos"] = obj.default_position
                if getattr(obj, "default_orientation", None) is not None:
                    obj_state["rot"] = obj.default_orientation
                if isinstance(obj, ArticulationObjCfg) and getattr(obj, "default_joint_positions", None):
                    obj_state["dof_pos"] = obj.default_joint_positions
                if obj_state:
                    env_state["objects"][obj.name] = obj_state

            default_states.append(env_state)

        self._set_states(default_states, env_ids=list(range(self.num_envs)))

        # Keep the alternate state buffer consistent
        if self._state_1 is not None:
            if self._state_0.body_q is not None:
                self._state_1.body_q.assign(self._state_0.body_q)
            if self._state_0.body_qd is not None:
                self._state_1.body_qd.assign(self._state_0.body_qd)
            if self._state_0.joint_q is not None:
                self._state_1.joint_q.assign(self._state_0.joint_q)
            if self._state_0.joint_qd is not None:
                self._state_1.joint_qd.assign(self._state_0.joint_qd)

    def _configure_builder_defaults(self, builder: newton.ModelBuilder) -> None:
        """Align Newton builder defaults with expected PhysX-like defaults."""
        # Newton defaults: mu=0.5, torsional=0.25, rolling=0.0005. PhysX defaults are closer to mu=1.0 and 0 for
        # torsional/rolling friction. Update defaults to reduce cross-simulator mismatch.
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.torsional_friction = 0.0
        builder.default_shape_cfg.rolling_friction = 0.0

        # Use detailed inertia validation/correction to avoid MuJoCo compile errors
        # for URDFs with invalid inertia tensors.
        builder.validate_inertia_detailed = True
        builder.bound_inertia = 1e-6

        # Map contact offset to Newton's shape thickness/contact margin and builder margin.
        contact_offset = getattr(self.scenario.sim_params, "contact_offset", None)
        if contact_offset is not None:
            offset = max(0.0, float(contact_offset))
            builder.default_shape_cfg.thickness = offset
            builder.default_shape_cfg.contact_margin = offset
            builder.rigid_contact_margin = offset

    def _get_ground_shape_cfg(self, builder: newton.ModelBuilder) -> newton.ModelBuilder.ShapeConfig:
        """Create a ground shape config that honors ScenarioCfg ground friction."""
        cfg = builder.default_shape_cfg.copy()
        ground_cfg = getattr(self.scenario, "ground", None)
        if ground_cfg is None:
            return cfg

        # Prefer dynamic friction for sliding, fallback to static friction.
        mu = getattr(ground_cfg, "dynamic_friction", None)
        if mu is None:
            mu = getattr(ground_cfg, "static_friction", None)
        if mu is not None:
            cfg.mu = float(mu)

        return cfg

    def _apply_foot_friction_overrides(self) -> None:
        """Apply ground friction to foot shapes based on robot feet_links."""
        if self._model is None or self._model.shape_material_mu is None:
            return

        ground_cfg = getattr(self.scenario, "ground", None)
        if ground_cfg is None:
            return

        mu = getattr(ground_cfg, "dynamic_friction", None)
        if mu is None:
            mu = getattr(ground_cfg, "static_friction", None)
        if mu is None:
            return

        mu = float(mu)
        shape_mu = self._model.shape_material_mu.numpy()
        shape_torsion = (
            self._model.shape_material_torsional_friction.numpy()
            if self._model.shape_material_torsional_friction is not None
            else None
        )
        shape_rolling = (
            self._model.shape_material_rolling_friction.numpy()
            if self._model.shape_material_rolling_friction is not None
            else None
        )

        shape_indices: set[int] = set()
        for env_id in range(self.num_envs):
            for robot in self.robots:
                foot_tokens = getattr(robot, "feet_links", None)
                if not foot_tokens:
                    continue
                for body_id in self._get_body_indices(env_id, robot.name):
                    body_name = self._model.body_key[body_id]
                    if not any(token in body_name for token in foot_tokens):
                        continue
                    for shape_idx in self._model.body_shapes.get(body_id, []):
                        shape_indices.add(shape_idx)

        if not shape_indices:
            return

        for shape_idx in shape_indices:
            if 0 <= shape_idx < len(shape_mu):
                shape_mu[shape_idx] = mu
                if shape_torsion is not None:
                    shape_torsion[shape_idx] = 0.0
                if shape_rolling is not None:
                    shape_rolling[shape_idx] = 0.0

        self._model.shape_material_mu.assign(shape_mu)
        if shape_torsion is not None:
            self._model.shape_material_torsional_friction.assign(shape_torsion)
        if shape_rolling is not None:
            self._model.shape_material_rolling_friction.assign(shape_rolling)

    def _build_model(self) -> None:
        """Build Newton model from scenario configuration."""
        builder = newton.ModelBuilder()
        self._configure_builder_defaults(builder)

        # dt is set during solver.step()

        # Add ground plane (global, shared across all worlds)
        builder.current_world = -1
        self._shape_color_overrides = {}
        ground_shape_cfg = self._get_ground_shape_cfg(builder)
        ground_shape_idx = builder.add_ground_plane(cfg=ground_shape_cfg)
        # Give the ground a neutral gray to avoid pure white output
        self._shape_color_overrides[ground_shape_idx] = np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32)

        # Add robots and objects for each environment (reuse a single template builder)
        self._obj_to_root_body = defaultdict(dict)

        env_builder = newton.ModelBuilder()
        self._configure_builder_defaults(env_builder)

        # Track local indices for this env template
        env_map = {}
        env_shape_colors: dict[int, np.ndarray] = {}

        # Add robot(s) once to template
        for robot in self.robots:
            start_count = len(env_builder.body_mass)
            self._add_robot_to_builder(env_builder, robot)
            end_count = len(env_builder.body_mass)
            if end_count > start_count:
                env_map[robot.name] = start_count

        # Add objects once to template
        for obj in self.objects:
            start_count = len(env_builder.body_mass)
            self._add_object_to_builder(env_builder, obj, env_shape_colors)
            end_count = len(env_builder.body_mass)
            if end_count > start_count:
                env_map[obj.name] = start_count

        # Instantiate identical worlds from template
        for env_id in range(self.num_envs):
            global_offset = len(builder.body_mass)
            shape_offset = builder.shape_count
            builder.add_world(env_builder)

            for name, local_idx in env_map.items():
                self._obj_to_root_body[env_id][name] = global_offset + local_idx

            for local_shape_idx, color in env_shape_colors.items():
                self._shape_color_overrides[shape_offset + local_shape_idx] = color

        # Finalize model
        self._model = builder.finalize(device=self._device)
        log.debug(f"Newton model built: {self._model.body_count} bodies, {self._model.joint_count} joints")

    def _add_robot_to_builder(self, builder: newton.ModelBuilder, robot) -> None:
        """Add a robot to the model builder using URDF import."""
        # Try to import URDF
        urdf_path = robot.urdf_path
        if urdf_path is None:
            log.error(f"Robot {robot.name} has no URDF path defined")
            raise ValueError(f"Robot {robot.name} requires urdf_path for Newton")

        # Parse URDF into builder
        ret = builder.add_urdf(
            urdf_path,
            xform=wp.transform(
                wp.vec3(*robot.default_position),
                wp.quat(*self._wxyz_to_xyzw(robot.default_orientation)),
            ),
            floating=not robot.fix_base_link,
            enable_self_collisions=robot.enabled_self_collisions,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=getattr(robot, "collapse_fixed_joints", False),
        )

    def _normalize_color(self, color) -> np.ndarray | None:
        """Normalize color list/tuple to RGBA float array in [0, 1]."""
        if color is None:
            return None
        col = np.array(color, dtype=np.float32).reshape(-1)
        if col.size not in (3, 4):
            return None
        if col.max() > 1.0:
            col = col / 255.0
        if col.size == 3:
            col = np.concatenate([col, np.array([1.0], dtype=np.float32)])
        return col.astype(np.float32)

    def _add_object_to_builder(
        self, builder: newton.ModelBuilder, obj, shape_colors: dict[int, np.ndarray] | None = None
    ) -> None:
        """Add an object to the model builder."""
        if isinstance(obj, PrimitiveCubeCfg):
            # Add box shape
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*obj.default_position),
                    wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                ),
                key=obj.name,
            )
            shape_idx = builder.add_shape_box(
                body=body,
                hx=obj.half_size[0],
                hy=obj.half_size[1],
                hz=obj.half_size[2],
            )
            if shape_colors is not None:
                color = self._normalize_color(getattr(obj, "color", None))
                if color is not None:
                    shape_colors[shape_idx] = color
        elif isinstance(obj, PrimitiveSphereCfg):
            # Add sphere shape
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*obj.default_position),
                    wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                ),
                key=obj.name,
            )
            shape_idx = builder.add_shape_sphere(body=body, radius=obj.radius)
            if shape_colors is not None:
                color = self._normalize_color(getattr(obj, "color", None))
                if color is not None:
                    shape_colors[shape_idx] = color
        elif isinstance(obj, PrimitiveCylinderCfg):
            # Add capsule shape (Newton uses capsules, approximate cylinder)
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*obj.default_position),
                    wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                ),
                key=obj.name,
            )
            shape_idx = builder.add_shape_capsule(body=body, radius=obj.radius, half_height=obj.height / 2)
            if shape_colors is not None:
                color = self._normalize_color(getattr(obj, "color", None))
                if color is not None:
                    shape_colors[shape_idx] = color
        elif isinstance(obj, ArticulationObjCfg):
            # Load articulated object from URDF
            urdf_path = obj.urdf_path
            if urdf_path:
                builder.add_urdf(
                    urdf_path,
                    xform=wp.transform(
                        wp.vec3(*obj.default_position),
                        wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                    ),
                    floating=not obj.fix_base_link,
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=getattr(obj, "collapse_fixed_joints", False),
                )
        else:
            # Try URDF for other file-based objects
            if hasattr(obj, "urdf_path") and obj.urdf_path:
                builder.add_urdf(
                    obj.urdf_path,
                    xform=wp.transform(
                        wp.vec3(*obj.default_position),
                        wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                    ),
                    floating=not getattr(obj, "fix_base_link", True),
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=getattr(obj, "collapse_fixed_joints", False),
                )

    def _build_name_caches(self) -> None:
        """Build caches for looking up body and joint indices by name."""
        self._body_name_cache = defaultdict(dict)
        self._joint_name_cache = defaultdict(dict)
        self._body_children = defaultdict(lambda: defaultdict(list))
        self._body_child_to_joint = defaultdict(dict)
        self._obj_body_indices = defaultdict(dict)
        self._obj_joint_indices = defaultdict(dict)
        self._obj_joint_name_cache = defaultdict(dict)
        self._obj_root_body_ids: dict[str, torch.Tensor] = {}
        self._obj_sorted_body_ids: dict[str, torch.Tensor] = {}
        self._obj_sorted_body_names: dict[str, list[str]] = {}
        self._obj_sorted_joint_names: dict[str, list[str]] = {}
        self._obj_joint_q_idx: dict[str, torch.Tensor] = {}
        self._obj_joint_qd_idx: dict[str, torch.Tensor] = {}
        self._obj_joint_valid: dict[str, torch.Tensor] = {}
        self._obj_root_joint_idx: dict[str, torch.Tensor] = {}
        self._obj_root_joint_type: dict[str, torch.Tensor] = {}
        self._joint_q_starts_t: torch.Tensor | None = None
        self._joint_qd_starts_t: torch.Tensor | None = None
        self._joint_types_t: torch.Tensor | None = None
        self._root_q_offsets: torch.Tensor | None = None
        self._root_qd_offsets: torch.Tensor | None = None

        # DEBUG: Print all body names
        # print(f"DEBUG: Model Body Names: {self._model.body_key}") # Commented out to be safe, I'll rely on add_urdf return checks

        self._body_parent_joint = {}  # body_idx -> joint_idx

        # Access model data on CPU for building caches
        body_keys = self._model.body_key
        body_worlds = self._model.body_world.numpy()

        for i, key in enumerate(body_keys):
            world = body_worlds[i]
            if world != -1:
                self._body_name_cache[world][key] = i

        joint_keys = self._model.joint_key
        joint_worlds = self._model.joint_world.numpy()
        joint_parents = self._model.joint_parent.numpy()
        joint_children = self._model.joint_child.numpy()

        for i, key in enumerate(joint_keys):
            world = joint_worlds[i]
            if world != -1:
                self._joint_name_cache[world][key] = i

        # Build body -> parent joint map
        for i, child_idx in enumerate(joint_children):
            if child_idx >= 0:
                self._body_parent_joint[child_idx] = i
                world = joint_worlds[i]
                if world != -1:
                    self._body_child_to_joint[world][child_idx] = i
                    parent_idx = joint_parents[i]
                    if parent_idx >= 0:
                        self._body_children[world][parent_idx].append(child_idx)

        # Access start indices for fast lookup
        self._joint_q_starts = self._model.joint_q_start.numpy()
        self._joint_qd_starts = self._model.joint_qd_start.numpy()
        self._joint_types = self._model.joint_type.numpy()

        # Build per-object joint name maps to disambiguate duplicate joint names
        for env_id in range(self.num_envs):
            for obj in (*self.robots, *self.objects):
                joint_map = {}
                for joint_idx in self._get_joint_indices(env_id, obj.name):
                    joint_name = self._model.joint_key[joint_idx]
                    if joint_name not in joint_map:
                        joint_map[joint_name] = joint_idx
                if joint_map:
                    self._obj_joint_name_cache[env_id][obj.name] = joint_map

    def _build_state_index_caches(self) -> None:
        """Precompute per-env indices for fast state extraction."""
        if self._model is None:
            return

        self._obj_root_body_ids = {}
        self._obj_sorted_body_ids = {}
        self._obj_sorted_body_names = {}
        self._obj_sorted_joint_names = {}
        self._obj_joint_q_idx = {}
        self._obj_joint_qd_idx = {}
        self._obj_joint_valid = {}
        self._obj_root_joint_idx = {}
        self._obj_root_joint_type = {}
        self._joint_q_starts_t = None
        self._joint_qd_starts_t = None
        self._joint_types_t = None
        self._root_q_offsets = None
        self._root_qd_offsets = None

        for obj in (*self.robots, *self.objects):
            name = obj.name
            root_ids = torch.full((self.num_envs,), -1, device=self._device, dtype=torch.long)
            root_joint_idx = torch.full((self.num_envs,), -1, device=self._device, dtype=torch.long)
            root_joint_type = torch.full((self.num_envs,), -1, device=self._device, dtype=torch.int32)
            sorted_body_ids_by_env: list[list[int]] = []
            body_len = None
            body_consistent = True

            for env_id in range(self.num_envs):
                root_idx = self._obj_to_root_body[env_id].get(name)
                if root_idx is not None:
                    root_ids[env_id] = int(root_idx)
                    joint_idx = self._body_parent_joint.get(root_idx)
                    if joint_idx is not None:
                        root_joint_idx[env_id] = int(joint_idx)
                        root_joint_type[env_id] = int(self._joint_types[joint_idx])

                body_ids = self._get_body_indices(env_id, name)
                if body_ids:
                    body_names = [self._model.body_key[idx] for idx in body_ids]
                    sorted_pairs = sorted(zip(body_names, body_ids), key=lambda pair: pair[0])
                    sorted_body_ids = [idx for _, idx in sorted_pairs]
                else:
                    sorted_body_ids = []

                if body_len is None:
                    body_len = len(sorted_body_ids)
                    if body_len and env_id == 0:
                        self._obj_sorted_body_names[name] = [n for n, _ in sorted_pairs]
                elif len(sorted_body_ids) != body_len:
                    body_consistent = False

                sorted_body_ids_by_env.append(sorted_body_ids)

            self._obj_root_body_ids[name] = root_ids
            self._obj_root_joint_idx[name] = root_joint_idx
            self._obj_root_joint_type[name] = root_joint_type
            if body_consistent and body_len and body_len > 0:
                self._obj_sorted_body_ids[name] = torch.tensor(
                    sorted_body_ids_by_env, device=self._device, dtype=torch.long
                )

            if not isinstance(obj, ArticulationObjCfg):
                continue

            joint_names = self._get_joint_names(name, sort=True)
            if not joint_names:
                continue
            self._obj_sorted_joint_names[name] = joint_names

            q_idx_by_env: list[list[int]] = []
            qd_idx_by_env: list[list[int]] = []
            valid_by_env: list[list[bool]] = []

            for env_id in range(self.num_envs):
                q_idx: list[int] = []
                qd_idx: list[int] = []
                valid: list[bool] = []

                for joint_name in joint_names:
                    joint_idx = self._get_joint_index(env_id, name, joint_name)
                    if joint_idx is None:
                        q_idx.append(0)
                        qd_idx.append(0)
                        valid.append(False)
                        continue
                    q_start = int(self._joint_q_starts[joint_idx])
                    qd_start = int(self._joint_qd_starts[joint_idx])
                    qd_end = int(self._joint_qd_starts[joint_idx + 1])
                    if qd_end - qd_start != 1:
                        q_idx.append(0)
                        qd_idx.append(0)
                        valid.append(False)
                        continue
                    q_idx.append(q_start)
                    qd_idx.append(qd_start)
                    valid.append(True)

                q_idx_by_env.append(q_idx)
                qd_idx_by_env.append(qd_idx)
                valid_by_env.append(valid)

            q_idx_tensor = torch.tensor(q_idx_by_env, device=self._device, dtype=torch.long)
            qd_idx_tensor = torch.tensor(qd_idx_by_env, device=self._device, dtype=torch.long)
            valid_tensor = torch.tensor(valid_by_env, device=self._device, dtype=torch.bool)

            if valid_tensor.numel() and (valid_tensor == valid_tensor[0]).all():
                valid_mask = valid_tensor[0]
            else:
                valid_mask = valid_tensor

            self._obj_joint_q_idx[name] = q_idx_tensor
            self._obj_joint_qd_idx[name] = qd_idx_tensor
            self._obj_joint_valid[name] = valid_mask

        if self._joint_q_starts is not None:
            self._joint_q_starts_t = torch.as_tensor(self._joint_q_starts, device=self._device, dtype=torch.long)
        if self._joint_qd_starts is not None:
            self._joint_qd_starts_t = torch.as_tensor(self._joint_qd_starts, device=self._device, dtype=torch.long)
        if self._joint_types is not None:
            self._joint_types_t = torch.as_tensor(self._joint_types, device=self._device, dtype=torch.int32)
        self._root_q_offsets = torch.arange(7, device=self._device, dtype=torch.long)
        self._root_qd_offsets = torch.arange(6, device=self._device, dtype=torch.long)

    def _apply_gravity_settings(self) -> None:
        """Apply scenario gravity to the Newton model."""
        if self._model is None:
            return
        gravity = getattr(self.scenario, "gravity", None)
        if gravity is None:
            return
        try:
            self._model.set_gravity(tuple(gravity))
        except Exception as exc:
            log.warning(f"Failed to set Newton gravity: {exc}")

    def _build_gravity_compensation(self) -> None:
        """Build list of body ids that should be gravity-compensated."""
        self._gravity_disabled_body_ids = defaultdict(list)
        self._gravity_compensation_enabled = False

        if self._model is None:
            return

        gravity = getattr(self.scenario, "gravity", None)
        if gravity is None:
            return

        for env_id in range(self.num_envs):
            body_ids: list[int] = []
            for robot in self.robots:
                if getattr(robot, "enabled_gravity", True):
                    continue
                body_ids.extend(self._get_body_indices(env_id, robot.name))

            if body_ids:
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for bid in body_ids:
                    if bid in seen:
                        continue
                    seen.add(bid)
                    deduped.append(bid)
                self._gravity_disabled_body_ids[env_id] = deduped
                self._gravity_compensation_enabled = True

        if self._gravity_compensation_enabled:
            self._gravity_vec = torch.tensor(gravity, dtype=torch.float32, device=self._device)

    def _apply_gravity_compensation(self) -> None:
        """Apply per-body gravity compensation for robots with gravity disabled."""
        if not self._gravity_compensation_enabled:
            return
        if self._state_0 is None or self._state_0.body_f is None:
            return
        if self._model is None or self._model.body_mass is None:
            return

        body_f = wp2torch(self._state_0.body_f)
        body_f.zero_()

        body_mass = wp2torch(self._model.body_mass)
        gravity = self._gravity_vec

        for body_ids in self._gravity_disabled_body_ids.values():
            for body_id in body_ids:
                m = body_mass[body_id]
                if m == 0:
                    continue
                body_f[body_id, 0:3] = -m * gravity

    def _apply_actuator_settings(self) -> None:
        """Apply actuator stiffness/damping/limits to the Newton model."""
        if self._model is None or self._model.joint_count == 0:
            return

        joint_target_ke = self._model.joint_target_ke.numpy()
        joint_target_kd = self._model.joint_target_kd.numpy()
        joint_armature = self._model.joint_armature.numpy()
        joint_effort_limit = self._model.joint_effort_limit.numpy()
        joint_velocity_limit = self._model.joint_velocity_limit.numpy()
        joint_target_pos = self._model.joint_target_pos.numpy()
        joint_target_vel = self._model.joint_target_vel.numpy()
        joint_q = self._model.joint_q.numpy()

        updated = False

        for env_id in range(self.num_envs):
            for robot in self.robots:
                if not isinstance(robot, RobotCfg) or not robot.actuators:
                    continue
                for joint_name, actuator in robot.actuators.items():
                    joint_idx = self._get_joint_index(env_id, robot.name, joint_name)
                    if joint_idx is None:
                        continue

                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end <= qd_start:
                        continue

                    control_mode = None
                    if robot.control_type:
                        control_mode = robot.control_type.get(joint_name)
                    is_effort = control_mode == "effort"

                    if is_effort:
                        # Effort-controlled joints should not have internal PD drives.
                        joint_target_ke[qd_start:qd_end] = 0.0
                        joint_target_kd[qd_start:qd_end] = 0.0
                        updated = True
                    else:
                        if actuator.stiffness is not None:
                            joint_target_ke[qd_start:qd_end] = actuator.stiffness
                            updated = True
                        if actuator.damping is not None:
                            joint_target_kd[qd_start:qd_end] = actuator.damping
                            updated = True
                    if actuator.armature is not None:
                        joint_armature[qd_start:qd_end] = actuator.armature
                        updated = True
                    if actuator.effort_limit_sim is not None:
                        joint_effort_limit[qd_start:qd_end] = actuator.effort_limit_sim
                        updated = True

                    vel_limit = (
                        actuator.velocity_limit_sim
                        if actuator.velocity_limit_sim is not None
                        else actuator.velocity_limit
                    )
                    if vel_limit is not None:
                        joint_velocity_limit[qd_start:qd_end] = vel_limit
                        updated = True

                    # Initialize target position from the current joint state for 1-DoF joints
                    if qd_end - qd_start == 1:
                        q_start = self._joint_q_starts[joint_idx]
                        joint_target_pos[qd_start] = joint_q[q_start]
                        joint_target_vel[qd_start] = 0.0
                        updated = True

        if updated:
            self._model.joint_target_ke.assign(joint_target_ke)
            self._model.joint_target_kd.assign(joint_target_kd)
            self._model.joint_armature.assign(joint_armature)
            self._model.joint_effort_limit.assign(joint_effort_limit)
            self._model.joint_velocity_limit.assign(joint_velocity_limit)
            self._model.joint_target_pos.assign(joint_target_pos)
            self._model.joint_target_vel.assign(joint_target_vel)

    def _get_body_indices(self, env_id: int, obj_name: str) -> list[int]:
        """Return body indices (root first) for an object in a given env."""
        cached = self._obj_body_indices[env_id].get(obj_name)
        if cached is not None:
            return cached

        root_idx = self._obj_to_root_body[env_id].get(obj_name)
        if root_idx is None:
            self._obj_body_indices[env_id][obj_name] = []
            return []

        body_ids = []
        stack = [root_idx]
        visited = set()

        children_map = self._body_children.get(env_id, {})
        while stack:
            body_idx = stack.pop()
            if body_idx in visited:
                continue
            visited.add(body_idx)
            body_ids.append(body_idx)
            for child_idx in children_map.get(body_idx, []):
                if child_idx not in visited:
                    stack.append(child_idx)

        self._obj_body_indices[env_id][obj_name] = body_ids
        return body_ids

    def _get_joint_indices(self, env_id: int, obj_name: str) -> list[int]:
        """Return joint indices (including root joint if any) for an object in a given env."""
        cached = self._obj_joint_indices[env_id].get(obj_name)
        if cached is not None:
            return cached

        body_ids = self._get_body_indices(env_id, obj_name)
        joint_ids = []
        joint_map = self._body_child_to_joint.get(env_id, {})
        for body_idx in body_ids:
            joint_idx = joint_map.get(body_idx)
            if joint_idx is not None:
                joint_ids.append(joint_idx)

        self._obj_joint_indices[env_id][obj_name] = joint_ids
        return joint_ids

    def _get_obj_joint_name_map(self, env_id: int, obj_name: str) -> dict[str, int]:
        """Return per-object joint name -> joint index mapping for a given env."""
        cached = self._obj_joint_name_cache.get(env_id, {}).get(obj_name)
        if cached is not None:
            return cached
        joint_map: dict[str, int] = {}
        if self._model is None:
            return joint_map
        for joint_idx in self._get_joint_indices(env_id, obj_name):
            joint_name = self._model.joint_key[joint_idx]
            if joint_name not in joint_map:
                joint_map[joint_name] = joint_idx
        self._obj_joint_name_cache[env_id][obj_name] = joint_map
        return joint_map

    def _get_joint_index(self, env_id: int, obj_name: str, joint_name: str) -> int | None:
        """Resolve a joint index for a specific object in a given env."""
        obj_map = self._get_obj_joint_name_map(env_id, obj_name)
        return obj_map.get(joint_name)

    def _collect_joint_names(self, env_id: int, obj_name: str) -> list[str]:
        """Collect 1-DoF joint names for an object in a given env."""
        names: list[str] = []
        for joint_idx in self._get_joint_indices(env_id, obj_name):
            qd_start = self._joint_qd_starts[joint_idx]
            qd_end = self._joint_qd_starts[joint_idx + 1]
            if qd_end - qd_start == 1:
                names.append(self._model.joint_key[joint_idx])
        return names

    @staticmethod
    def _reorder_quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Reorder quaternion from xyzw to wxyz for torch tensors."""
        return torch.stack(
            [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
            dim=-1,
        )

    @staticmethod
    def _reorder_quat_wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Reorder quaternion from wxyz to xyzw for torch tensors."""
        return torch.stack(
            [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
            dim=-1,
        )

    def _pack_body_state(self, body_q: torch.Tensor, body_qd: torch.Tensor | None, body_ids: list[int]) -> torch.Tensor:
        """Pack body state into [pos, quat, lin_vel, ang_vel] for a list of body indices."""
        if not body_ids:
            return torch.zeros((0, 13), device=self._device)

        pos = body_q[body_ids, 0:3]
        quat = self._reorder_quat_xyzw_to_wxyz(body_q[body_ids, 3:7])

        if body_qd is None:
            lin_vel = torch.zeros_like(pos)
            ang_vel = torch.zeros_like(pos)
        else:
            lin_vel = body_qd[body_ids, 0:3]
            ang_vel = body_qd[body_ids, 3:6]

        return torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)

    def _pack_body_state_batch(
        self, body_q: torch.Tensor, body_qd: torch.Tensor | None, body_ids: torch.Tensor
    ) -> torch.Tensor:
        """Pack body state for batched indices (supports 1D or 2D index tensors)."""
        if body_ids.numel() == 0:
            return torch.zeros((*body_ids.shape, 13), device=self._device)

        pos = body_q[body_ids, 0:3]
        quat = self._reorder_quat_xyzw_to_wxyz(body_q[body_ids, 3:7])

        if body_qd is None:
            lin_vel = torch.zeros_like(pos)
            ang_vel = torch.zeros_like(pos)
        else:
            lin_vel = body_qd[body_ids, 0:3]
            ang_vel = body_qd[body_ids, 3:6]

        return torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)

    @staticmethod
    def _coerce_dof_values(value, dof_count: int) -> list[float] | None:
        """Normalize a DOF value into a list of floats matching dof_count."""
        if dof_count <= 0:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return [float(value.item())] if dof_count == 1 else None
            values = value.detach().cpu().numpy().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return [float(value)] if dof_count == 1 else None

        if len(values) == dof_count:
            return [float(v) for v in values]
        if len(values) == 1 and dof_count == 1:
            return [float(values[0])]
        return None

    def _get_world_up(self) -> np.ndarray:
        """Get world up axis as a unit vector."""
        up_axis = getattr(self._model, "up_axis", "Z") if self._model is not None else "Z"
        if isinstance(up_axis, int):
            axis_idx = int(up_axis)
        else:
            axis_idx = "XYZ".index(str(up_axis).upper())
        up = np.zeros(3, dtype=np.float32)
        up[axis_idx] = 1.0
        return up

    def _camera_pose_to_transform(
        self, pos: tuple[float, float, float], look_at: tuple[float, float, float] | None
    ) -> wp.transformf:
        """Build a camera-to-world transform for SensorTiledCamera."""
        pos_np = np.array(pos, dtype=np.float32)
        world_up = self._get_world_up()

        if look_at is None:
            # Default forward: pick a direction not parallel to world_up
            if abs(world_up[2]) > 0.9:
                forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            else:
                forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            target = pos_np + forward
        else:
            target = np.array(look_at, dtype=np.float32)

        forward = target - pos_np
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            forward = forward / norm

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Choose an arbitrary axis not parallel to forward
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(forward[0]) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, fallback)
            right_norm = np.linalg.norm(right)
        right = right / (right_norm + 1e-6)

        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-6)

        # Camera frame uses -Z as forward
        R = np.column_stack([right, up, -forward])
        quat_xyzw = tr.Rotation.from_matrix(R).as_quat()

        return wp.transformf(wp.vec3(*pos_np), wp.quat(*quat_xyzw))

    def _build_camera_transforms(self, cam_indices: list[int]) -> wp.array:
        """Build per-camera, per-world transforms for SensorTiledCamera."""
        transforms = []
        for cam_idx in cam_indices:
            cam_cfg = self.scenario.cameras[cam_idx]
            look_at = cam_cfg.look_at if getattr(cam_cfg, "look_at", None) else None
            cam_tf = self._camera_pose_to_transform(cam_cfg.pos, look_at)
            transforms.append([cam_tf for _ in range(self.num_envs)])
        return wp.array(transforms, dtype=wp.transformf, device=self._model.device if self._model else None)

    def _init_tiled_cameras(self) -> None:
        """Initialize SensorTiledCamera groups for decoupled camera rendering."""
        self._camera_groups = []
        self._use_tiled_camera = False

        if not self.scenario.cameras or self._model is None:
            return

        # Group cameras by resolution since SensorTiledCamera requires uniform size per sensor
        cam_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, cam_cfg in enumerate(self.scenario.cameras):
            cam_groups[(cam_cfg.width, cam_cfg.height)].append(idx)

        try:
            with wp.ScopedDevice(self._model.device):
                for (width, height), cam_indices in cam_groups.items():
                    sensor = SensorTiledCamera(
                        model=self._model,
                        num_cameras=len(cam_indices),
                        width=width,
                        height=height,
                        options=SensorTiledCamera.Options(
                            default_light=True,
                            default_light_shadows=True,
                        ),
                    )
                    if self._shape_color_overrides:
                        colors = np.ones((self._model.shape_count, 4), dtype=np.float32)
                        for shape_idx, color in self._shape_color_overrides.items():
                            if 0 <= shape_idx < colors.shape[0]:
                                colors[shape_idx] = color
                        sensor.render_context.shape_colors = wp.array(colors, dtype=wp.vec4f)
                    fovs = [math.radians(self.scenario.cameras[i].vertical_fov) for i in cam_indices]
                    camera_rays = sensor.compute_pinhole_camera_rays(fovs)
                    color_image = sensor.create_color_image_output()
                    depth_image = sensor.create_depth_image_output()
                    camera_transforms = self._build_camera_transforms(cam_indices)

                    self._camera_groups.append({
                        "width": width,
                        "height": height,
                        "cam_indices": cam_indices,
                        "sensor": sensor,
                        "camera_rays": camera_rays,
                        "color_image": color_image,
                        "depth_image": depth_image,
                        "camera_transforms": camera_transforms,
                    })

            self._use_tiled_camera = True
            log.info(
                f"Initialized SensorTiledCamera for {len(self.scenario.cameras)} camera(s) "
                f"across {len(self._camera_groups)} resolution group(s)."
            )
        except Exception as e:
            log.warning(f"Failed to initialize SensorTiledCamera. Camera outputs disabled. Error: {e}")
            self._camera_groups = []
            self._use_tiled_camera = False

    def _render_tiled_cameras(self, env_ids: list[int]) -> dict:
        """Render cameras using SensorTiledCamera and return CameraState dict."""
        camera_states = {}
        if not self._camera_groups or self._state_0 is None or self._model is None:
            return camera_states

        num_worlds = self._model.num_worlds
        use_env_ids = env_ids if env_ids is not None else list(range(num_worlds))

        with wp.ScopedDevice(self._model.device):
            for group in self._camera_groups:
                sensor: SensorTiledCamera = group["sensor"]
                sensor.render(
                    self._state_0,
                    group["camera_transforms"],
                    group["camera_rays"],
                    color_image=group["color_image"],
                    depth_image=group["depth_image"],
                )

                color_np = group["color_image"].numpy() if group["color_image"] is not None else None
                depth_np = group["depth_image"].numpy() if group["depth_image"] is not None else None

                width = group["width"]
                height = group["height"]

                for local_idx, cam_idx in enumerate(group["cam_indices"]):
                    cam_cfg = self.scenario.cameras[cam_idx]
                    rgb_tensor = None
                    depth_tensor = None

                    if color_np is not None:
                        color_cam = color_np[:, local_idx, :].reshape(num_worlds, height, width)
                        r = (color_cam & 0xFF).astype(np.uint8)
                        g = ((color_cam >> 8) & 0xFF).astype(np.uint8)
                        b = ((color_cam >> 16) & 0xFF).astype(np.uint8)
                        rgb = np.stack([r, g, b], axis=-1)
                        rgb_tensor = torch.from_numpy(rgb).to(self._device)
                        if len(use_env_ids) != num_worlds:
                            rgb_tensor = rgb_tensor[use_env_ids]

                    if depth_np is not None:
                        depth_cam = depth_np[:, local_idx, :].reshape(num_worlds, height, width)
                        depth = depth_cam[..., None]
                        depth_tensor = torch.from_numpy(depth).to(self._device)
                        if len(use_env_ids) != num_worlds:
                            depth_tensor = depth_tensor[use_env_ids]

                    intrinsics = (
                        torch.tensor(cam_cfg.intrinsics, dtype=torch.float32, device=self.device)
                        .unsqueeze(0)
                        .expand(len(use_env_ids), -1, -1)
                    )

                    camera_states[cam_cfg.name] = CameraState(
                        rgb=rgb_tensor,
                        depth=depth_tensor,
                        intrinsics=intrinsics,
                    )

        return camera_states

    def _simulate(self) -> None:
        """Advance simulation by one step (with decimation substeps)."""
        dt = self.scenario.sim_params.dt if self.scenario.sim_params.dt else 0.005

        for _ in range(self.decimation):
            # Apply gravity compensation (per-body) if configured
            self._apply_gravity_compensation()

            # Generate contacts only when the solver requires external contacts
            if self._use_mujoco_contacts:
                contacts = self._contacts if self._contacts is not None else Contacts(0, 0, device=self._device)
            else:
                contacts = self._model.collide(self._state_0)

            # Step solver
            self._solver.step(
                self._state_0,
                self._state_1,
                self._control,
                contacts,
                dt=dt,
            )

            # Populate contacts with force data from solver (for contact force queries)
            if self._contacts is not None:
                populate_contacts(self._contacts, self._solver)

            # Swap state buffers
            self._state_0, self._state_1 = self._state_1, self._state_0
            self._sim_time += dt

        self._render_viewer()

    def _render_viewer(self) -> None:
        if self._viewer is None or self.headless or self._state_0 is None:
            return
        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self._state_0)
        self._viewer.end_frame()

    def render(self) -> None:
        """Render the current state to the Newton viewer."""
        self._render_viewer()

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        """Get current states of all robots and objects."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        robot_states = {}
        object_states = {}
        camera_states = {}

        # Extract robot states
        for robot in self.robots:
            robot_states[robot.name] = self._extract_robot_state(robot.name, env_ids)

        # Extract object states
        for obj in self.objects:
            object_states[obj.name] = self._extract_object_state(obj.name, env_ids)

        # Camera rendering using SensorTiledCamera (decoupled from GUI window size)
        if self._use_tiled_camera:
            camera_states = self._render_tiled_cameras(env_ids)

        extras = self.get_extra()
        return TensorState(
            objects=object_states,
            robots=robot_states,
            cameras=camera_states,
            extras=extras,
        )

    def _extract_robot_state(self, robot_name: str, env_ids: list[int]) -> RobotState:
        """Extract state for a robot across specified environments."""
        state = self._state_0

        body_q = wp2torch(state.body_q) if state is not None else None
        body_qd = wp2torch(state.body_qd) if state is not None else None
        joint_q = wp2torch(state.joint_q) if state is not None else None
        joint_qd = wp2torch(state.joint_qd) if state is not None else None

        joint_target_pos = (
            wp2torch(self._control.joint_target_pos) if self._control and self._control.joint_target_pos else None
        )
        joint_target_vel = (
            wp2torch(self._control.joint_target_vel) if self._control and self._control.joint_target_vel else None
        )
        joint_f = wp2torch(self._control.joint_f) if self._control and self._control.joint_f else None

        num_envs = len(env_ids)
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self._device, dtype=torch.long)
        else:
            env_ids_t = torch.tensor(env_ids, device=self._device, dtype=torch.long)
        root_state = torch.zeros(num_envs, 13, device=self._device)

        joint_names = self._obj_sorted_joint_names.get(robot_name)
        if joint_names is None:
            joint_names = self._get_joint_names(robot_name, sort=True)
        num_joints = len(joint_names)
        joint_pos = torch.zeros(num_envs, num_joints, device=self._device)
        joint_vel = torch.zeros(num_envs, num_joints, device=self._device)
        joint_pos_target = (
            torch.zeros(num_envs, num_joints, device=self._device) if joint_target_pos is not None else None
        )
        joint_vel_target = (
            torch.zeros(num_envs, num_joints, device=self._device) if joint_target_vel is not None else None
        )
        joint_effort_target = torch.zeros(num_envs, num_joints, device=self._device) if joint_f is not None else None

        body_state = None
        if body_q is not None and robot_name in self._obj_sorted_body_ids:
            body_ids_all = self._obj_sorted_body_ids[robot_name]
            if body_ids_all.shape[0] >= self.num_envs:
                body_ids = body_ids_all[env_ids_t]
                body_state = self._pack_body_state_batch(body_q, body_qd, body_ids)

        if body_q is not None and robot_name in self._obj_root_body_ids:
            root_ids_all = self._obj_root_body_ids[robot_name]
            if root_ids_all.numel() >= self.num_envs:
                root_ids = root_ids_all[env_ids_t]
                valid = root_ids >= 0
                if valid.any():
                    root_ids_safe = root_ids.clone()
                    root_ids_safe[~valid] = 0
                    root_state_vals = self._pack_body_state_batch(body_q, body_qd, root_ids_safe)
                    root_state[valid] = root_state_vals[valid]

        def _gather_dof(src: torch.Tensor, idx: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            idx_safe = idx.clone()
            if valid_mask.dim() == 1:
                if not valid_mask.all():
                    idx_safe[:, ~valid_mask] = 0
                out = src[idx_safe]
                if not valid_mask.all():
                    out[:, ~valid_mask] = 0
            else:
                valid_sel = valid_mask[env_ids_t]
                idx_safe[~valid_sel] = 0
                out = src[idx_safe]
                out[~valid_sel] = 0
            return out

        if num_joints and robot_name in self._obj_joint_q_idx:
            q_idx_all = self._obj_joint_q_idx[robot_name]
            qd_idx_all = self._obj_joint_qd_idx[robot_name]
            valid_mask = self._obj_joint_valid[robot_name]
            q_idx = q_idx_all[env_ids_t]
            qd_idx = qd_idx_all[env_ids_t]

            if joint_q is not None:
                joint_pos = _gather_dof(joint_q, q_idx, valid_mask)
            if joint_qd is not None:
                joint_vel = _gather_dof(joint_qd, qd_idx, valid_mask)
            if joint_target_pos is not None:
                joint_pos_target = _gather_dof(joint_target_pos, qd_idx, valid_mask)
            if joint_target_vel is not None:
                joint_vel_target = _gather_dof(joint_target_vel, qd_idx, valid_mask)
            if joint_f is not None:
                joint_effort_target = _gather_dof(joint_f, qd_idx, valid_mask)
        else:
            for row, env_id in enumerate(env_ids):
                for col, joint_name in enumerate(joint_names):
                    joint_idx = self._get_joint_index(env_id, robot_name, joint_name)
                    if joint_idx is None:
                        continue
                    q_start = self._joint_q_starts[joint_idx]
                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end <= qd_start:
                        continue
                    if qd_end - qd_start != 1:
                        continue
                    if joint_q is not None:
                        joint_pos[row, col] = joint_q[q_start]
                    if joint_qd is not None:
                        joint_vel[row, col] = joint_qd[qd_start]
                    if joint_target_pos is not None:
                        joint_pos_target[row, col] = joint_target_pos[qd_start]
                    if joint_target_vel is not None:
                        joint_vel_target[row, col] = joint_target_vel[qd_start]
                    if joint_f is not None:
                        joint_effort_target[row, col] = joint_f[qd_start]

        if body_state is None and body_q is not None:
            body_states = []
            for row, env_id in enumerate(env_ids):
                root_idx = self._obj_to_root_body[env_id].get(robot_name)
                if root_idx is not None:
                    root_state[row] = self._pack_body_state(body_q, body_qd, [root_idx])[0]

                body_ids = self._get_body_indices(env_id, robot_name)
                if body_ids:
                    body_names = [self._model.body_key[idx] for idx in body_ids]
                    sorted_pairs = sorted(zip(body_names, body_ids), key=lambda pair: pair[0])
                    sorted_body_ids = [idx for _, idx in sorted_pairs]
                else:
                    sorted_body_ids = []
                body_states.append(self._pack_body_state(body_q, body_qd, sorted_body_ids))
            if body_states and body_states[0] is not None:
                body_state = torch.stack(body_states, dim=0)

        return RobotState(
            root_state=root_state,
            body_names=self._obj_sorted_body_names.get(robot_name, self._get_body_names(robot_name)),
            body_state=body_state,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_pos_target=joint_pos_target,
            joint_vel_target=joint_vel_target,
            joint_effort_target=joint_effort_target,
        )

    def _extract_object_state(self, obj_name: str, env_ids: list[int]) -> ObjectState:
        """Extract state for an object across specified environments."""
        num_envs = len(env_ids)
        root_state = torch.zeros(num_envs, 13, device=self._device)

        state = self._state_0
        body_q = wp2torch(state.body_q) if state is not None else None
        body_qd = wp2torch(state.body_qd) if state is not None else None
        joint_q = wp2torch(state.joint_q) if state is not None else None
        joint_qd = wp2torch(state.joint_qd) if state is not None else None

        obj_cfg = self.object_dict.get(obj_name)
        is_articulation = isinstance(obj_cfg, ArticulationObjCfg)

        body_states = []
        joint_names = self._get_joint_names(obj_name, sort=True) if is_articulation else []
        num_joints = len(joint_names)
        joint_pos = torch.zeros(num_envs, num_joints, device=self._device) if is_articulation else None
        joint_vel = torch.zeros(num_envs, num_joints, device=self._device) if is_articulation else None

        for row, env_id in enumerate(env_ids):
            root_idx = self._obj_to_root_body[env_id].get(obj_name)
            if root_idx is not None and body_q is not None:
                root_state[row] = self._pack_body_state(body_q, body_qd, [root_idx])[0]

            if is_articulation:
                body_ids = self._get_body_indices(env_id, obj_name)
                if body_q is not None:
                    if body_ids:
                        body_names = [self._model.body_key[idx] for idx in body_ids]
                        sorted_pairs = sorted(zip(body_names, body_ids), key=lambda pair: pair[0])
                        sorted_body_ids = [idx for _, idx in sorted_pairs]
                    else:
                        sorted_body_ids = []
                    body_states.append(self._pack_body_state(body_q, body_qd, sorted_body_ids))
                else:
                    body_states.append(None)

                for col, joint_name in enumerate(joint_names):
                    joint_idx = self._get_joint_index(env_id, obj_name, joint_name)
                    if joint_idx is None:
                        continue
                    q_start = self._joint_q_starts[joint_idx]
                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end <= qd_start:
                        continue
                    if qd_end - qd_start != 1:
                        continue
                    if joint_q is not None:
                        joint_pos[row, col] = joint_q[q_start]
                    if joint_qd is not None:
                        joint_vel[row, col] = joint_qd[qd_start]

        body_state = None
        if body_states and body_states[0] is not None:
            body_state = torch.stack(body_states, dim=0)

        return ObjectState(
            root_state=root_state,
            body_names=self._get_body_names(obj_name) if is_articulation else None,
            body_state=body_state,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def _set_states(self, states: TensorState | list, env_ids: list[int] | None = None) -> None:
        """Set the physics state from a TensorState or list."""
        if isinstance(states, TensorState):
            self._set_states_tensor(states, env_ids)
            return

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, np.ndarray):
            env_ids = env_ids.tolist()

        env_ids = [int(e) for e in env_ids]

        # Convert to nested dict if needed
        if isinstance(states, TensorState):
            states_nested = state_tensor_to_nested(self, states)

            def state_lookup(idx: int, env_id: int):
                return states_nested[env_id]

        else:
            states_nested = states
            if len(states_nested) == self.num_envs:

                def state_lookup(idx: int, env_id: int):
                    return states_nested[env_id]

            else:

                def state_lookup(idx: int, env_id: int):
                    return states_nested[idx]

        # Prepare arrays for update (clone to host/numpy)
        body_q = self._model.body_q.numpy()
        body_qd = self._model.body_qd.numpy()
        joint_q = self._model.joint_q.numpy() if self._model.joint_q is not None else None
        joint_qd = self._model.joint_qd.numpy() if self._model.joint_qd is not None else None
        joint_X_p = self._model.joint_X_p.numpy() if self._model.joint_X_p is not None else None

        dirty_joints = False
        dirty_bodies = False
        dirty_body_vels = False
        dirty_joint_vels = False

        control_joint_target_pos = (
            wp2torch(self._control.joint_target_pos) if self._control and self._control.joint_target_pos else None
        )
        control_joint_target_vel = (
            wp2torch(self._control.joint_target_vel) if self._control and self._control.joint_target_vel else None
        )

        for i, env_id in enumerate(env_ids):
            state_dict = state_lookup(i, env_id)

            # Combine robots and objects
            all_state = {**state_dict.get("objects", {}), **state_dict.get("robots", {})}

            for name, obj_state in all_state.items():
                # 1. Update Root (Position/Rotation)
                # Try object-specific root map first
                body_idx = self._obj_to_root_body[env_id].get(name)
                if body_idx is None:
                    body_idx = self._body_name_cache[env_id].get(name)

                if body_idx is not None:
                    # body_idx = self._body_name_cache[env_id][name]

                    pos = obj_state.get("pos")
                    quat = obj_state.get("rot")  # wxyz
                    vel = obj_state.get("vel")
                    ang_vel = obj_state.get("ang_vel")

                    # Convert data to numpy
                    def to_np(x):
                        if hasattr(x, "cpu"):
                            return x.cpu().numpy()
                        return np.array(x)

                    pos_np = to_np(pos) if pos is not None else None
                    quat_np = to_np(quat) if quat is not None else None
                    vel_np = to_np(vel) if vel is not None else None
                    ang_vel_np = to_np(ang_vel) if ang_vel is not None else None

                    if pos is not None or quat is not None:
                        # Handle implicit parent joint
                        if body_idx in self._body_parent_joint:
                            joint_idx = self._body_parent_joint[body_idx]
                            j_type = self._joint_types[joint_idx]

                            if j_type == JointType.FREE:
                                # Update joint_q (7 dims: 3 pos, 4 quat)
                                q_start = self._joint_q_starts[joint_idx]
                                if pos_np is not None:
                                    joint_q[q_start : q_start + 3] = pos_np
                                if quat_np is not None:
                                    xyzw = self._wxyz_to_xyzw(quat_np)
                                    joint_q[q_start + 3 : q_start + 7] = xyzw
                                dirty_joints = True

                            elif j_type == JointType.FIXED:
                                # Update joint_X_p (Transform in parent frame)
                                if pos_np is not None:
                                    joint_X_p[joint_idx][:3] = pos_np
                                if quat_np is not None:
                                    xyzw = self._wxyz_to_xyzw(quat_np)
                                    joint_X_p[joint_idx][3:] = xyzw
                                dirty_joints = True
                        else:
                            # Primitive without joint -> update body_q directly
                            if pos_np is not None:
                                body_q[body_idx][:3] = pos_np
                            if quat_np is not None:
                                xyzw = self._wxyz_to_xyzw(quat_np)
                                body_q[body_idx][3:] = xyzw
                            dirty_bodies = True

                    if vel is not None or ang_vel is not None:
                        if vel_np is not None:
                            body_qd[body_idx][:3] = vel_np
                        if ang_vel_np is not None:
                            body_qd[body_idx][3:6] = ang_vel_np
                        dirty_body_vels = True
                    elif pos is not None or quat is not None:
                        body_qd[body_idx][:6] = 0.0
                        dirty_body_vels = True

                # 2. Update Joint DOFs (for robots/articulations)
                dof_pos = obj_state.get("dof_pos") or {}
                dof_vel = obj_state.get("dof_vel") or {}

                for j_name, j_pos in dof_pos.items():
                    if joint_q is None or joint_qd is None:
                        continue
                    j_idx = self._get_joint_index(env_id, name, j_name)
                    if j_idx is None:
                        continue
                    q_start = self._joint_q_starts[j_idx]
                    qd_start = self._joint_qd_starts[j_idx]
                    qd_end = self._joint_qd_starts[j_idx + 1]
                    if qd_end <= qd_start:
                        continue
                    if qd_end - qd_start != 1:
                        continue

                    values = self._coerce_dof_values(j_pos, qd_end - qd_start)
                    if values is None:
                        continue

                    joint_q[q_start] = values[0]
                    dirty_joints = True

                    if j_name in dof_vel:
                        vel_values = self._coerce_dof_values(dof_vel[j_name], qd_end - qd_start)
                        if vel_values is not None:
                            joint_qd[qd_start] = vel_values[0]
                            dirty_joint_vels = True
                    else:
                        joint_qd[qd_start] = 0.0
                        dirty_joint_vels = True

                    if control_joint_target_pos is not None:
                        control_joint_target_pos[qd_start] = values[0]
                    if control_joint_target_vel is not None:
                        control_joint_target_vel[qd_start] = 0.0

                for j_name, j_vel in dof_vel.items():
                    if j_name in dof_pos:
                        continue
                    if joint_qd is None:
                        continue
                    j_idx = self._get_joint_index(env_id, name, j_name)
                    if j_idx is None:
                        continue
                    qd_start = self._joint_qd_starts[j_idx]
                    qd_end = self._joint_qd_starts[j_idx + 1]
                    if qd_end <= qd_start:
                        continue
                    if qd_end - qd_start != 1:
                        continue
                    vel_values = self._coerce_dof_values(j_vel, qd_end - qd_start)
                    if vel_values is None:
                        continue
                    joint_qd[qd_start] = vel_values[0]
                    dirty_joint_vels = True

        # Write back to Newton arrays
        if dirty_bodies:
            self._model.body_q.assign(body_q)
            self._state_0.body_q.assign(body_q)

        if dirty_body_vels:
            self._model.body_qd.assign(body_qd)
            self._state_0.body_qd.assign(body_qd)

        if dirty_joint_vels:
            self._model.joint_qd.assign(joint_qd)
            self._state_0.joint_qd.assign(joint_qd)

        if dirty_joints:
            self._model.joint_q.assign(joint_q)
            self._state_0.joint_q.assign(joint_q)
            self._model.joint_X_p.assign(joint_X_p)

            # Run Forward Kinematics to propagate joint changes to bodies
            eval_fk(self._model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)

    def _set_states_tensor(self, states: TensorState, env_ids: list[int] | None) -> None:
        """Fast-path setter for TensorState using cached indices."""
        if self._model is None or self._state_0 is None:
            return

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self._device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        body_q_model = wp2torch(self._model.body_q) if self._model.body_q is not None else None
        body_q_state = wp2torch(self._state_0.body_q) if self._state_0.body_q is not None else None
        body_qd_model = wp2torch(self._model.body_qd) if self._model.body_qd is not None else None
        body_qd_state = wp2torch(self._state_0.body_qd) if self._state_0.body_qd is not None else None
        joint_q_model = wp2torch(self._model.joint_q) if self._model.joint_q is not None else None
        joint_q_state = wp2torch(self._state_0.joint_q) if self._state_0.joint_q is not None else None
        joint_qd_model = wp2torch(self._model.joint_qd) if self._model.joint_qd is not None else None
        joint_qd_state = wp2torch(self._state_0.joint_qd) if self._state_0.joint_qd is not None else None
        joint_X_p = wp2torch(self._model.joint_X_p) if self._model.joint_X_p is not None else None

        body_q_targets = [t for t in (body_q_model, body_q_state) if t is not None]
        body_qd_targets = [t for t in (body_qd_model, body_qd_state) if t is not None]
        joint_q_targets = [t for t in (joint_q_model, joint_q_state) if t is not None]
        joint_qd_targets = [t for t in (joint_qd_model, joint_qd_state) if t is not None]

        control_joint_target_pos = (
            wp2torch(self._control.joint_target_pos) if self._control and self._control.joint_target_pos else None
        )
        control_joint_target_vel = (
            wp2torch(self._control.joint_target_vel) if self._control and self._control.joint_target_vel else None
        )

        dirty_joints = False

        free_type = int(JointType.FREE)
        fixed_type = int(JointType.FIXED)

        def _assign(targets: list[torch.Tensor], idx: torch.Tensor, values: torch.Tensor) -> None:
            for tgt in targets:
                tgt[idx] = values

        def _scatter_values(
            targets: list[torch.Tensor], idx: torch.Tensor, values: torch.Tensor, valid_mask: torch.Tensor
        ) -> None:
            if not targets:
                return
            if valid_mask.dim() == 1:
                if valid_mask.all():
                    for tgt in targets:
                        tgt[idx] = values
                else:
                    idx_sel = idx[:, valid_mask]
                    vals_sel = values[:, valid_mask]
                    for tgt in targets:
                        tgt[idx_sel] = vals_sel
            else:
                valid_sel = valid_mask[env_ids_t]
                flat_idx = idx[valid_sel]
                flat_vals = values[valid_sel]
                for tgt in targets:
                    tgt[flat_idx] = flat_vals

        def _apply_root_state(name: str, root_state: torch.Tensor) -> None:
            nonlocal dirty_joints
            root_ids_all = self._obj_root_body_ids.get(name)
            if root_ids_all is None:
                return
            root_ids = root_ids_all[env_ids_t]
            valid_root = root_ids >= 0
            if not valid_root.any():
                return
            root_state = root_state[valid_root]
            root_ids = root_ids[valid_root]

            pos = root_state[:, 0:3]
            quat_xyzw = self._reorder_quat_wxyz_to_xyzw(root_state[:, 3:7])
            lin_vel = root_state[:, 7:10]
            ang_vel = root_state[:, 10:13]

            joint_idx_all = self._obj_root_joint_idx.get(name)
            joint_type_all = self._obj_root_joint_type.get(name)
            if joint_idx_all is None or joint_type_all is None:
                if body_q_targets:
                    _assign(body_q_targets, root_ids, torch.cat([pos, quat_xyzw], dim=-1))
                if body_qd_targets:
                    _assign(body_qd_targets, root_ids, torch.cat([lin_vel, ang_vel], dim=-1))
                return

            joint_idx = joint_idx_all[env_ids_t][valid_root]
            joint_type = joint_type_all[env_ids_t][valid_root]

            free_mask = joint_type == free_type
            fixed_mask = joint_type == fixed_type
            no_joint_mask = ~(free_mask | fixed_mask)

            if free_mask.any() and joint_q_targets and self._joint_q_starts_t is not None:
                j_idx = joint_idx[free_mask]
                q_start = self._joint_q_starts_t[j_idx]
                q_idx = q_start[:, None] + (
                    self._root_q_offsets if self._root_q_offsets is not None else torch.arange(7, device=self._device)
                )
                values = torch.cat([pos[free_mask], quat_xyzw[free_mask]], dim=-1)
                _assign(joint_q_targets, q_idx, values)
                dirty_joints = True

            if free_mask.any() and joint_qd_targets and self._joint_qd_starts_t is not None:
                j_idx = joint_idx[free_mask]
                qd_start = self._joint_qd_starts_t[j_idx]
                qd_idx = qd_start[:, None] + (
                    self._root_qd_offsets if self._root_qd_offsets is not None else torch.arange(6, device=self._device)
                )
                values = torch.cat([lin_vel[free_mask], ang_vel[free_mask]], dim=-1)
                _assign(joint_qd_targets, qd_idx, values)
                dirty_joints = True

            if fixed_mask.any() and joint_X_p is not None:
                j_idx = joint_idx[fixed_mask]
                joint_X_p[j_idx, 0:3] = pos[fixed_mask]
                joint_X_p[j_idx, 3:7] = quat_xyzw[fixed_mask]
                dirty_joints = True

            if no_joint_mask.any():
                b_ids = root_ids[no_joint_mask]
                if body_q_targets:
                    _assign(body_q_targets, b_ids, torch.cat([pos[no_joint_mask], quat_xyzw[no_joint_mask]], dim=-1))
                if body_qd_targets:
                    _assign(body_qd_targets, b_ids, torch.cat([lin_vel[no_joint_mask], ang_vel[no_joint_mask]], dim=-1))

        # Apply object states
        for obj in self.objects:
            if obj.name not in states.objects:
                continue
            obj_state = states.objects[obj.name]
            root_state = obj_state.root_state
            if root_state.device != self._device:
                root_state = root_state.to(self._device)
            _apply_root_state(obj.name, root_state[env_ids_t])

            if isinstance(obj, ArticulationObjCfg) and obj_state.joint_pos is not None:
                q_idx_all = self._obj_joint_q_idx.get(obj.name)
                qd_idx_all = self._obj_joint_qd_idx.get(obj.name)
                valid_mask = self._obj_joint_valid.get(obj.name)
                if q_idx_all is not None and qd_idx_all is not None and valid_mask is not None:
                    q_idx = q_idx_all[env_ids_t]
                    qd_idx = qd_idx_all[env_ids_t]
                    joint_pos = obj_state.joint_pos
                    if joint_pos.device != self._device:
                        joint_pos = joint_pos.to(self._device)
                    joint_pos = joint_pos[env_ids_t]
                    joint_vel = obj_state.joint_vel
                    if joint_vel is not None:
                        if joint_vel.device != self._device:
                            joint_vel = joint_vel.to(self._device)
                        joint_vel = joint_vel[env_ids_t]
                    _scatter_values(joint_q_targets, q_idx, joint_pos, valid_mask)
                    if joint_vel is not None:
                        _scatter_values(joint_qd_targets, qd_idx, joint_vel, valid_mask)
                    else:
                        _scatter_values(joint_qd_targets, qd_idx, torch.zeros_like(joint_pos), valid_mask)
                    dirty_joints = True

        # Apply robot states
        for robot in self.robots:
            if robot.name not in states.robots:
                continue
            robot_state = states.robots[robot.name]
            root_state = robot_state.root_state
            if root_state.device != self._device:
                root_state = root_state.to(self._device)
            _apply_root_state(robot.name, root_state[env_ids_t])

            if robot_state.joint_pos is not None:
                q_idx_all = self._obj_joint_q_idx.get(robot.name)
                qd_idx_all = self._obj_joint_qd_idx.get(robot.name)
                valid_mask = self._obj_joint_valid.get(robot.name)
                if q_idx_all is not None and qd_idx_all is not None and valid_mask is not None:
                    q_idx = q_idx_all[env_ids_t]
                    qd_idx = qd_idx_all[env_ids_t]
                    joint_pos = robot_state.joint_pos
                    if joint_pos.device != self._device:
                        joint_pos = joint_pos.to(self._device)
                    joint_pos = joint_pos[env_ids_t]
                    joint_vel = robot_state.joint_vel
                    if joint_vel is not None:
                        if joint_vel.device != self._device:
                            joint_vel = joint_vel.to(self._device)
                        joint_vel = joint_vel[env_ids_t]
                    _scatter_values(joint_q_targets, q_idx, joint_pos, valid_mask)
                    if joint_vel is not None:
                        _scatter_values(joint_qd_targets, qd_idx, joint_vel, valid_mask)
                    else:
                        _scatter_values(joint_qd_targets, qd_idx, torch.zeros_like(joint_pos), valid_mask)

                    if control_joint_target_pos is not None:
                        _scatter_values([control_joint_target_pos], qd_idx, joint_pos, valid_mask)
                    if control_joint_target_vel is not None:
                        if joint_vel is None:
                            joint_vel = torch.zeros_like(joint_pos)
                        _scatter_values([control_joint_target_vel], qd_idx, joint_vel, valid_mask)
                    dirty_joints = True

        if dirty_joints:
            eval_fk(self._model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)

    def _set_dof_targets(self, actions: list[Action] | torch.Tensor | np.ndarray) -> None:
        """Set DOF position/velocity targets for robot joints."""
        self._actions_cache = actions

        if self._control is None:
            return

        joint_target_pos = (
            wp2torch(self._control.joint_target_pos) if self._control.joint_target_pos is not None else None
        )
        joint_target_vel = (
            wp2torch(self._control.joint_target_vel) if self._control.joint_target_vel is not None else None
        )
        joint_f = wp2torch(self._control.joint_f) if self._control.joint_f is not None else None

        # Fast path: tensor/ndarray actions (vectorized envs).
        if isinstance(actions, (torch.Tensor, np.ndarray)):
            if not self.robots:
                return

            # Prefer effort control if any joint is configured as effort (matches legged tasks)
            robot = self.robots[0]
            use_effort = False
            if isinstance(robot, RobotCfg) and robot.control_type:
                use_effort = any(mode == "effort" for mode in robot.control_type.values())

            target_device = None
            if use_effort and joint_f is not None:
                target_device = joint_f.device
            elif joint_target_pos is not None:
                target_device = joint_target_pos.device
            elif joint_f is not None:
                target_device = joint_f.device

            if target_device is None:
                return

            action_tensor = torch.as_tensor(actions, dtype=torch.float32)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if action_tensor.device != target_device:
                action_tensor = action_tensor.to(target_device)

            # Broadcast single action across all envs if needed
            if action_tensor.shape[0] == 1 and self.num_envs > 1:
                action_tensor = action_tensor.repeat(self.num_envs, 1)

            robot_name = robot.name
            qd_idx_all = self._obj_joint_qd_idx.get(robot_name)
            valid_mask = self._obj_joint_valid.get(robot_name)

            if qd_idx_all is not None and valid_mask is not None:
                env_count = min(self.num_envs, action_tensor.shape[0], qd_idx_all.shape[0])
                if env_count <= 0:
                    return
                max_joints = min(action_tensor.shape[1], qd_idx_all.shape[1])
                if max_joints <= 0:
                    return

                qd_idx = qd_idx_all[:env_count, :max_joints]
                if qd_idx.device != target_device:
                    qd_idx = qd_idx.to(target_device)
                action_slice = action_tensor[:env_count, :max_joints]

                if isinstance(valid_mask, torch.Tensor):
                    if valid_mask.ndim == 1:
                        mask = valid_mask[:max_joints]
                        if mask.device != target_device:
                            mask = mask.to(target_device)
                        qd_idx_sel = qd_idx[:, mask]
                        action_sel = action_slice[:, mask]
                        idx_flat = qd_idx_sel.reshape(-1)
                        val_flat = action_sel.reshape(-1)
                    else:
                        mask = valid_mask[:env_count, :max_joints]
                        if mask.device != target_device:
                            mask = mask.to(target_device)
                        idx_flat = qd_idx[mask]
                        val_flat = action_slice[mask]
                else:
                    idx_flat = qd_idx.reshape(-1)
                    val_flat = action_slice.reshape(-1)

                if idx_flat.numel() == 0:
                    return

                if use_effort and joint_f is not None:
                    joint_f[idx_flat] = val_flat
                else:
                    if joint_target_pos is not None:
                        joint_target_pos[idx_flat] = val_flat
                    if joint_target_vel is not None:
                        joint_target_vel[idx_flat] = 0.0
                return

            joint_names = self._get_joint_names(robot.name, sort=True)
            max_joints = min(action_tensor.shape[1], len(joint_names))

            for env_id in range(min(self.num_envs, action_tensor.shape[0])):
                for col in range(max_joints):
                    joint_name = joint_names[col]
                    joint_idx = self._get_joint_index(env_id, robot.name, joint_name)
                    if joint_idx is None:
                        continue
                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end - qd_start != 1:
                        continue

                    value = action_tensor[env_id, col]
                    if use_effort and joint_f is not None:
                        joint_f[qd_start] = value
                    else:
                        if joint_target_pos is not None:
                            joint_target_pos[qd_start] = value
                        if joint_target_vel is not None:
                            joint_target_vel[qd_start] = 0.0
            return

        for env_id, action in enumerate(actions):
            for robot in self.robots:
                if robot.name not in action:
                    continue
                robot_action = action[robot.name]
                dof_pos = robot_action.get("dof_pos_target") or {}
                dof_effort = robot_action.get("dof_effort_target") or {}
                dof_vel_target = robot_action.get("dof_vel_target") or {}

                if joint_target_pos is not None:
                    for joint_name, target in dof_pos.items():
                        joint_idx = self._get_joint_index(env_id, robot.name, joint_name)
                        if joint_idx is None:
                            continue
                        qd_start = self._joint_qd_starts[joint_idx]
                        qd_end = self._joint_qd_starts[joint_idx + 1]
                        if qd_end - qd_start != 1:
                            continue
                        values = self._coerce_dof_values(target, qd_end - qd_start)
                        if values is None:
                            continue
                        joint_target_pos[qd_start] = values[0]
                        if joint_target_vel is not None and joint_name not in dof_vel_target:
                            joint_target_vel[qd_start] = 0.0

                if joint_target_vel is not None:
                    for joint_name, target in dof_vel_target.items():
                        joint_idx = self._get_joint_index(env_id, robot.name, joint_name)
                        if joint_idx is None:
                            continue
                        qd_start = self._joint_qd_starts[joint_idx]
                        qd_end = self._joint_qd_starts[joint_idx + 1]
                        if qd_end - qd_start != 1:
                            continue
                        values = self._coerce_dof_values(target, qd_end - qd_start)
                        if values is None:
                            continue
                        joint_target_vel[qd_start] = values[0]

                if joint_f is not None:
                    for joint_name, effort in dof_effort.items():
                        joint_idx = self._get_joint_index(env_id, robot.name, joint_name)
                        if joint_idx is None:
                            continue
                        qd_start = self._joint_qd_starts[joint_idx]
                        qd_end = self._joint_qd_starts[joint_idx + 1]
                        if qd_end - qd_start != 1:
                            continue
                        values = self._coerce_dof_values(effort, qd_end - qd_start)
                        if values is None:
                            continue
                        joint_f[qd_start] = values[0]

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get joint names for an articulated object."""
        obj_cfg = self.object_dict.get(obj_name)
        env_id = 0

        if isinstance(obj_cfg, RobotCfg):
            if obj_cfg.joint_limits:
                names = list(obj_cfg.joint_limits.keys())
            elif obj_cfg.actuators:
                names = list(obj_cfg.actuators.keys())
            else:
                names = []
            obj_joint_map = self._get_obj_joint_name_map(env_id, obj_name)
            if obj_joint_map:
                names = [name for name in names if name in obj_joint_map]
            if not names:
                names = self._collect_joint_names(env_id, obj_name)
        else:
            names = self._collect_joint_names(env_id, obj_name)

        if sort:
            names.sort()
        return names

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get body/link names for an articulated object."""
        obj_cfg = self.object_dict.get(obj_name)
        if not isinstance(obj_cfg, ArticulationObjCfg):
            return []

        env_id = 0
        body_ids = self._get_body_indices(env_id, obj_name)
        if not body_ids:
            return []

        # Include the root link to align with other simulators (IsaacGym/MuJoCo).
        names = [self._model.body_key[idx] for idx in body_ids]
        if sort:
            names.sort()
        return names

    def _get_body_ids_reindex(self, obj_name: str) -> list[int]:
        """Get body indices for reindexing contact forces.

        Returns a list of body indices in sorted body name order, which can be used
        to reorder contact forces to match the sorted body order expected by ContactForces query.

        Args:
            obj_name: Name of the object (robot or articulated object)

        Returns:
            List of body indices in sorted name order
        """
        env_id = 0
        body_ids = self._get_body_indices(env_id, obj_name)
        if not body_ids:
            return []

        # Get body names and their indices
        body_names_with_indices = [(self._model.body_key[idx], idx) for idx in body_ids]

        # Sort by body name and return the indices
        sorted_pairs = sorted(body_names_with_indices, key=lambda x: x[0])
        return [idx for _, idx in sorted_pairs]

    def init_contact_sensor(self, robot_name: str) -> None:
        """Initialize contact sensor for the given robot.

        This is called by ContactForces.bind_handler() when contact force query is registered.

        Args:
            robot_name: Name of the robot to create contact sensor for
        """
        if self._contact_sensor is not None:
            return  # Already initialized

        if self._model is None:
            log.warning("Cannot initialize contact sensor: model not yet built")
            return

        # Create sensor that senses contact forces on all bodies of the robot
        # We use body pattern matching to select all bodies belonging to this robot
        body_ids = self._get_body_indices(0, robot_name)
        if not body_ids:
            log.warning(f"No bodies found for robot {robot_name}")
            return

        # Get body names for pattern matching
        body_names = [self._model.body_key[idx] for idx in body_ids]

        # Create contact sensor for these bodies
        self._contact_sensor = SensorContact(
            self._model,
            sensing_obj_bodies=body_names,
            include_total=True,
        )
        log.info(f"Initialized Newton contact sensor for {robot_name} with {len(body_names)} bodies")

    def _resolve_contact_capacity(self, fallback: int | None) -> int:
        if self._solver is None:
            return fallback if fallback is not None else 2048
        mjw_data = getattr(self._solver, "mjw_data", None)
        if mjw_data is not None:
            return int(mjw_data.naconmax)
        mj_model = getattr(self._solver, "mj_model", None)
        if mj_model is not None:
            return int(mj_model.nconmax)
        return fallback if fallback is not None else 2048

    def _ensure_contact_buffers(self) -> None:
        if self._contacts is None:
            return
        if not hasattr(self._contacts, "pair"):
            self._contacts.pair = wp.zeros(
                self._contacts.rigid_contact_max,
                dtype=wp.vec2i,
                device=self._contacts.device,
            )
        if not hasattr(self._contacts, "normal"):
            self._contacts.normal = wp.zeros(
                self._contacts.rigid_contact_max,
                dtype=wp.vec3,
                device=self._contacts.device,
            )
        if not hasattr(self._contacts, "force"):
            self._contacts.force = wp.zeros(
                self._contacts.rigid_contact_max,
                dtype=wp.float32,
                device=self._contacts.device,
            )

    def get_contact_forces(self) -> torch.Tensor:
        """Get the current contact forces from the contact sensor.

        Returns:
            Tensor of shape (num_bodies, 3) containing contact forces for each body
        """
        if self._contact_sensor is None or self._contacts is None:
            return torch.zeros((0, 3), device=self.device)

        self._ensure_contact_buffers()

        # Evaluate contact sensor with current contacts
        self._contact_sensor.eval(self._contacts)

        # Get net force from sensor
        net_force = self._contact_sensor.get_total_force()

        # Convert warp array to torch tensor
        # net_force shape is (num_sensing_objs, num_counterparts) with each element being vec3
        # Since we used include_total=True, the first column is the total force
        net_force_np = net_force.numpy()

        # Reshape to (num_bodies, 3) - take the total force (first column)
        if net_force_np.shape[1] > 0:
            forces = torch.tensor(net_force_np[:, 0, :], dtype=torch.float32, device=self.device)
        else:
            forces = torch.zeros((net_force_np.shape[0], 3), device=self.device)

        return forces

    def close(self) -> None:
        """Clean up Newton resources."""
        log.info("Closing Newton handler")
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._camera_groups = []
        self._use_tiled_camera = False
        self._model = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._solver = None

    @property
    def device(self) -> torch.device:
        """Return the device used for tensors."""
        return torch.device(self._device)

    # Quaternion conversion utilities
    @staticmethod
    def _wxyz_to_xyzw(quat) -> tuple:
        """Convert quaternion from wxyz (MetaSim) to xyzw (Newton) format."""
        if quat is None:
            return (0.0, 0.0, 0.0, 1.0)
        return (quat[1], quat[2], quat[3], quat[0])

    @staticmethod
    def _xyzw_to_wxyz(quat) -> tuple:
        """Convert quaternion from xyzw (Newton) to wxyz (MetaSim) format."""
        if quat is None:
            return (1.0, 0.0, 0.0, 0.0)
        return (quat[3], quat[0], quat[1], quat[2])
