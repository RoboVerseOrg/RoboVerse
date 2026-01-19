from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger as log

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

from collections import defaultdict

import newton
import numpy as np
import scipy.spatial.transform as tr
import warp as wp
from newton import Contacts
from newton._src.sim.articulation import eval_fk
from newton._src.sim.joints import JointType
from newton.sensors import SensorContact, populate_contacts
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

        self._scenario = scenario
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Newton model and state
        self._model: newton.Model | None = None
        self._state_0: newton.State | None = None
        self._state_1: newton.State | None = None
        self._control: newton.Control | None = None
        self._solver: SolverMuJoCo | None = None
        self._viewer = None
        self._newton_camera = None
        self._sim_time = 0.0

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
        self._solver = SolverMuJoCo(
            self._model,
            njmax=sim_params.njmax,
            nconmax=sim_params.nconmax,
        )

        # Initialize Contacts object for contact force queries
        nconmax = self._resolve_contact_capacity(sim_params.nconmax)
        self._contacts = Contacts(
            rigid_contact_max=nconmax,
            soft_contact_max=0,
            device=self._device,
        )
        self._ensure_contact_buffers()

        # Initialize Viewer for Rendering (GUI is controlled by headless only)
        self._viewer = None
        self._newton_camera = None  # Newton's native Camera for ViewerGL rendering
        headless = getattr(self.scenario, "headless", True)
        if (not headless) or self.scenario.cameras:
            if self.scenario.cameras:
                max_w = max(c.width for c in self.scenario.cameras)
                max_h = max(c.height for c in self.scenario.cameras)
            else:
                # Default window size when no cameras are defined
                max_w, max_h = 1280, 720

            self._viewer = ViewerGL(width=max_w, height=max_h, headless=headless)
            self._viewer.set_model(self._model)
            # Set world offsets for visual separation of parallel environments
            spacing = self.scenario.env_spacing
            self._viewer.set_world_offsets((spacing, spacing, 0.0))
            if not headless:
                if self._viewer.ui is None or not self._viewer.ui.is_available:
                    log.warning("Newton Viewer UI is unavailable. Install `imgui-bundle` to enable the left panel.")

            if self.scenario.cameras:
                # Create Newton Camera from first camera config
                # We'll update it per-camera during _get_states
                from newton._src.viewer.camera import Camera as NewtonCamera

                first_cam = self.scenario.cameras[0]
                pitch, yaw = self._look_at_to_pitch_yaw(first_cam.pos, first_cam.look_at)
                self._newton_camera = NewtonCamera(
                    fov=first_cam.vertical_fov,
                    width=first_cam.width,
                    height=first_cam.height,
                    pos=first_cam.pos,
                    up_axis="Z",
                )
                self._newton_camera.pitch = pitch
                self._newton_camera.yaw = yaw

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

    def _build_model(self) -> None:
        """Build Newton model from scenario configuration."""
        builder = newton.ModelBuilder()

        # Set timestep
        if self.scenario.sim_params.dt is not None:
            builder.default_shape_thickness = 0.001
            # dt is set during solver.step()

        # Add ground plane (global, shared across all worlds)
        builder.current_world = -1
        builder.add_ground_plane()

        # Add robots and objects for each environment
        # Add robots and objects for each environment
        self._obj_to_root_body = defaultdict(dict)

        for env_id in range(self.num_envs):
            # Create sub-builder for this environment
            env_builder = newton.ModelBuilder()

            # Track local indices for this env
            env_map = {}

            # Add robot(s)
            for robot in self.robots:
                # Check current body count
                start_count = len(env_builder.body_mass)
                self._add_robot_to_builder(env_builder, robot)
                end_count = len(env_builder.body_mass)
                if end_count > start_count:
                    # Assume first added body is root
                    env_map[robot.name] = start_count

            # Add objects
            for obj in self.objects:
                start_count = len(env_builder.body_mass)
                self._add_object_to_builder(env_builder, obj)
                end_count = len(env_builder.body_mass)
                if end_count > start_count:
                    env_map[obj.name] = start_count

            # Add this environment as a separate world
            # Calculate global offset
            global_offset = len(builder.body_mass)
            builder.add_world(env_builder)

            # Store global indices
            for name, local_idx in env_map.items():
                self._obj_to_root_body[env_id][name] = global_offset + local_idx

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
        )

    def _add_object_to_builder(self, builder: newton.ModelBuilder, obj) -> None:
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
            builder.add_shape_box(
                body=body,
                hx=obj.half_size[0],
                hy=obj.half_size[1],
                hz=obj.half_size[2],
            )
        elif isinstance(obj, PrimitiveSphereCfg):
            # Add sphere shape
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*obj.default_position),
                    wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                ),
                key=obj.name,
            )
            builder.add_shape_sphere(body=body, radius=obj.radius)
        elif isinstance(obj, PrimitiveCylinderCfg):
            # Add capsule shape (Newton uses capsules, approximate cylinder)
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*obj.default_position),
                    wp.quat(*self._wxyz_to_xyzw(obj.default_orientation)),
                ),
                key=obj.name,
            )
            builder.add_shape_capsule(body=body, radius=obj.radius, half_height=obj.height / 2)
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
                )

    def _build_name_caches(self) -> None:
        """Build caches for looking up body and joint indices by name."""
        self._body_name_cache = defaultdict(dict)
        self._joint_name_cache = defaultdict(dict)
        self._body_children = defaultdict(lambda: defaultdict(list))
        self._body_child_to_joint = defaultdict(dict)
        self._obj_body_indices = defaultdict(dict)
        self._obj_joint_indices = defaultdict(dict)

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
                    joint_idx = self._joint_name_cache[env_id].get(joint_name)
                    if joint_idx is None:
                        continue

                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end <= qd_start:
                        continue

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

    @staticmethod
    def _reorder_quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Reorder quaternion from xyzw to wxyz for torch tensors."""
        return torch.stack(
            [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
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

    def _look_at_to_pitch_yaw(self, pos, look_at):
        """Convert camera position and look_at target to pitch and yaw angles.

        Newton's Camera uses pitch (elevation) and yaw (azimuth) angles.
        For Z-up: yaw rotates around Z, pitch is elevation from XY plane.
        """
        pos = np.array(pos, dtype=np.float32)
        target = np.array(look_at, dtype=np.float32)

        # Direction from camera to target
        direction = target - pos
        direction /= np.linalg.norm(direction) + 1e-6

        # For Z-up coordinate system:
        # yaw = angle in XY plane from +X axis (counterclockwise)
        # pitch = elevation angle from XY plane

        # Compute yaw (azimuth)
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))

        # Compute pitch (elevation)
        horizontal_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch = np.degrees(np.arctan2(direction[2], horizontal_dist))

        return pitch, yaw

    def _look_at_to_quat(self, pos: tuple[float, float, float], look_at: tuple[float, float, float]) -> str:
        """Compute quaternion (w x y z) for camera looking at target from pos."""
        pos = np.array(pos)
        target = np.array(look_at)

        # Camera forward (-Z) points to target
        forward = target - pos
        forward /= np.linalg.norm(forward) + 1e-6

        # World Up
        world_up = np.array([0.0, 0.0, 1.0])

        # Right (+X)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Degenerate case (looking straight up/down), assume X is right
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= right_norm

        # Up (+Y)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-6

        # Rotation Matrix: [right, up, -forward]
        # Columns are the axes of the camera frame in world coordinates
        R = np.column_stack([right, up, -forward])

        # Convert to quaternion (x, y, z, w)
        quat = tr.Rotation.from_matrix(R).as_quat()

        # Return w x y z
        return f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}"

    def _simulate(self) -> None:
        """Advance simulation by one step (with decimation substeps)."""
        dt = self.scenario.sim_params.dt if self.scenario.sim_params.dt else 0.005

        for _ in range(self.decimation):
            # Apply gravity compensation (per-body) if configured
            self._apply_gravity_compensation()

            # Generate contacts
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

        # Camera rendering using ViewerGL
        if self._viewer and self.scenario.cameras and self._newton_camera:
            from pyglet.math import Vec3 as PyVec3  # Newton Camera uses pyglet Vec3

            self._viewer.log_state(self._state_0)

            # Get world offsets for per-environment rendering
            world_offsets_np = None
            if self._viewer.world_offsets is not None:
                world_offsets_np = self._viewer.world_offsets.numpy()

            for cam_cfg in self.scenario.cameras:
                rgb_list = []
                depth_list = []

                for env_id in env_ids:
                    # Update Newton Camera params
                    self._newton_camera.width = cam_cfg.width
                    self._newton_camera.height = cam_cfg.height
                    self._newton_camera.fov = cam_cfg.vertical_fov

                    # Calculate camera position with world offset for this environment
                    cam_pos = list(cam_cfg.pos)
                    look_at = list(cam_cfg.look_at) if hasattr(cam_cfg, "look_at") and cam_cfg.look_at else None

                    if world_offsets_np is not None and env_id < len(world_offsets_np):
                        offset = world_offsets_np[env_id]
                        cam_pos[0] += offset[0]
                        cam_pos[1] += offset[1]
                        cam_pos[2] += offset[2]
                        if look_at is not None:
                            look_at[0] += offset[0]
                            look_at[1] += offset[1]
                            look_at[2] += offset[2]

                    # Set camera position
                    self._newton_camera.pos = PyVec3(*cam_pos)

                    if look_at is not None:
                        pitch, yaw = self._look_at_to_pitch_yaw(cam_pos, look_at)
                        self._newton_camera.pitch = pitch
                        self._newton_camera.yaw = yaw
                    else:
                        # Default orientation (looking down -Y for Z-up)
                        self._newton_camera.pitch = 0.0
                        self._newton_camera.yaw = -180.0

                    # Render using Newton Camera
                    self._viewer.renderer.render(self._newton_camera, self._viewer.objects, self._viewer.lines)

                    # Get frame - returns warp array (H, W, 3) uint8
                    rgb_wp = self._viewer.get_frame()

                    # Convert to torch tensor
                    rgb_np = rgb_wp.numpy()
                    rgb_tensor = torch.from_numpy(rgb_np.copy()).to(self._device)
                    rgb_list.append(rgb_tensor)

                    # Placeholder depth (ViewerGL doesn't expose depth readback easily)
                    depth_tensor = torch.zeros((cam_cfg.height, cam_cfg.width), device=self._device)
                    depth_list.append(depth_tensor.unsqueeze(-1))  # Shape: (H, W, 1)

                # Stack all environment renders
                rgb_stacked = torch.stack(rgb_list, dim=0)  # (num_envs, H, W, 3)
                depth_stacked = torch.stack(depth_list, dim=0)  # (num_envs, H, W, 1)

                camera_states[cam_cfg.name] = CameraState(
                    rgb=rgb_stacked,
                    depth=depth_stacked,
                    intrinsics=torch.tensor(cam_cfg.intrinsics, device=self.device)
                    .unsqueeze(0)
                    .expand(len(env_ids), -1, -1),
                )

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
        root_state = torch.zeros(num_envs, 13, device=self._device)

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

        body_states = []

        for row, env_id in enumerate(env_ids):
            root_idx = self._obj_to_root_body[env_id].get(robot_name)
            if root_idx is not None and body_q is not None:
                root_state[row] = self._pack_body_state(body_q, body_qd, [root_idx])[0]

            body_ids = self._get_body_indices(env_id, robot_name)
            if body_q is not None:
                body_ids_no_root = body_ids[1:]
                if body_ids_no_root:
                    body_names = [self._model.body_key[idx] for idx in body_ids_no_root]
                    sorted_pairs = sorted(zip(body_names, body_ids_no_root), key=lambda pair: pair[0])
                    sorted_body_ids = [idx for _, idx in sorted_pairs]
                else:
                    sorted_body_ids = []
                body_states.append(self._pack_body_state(body_q, body_qd, sorted_body_ids))
            else:
                body_states.append(None)

            for col, joint_name in enumerate(joint_names):
                joint_idx = self._joint_name_cache[env_id].get(joint_name)
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

        body_state = None
        if body_states and body_states[0] is not None:
            body_state = torch.stack(body_states, dim=0)

        return RobotState(
            root_state=root_state,
            body_names=self._get_body_names(robot_name),
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
                    body_ids_no_root = body_ids[1:]
                    if body_ids_no_root:
                        body_names = [self._model.body_key[idx] for idx in body_ids_no_root]
                        sorted_pairs = sorted(zip(body_names, body_ids_no_root), key=lambda pair: pair[0])
                        sorted_body_ids = [idx for _, idx in sorted_pairs]
                    else:
                        sorted_body_ids = []
                    body_states.append(self._pack_body_state(body_q, body_qd, sorted_body_ids))
                else:
                    body_states.append(None)

                for col, joint_name in enumerate(joint_names):
                    joint_idx = self._joint_name_cache[env_id].get(joint_name)
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
                    if j_name not in self._joint_name_cache[env_id]:
                        continue
                    j_idx = self._joint_name_cache[env_id][j_name]
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
                    if j_name not in self._joint_name_cache[env_id]:
                        continue
                    j_idx = self._joint_name_cache[env_id][j_name]
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

            joint_names = self._get_joint_names(robot.name, sort=True)
            max_joints = min(action_tensor.shape[1], len(joint_names))

            for env_id in range(min(self.num_envs, action_tensor.shape[0])):
                for col in range(max_joints):
                    joint_name = joint_names[col]
                    joint_idx = self._joint_name_cache[env_id].get(joint_name)
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
                        joint_idx = self._joint_name_cache[env_id].get(joint_name)
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
                        joint_idx = self._joint_name_cache[env_id].get(joint_name)
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
                        joint_idx = self._joint_name_cache[env_id].get(joint_name)
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
            if env_id in self._joint_name_cache:
                names = [name for name in names if name in self._joint_name_cache[env_id]]
            if not names:
                joint_ids = self._get_joint_indices(env_id, obj_name)
                for joint_idx in joint_ids:
                    qd_start = self._joint_qd_starts[joint_idx]
                    qd_end = self._joint_qd_starts[joint_idx + 1]
                    if qd_end - qd_start == 1:
                        names.append(self._model.joint_key[joint_idx])
        else:
            joint_ids = self._get_joint_indices(env_id, obj_name)
            names = []
            for joint_idx in joint_ids:
                qd_start = self._joint_qd_starts[joint_idx]
                qd_end = self._joint_qd_starts[joint_idx + 1]
                if qd_end - qd_start == 1:
                    names.append(self._model.joint_key[joint_idx])

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

        names = [self._model.body_key[idx] for idx in body_ids[1:]]
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
