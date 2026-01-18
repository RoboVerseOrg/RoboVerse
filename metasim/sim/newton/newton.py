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
from newton._src.sim.articulation import eval_fk
from newton._src.sim.joints import JointType
from newton.solvers import SolverMuJoCo
from newton.viewer import ViewerGL

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
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

        # Caches for efficient lookups
        self._joint_name_to_id: dict[str, dict[str, int]] = {}
        self._body_name_to_id: dict[str, dict[str, int]] = {}
        self._robot_joint_ids: dict[str, list[int]] = {}
        self._robot_body_ids: dict[str, list[int]] = {}

        # Actions cache
        self._actions_cache: list[Action] = []

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

        # Create states (double-buffering for solver)
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # Create MuJoCo solver
        self._solver = SolverMuJoCo(self._model)

        # Initialize Viewer for Rendering (if needed)
        self._viewer = None
        self._newton_camera = None  # Newton's native Camera for ViewerGL rendering
        if self.scenario.cameras:
            max_w = max([c.width for c in self.scenario.cameras])
            max_h = max([c.height for c in self.scenario.cameras])

            headless = getattr(self.scenario, "headless", True)

            self._viewer = ViewerGL(width=max_w, height=max_h, headless=headless)
            self._viewer.set_model(self._model)
            if not headless:
                if self._viewer.ui is None or not self._viewer.ui.is_available:
                    log.warning("Newton Viewer UI is unavailable. Install `imgui-bundle` to enable the left panel.")

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

        # Build name-to-ID caches
        self._build_name_caches()

        log.info(f"Newton launched with {self.num_envs} worlds, solver={self._solver.__class__.__name__}")

        return super().launch()

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

        for i, key in enumerate(joint_keys):
            world = joint_worlds[i]
            if world != -1:
                self._joint_name_cache[world][key] = i

        # Build body -> parent joint map
        joint_children = self._model.joint_child.numpy()
        for i, child_idx in enumerate(joint_children):
            if child_idx >= 0:
                self._body_parent_joint[child_idx] = i

        # Access start indices for fast lookup
        self._joint_q_starts = self._model.joint_q_start.numpy()
        self._joint_qd_starts = self._model.joint_qd_start.numpy()
        self._joint_types = self._model.joint_type.numpy()

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

            for cam_cfg in self.scenario.cameras:
                # Update Newton Camera params
                self._newton_camera.width = cam_cfg.width
                self._newton_camera.height = cam_cfg.height
                self._newton_camera.fov = cam_cfg.vertical_fov

                # Set camera position and compute pitch/yaw from look_at
                self._newton_camera.pos = PyVec3(*cam_cfg.pos)

                if hasattr(cam_cfg, "look_at") and cam_cfg.look_at is not None:
                    pitch, yaw = self._look_at_to_pitch_yaw(cam_cfg.pos, cam_cfg.look_at)
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

                # Convert to torch tensor (N, H, W, 3) float32 [0, 1]
                rgb_np = rgb_wp.numpy()
                # Return uint8 RGB to match MuJoCo convention (test script expects uint8 for imageio)
                rgb_tensor = torch.from_numpy(rgb_np.copy()).to(self._device)

                # Placeholder depth (ViewerGL doesn't expose depth readback easily)
                depth_tensor = torch.zeros((cam_cfg.height, cam_cfg.width), device=self._device)

                # Add batch dim
                if self.num_envs == 1:
                    rgb_tensor = rgb_tensor.unsqueeze(0)  # (1, H, W, 3)
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
                else:
                    rgb_tensor = rgb_tensor.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(-1).expand(self.num_envs, -1, -1, -1)

                camera_states[cam_cfg.name] = CameraState(
                    rgb=rgb_tensor,
                    depth=depth_tensor,
                    intrinsics=torch.tensor(cam_cfg.intrinsics, device=self.device)
                    .unsqueeze(0)
                    .expand(self.num_envs, -1, -1),
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

        # Get body transforms and velocities from Newton state
        # Newton State has: body_q (transforms), body_qd (velocities), joint_q, joint_qd
        body_q = wp2torch(state.body_q)  # (num_bodies, 7) - pos(3) + quat(4)
        body_qd = wp2torch(state.body_qd)  # (num_bodies, 6) - lin_vel(3) + ang_vel(3)
        joint_q = wp2torch(state.joint_q)  # (num_joints,)
        joint_qd = wp2torch(state.joint_qd)  # (num_joints,)

        # TODO: Map body/joint indices per robot per world
        # For now, return placeholder with correct structure
        num_envs = len(env_ids)

        # Combine into root_state format: (N, 13) = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # Newton quat is xyzw, need to convert to wxyz for MetaSim
        root_state = torch.zeros(num_envs, 13, device=self._device)

        return RobotState(
            root_state=root_state,
            body_names=[],  # TODO: populate from cache
            body_state=None,
            joint_pos=joint_q.unsqueeze(0).expand(num_envs, -1) if joint_q is not None else None,
            joint_vel=joint_qd.unsqueeze(0).expand(num_envs, -1) if joint_qd is not None else None,
            joint_pos_target=None,
            joint_vel_target=None,
            joint_effort_target=None,
        )

    def _extract_object_state(self, obj_name: str, env_ids: list[int]) -> ObjectState:
        """Extract state for an object across specified environments."""
        num_envs = len(env_ids)
        root_state = torch.zeros(num_envs, 13, device=self._device)

        return ObjectState(root_state=root_state)

    def _set_states(self, states: TensorState | list, env_ids: list[int] | None = None) -> None:
        """Set the physics state from a TensorState or list."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Convert to nested dict if needed
        if isinstance(states, TensorState):
            states_nested = state_tensor_to_nested(self, states)
        else:
            states_nested = states

        # Prepare arrays for update (clone to host/numpy)
        body_q = self._model.body_q.numpy()
        # body_qd = self._model.body_qd.numpy()
        joint_q = self._model.joint_q.numpy()
        # joint_qd = self._model.joint_qd.numpy()
        joint_X_p = self._model.joint_X_p.numpy()

        dirty_joints = False
        dirty_bodies = False

        for i, env_id in enumerate(env_ids):
            state_dict = states_nested[i]

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

                    if pos is not None or quat is not None:
                        # Convert data to numpy
                        def to_np(x):
                            if hasattr(x, "cpu"):
                                return x.cpu().numpy()
                            return np.array(x)

                        pos_np = to_np(pos) if pos is not None else None
                        quat_np = to_np(quat) if quat is not None else None

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

                # 2. Update Joint DOFs (for robots)
                if "dof_pos" in obj_state:
                    for j_name, j_pos in obj_state["dof_pos"].items():
                        # Find joint index
                        if j_name in self._joint_name_cache[env_id]:
                            j_idx = self._joint_name_cache[env_id][j_name]
                            q_start = self._joint_q_starts[j_idx]

                            # Update joint position
                            val = j_pos.item() if hasattr(j_pos, "item") else j_pos
                            joint_q[q_start] = val
                            dirty_joints = True

        # Write back to Newton arrays
        if dirty_bodies:
            self._model.body_q.assign(body_q)
            self._state_0.body_q.assign(body_q)

        if dirty_joints:
            self._model.joint_q.assign(joint_q)
            self._state_0.joint_q.assign(joint_q)
            self._model.joint_X_p.assign(joint_X_p)

            # Run Forward Kinematics to propagate joint changes to bodies
            eval_fk(self._model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)
        pass

    def _set_dof_targets(self, actions: list[Action]) -> None:
        """Set DOF position/velocity targets for robot joints."""
        self._actions_cache = actions

        # Write targets to Newton Control object
        # TODO: Map action dict to control array indices
        pass

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get joint names for an articulated object."""
        # TODO: Extract from Newton model
        return []

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get body/link names for an articulated object."""
        # TODO: Extract from Newton model
        return []

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
