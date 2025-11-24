from __future__ import annotations

import torch
import numpy as np
import warnings
from collections import deque

from metasim.sim.base import BaseSimHandler, BaseQueryType
from metasim.utils.math import quat_apply, convert_quat

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass


class ContactForces(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """
    def __init__(self, history_length: int = 3):
        super().__init__()
        self.history_length = history_length
        self._current_contact_force = None
        self._contact_forces_queue = deque(maxlen=history_length)

    def bind_handler(self, handler:BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        if self.simulator in ["isaacgym", "mujoco"]:
            self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
        elif self.simulator == "isaacsim":
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = torch.tensor([self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names], dtype=torch.int, device=self.handler.device)
        else:
            raise NotImplementedError
        self.initialize()
        self.__call__()

    def initialize(self):
        for _ in range(self.history_length):
            if self.simulator == "isaacgym":
                self._current_contact_force = isaacgym.gymtorch.wrap_tensor(self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim))
            elif self.simulator == "isaacsim":
                self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
            elif self.simulator == "mujoco":
                self._current_contact_force = self._get_contact_forces_mujoco()
            else:
                raise NotImplementedError
            self._contact_forces_queue.append(self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :])

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """
        Compute net contact forces on each body.
        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        return contact_forces

    def __call__(self):
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self._current_contact_force = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError
        self._contact_forces_queue.append(self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :])
        return {self.robots[0].name: self}

    @property
    def contact_forces_history(self) -> torch.Tensor:
        return torch.stack(list(self._contact_forces_queue), dim=1) # (num_envs, history_length, num_bodies, 3)

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_queue[-1]

class LidarPointCloud(BaseQueryType):
    """Optional query that produces a LiDAR point cloud using LidarSensor + Warp.

    Notes
    - Supports IsaacGym and MuJoCo via common state interface; raycasting is done against a generated terrain mesh.
    - Robot self-geometry is not included in the mesh to keep this query generic and lightweight.
    - Requires packages: LidarSensor, warp, trimesh. If unavailable, returns None payload when enabled.
    - Quaternions: handler states use (w,x,y,z). LidarSensor expects (x,y,z,w). Conversion is handled internally.
    """

    def __init__(
        self,
        link_name: str = "mid360_link",
        sensor_type: str = "mid360",
        apply_optical_center_offset: bool = True,
        optical_center_offset_z: float = 0.03503,
        enabled: bool = False,
    ):
        super().__init__()
        self.link_name = link_name
        self.sensor_type = sensor_type
        self.apply_optical_center_offset = apply_optical_center_offset
        self.optical_center_offset_z = optical_center_offset_z
        self.enabled = enabled

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.handler = handler
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        self.device = str(handler.device) # warp only accepts str device
        self._init_backend()

    def _init_backend(self):
        if self.simulator in ["isaacgym", "mujoco"]:
            self._init_backend_mujoco_isaacgym()
        elif self.simulator == "isaacsim":
            self._init_backend_isaacsim()

    def _init_backend_mujoco_isaacgym(self):
        import warp as wp
        import trimesh
        from LidarSensor.lidar_sensor import LidarSensor
        from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig

        self.wp = wp
        self.trimesh = trimesh
        self.LidarSensor = LidarSensor
        self.LidarConfig = LidarConfig
        self.wp.init()
        vertices = self.handler._ground_mesh_vertices
        triangles = self.handler._ground_mesh_triangles

        import torch

        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        faces_wp_int32_array = self.wp.from_numpy(triangles.reshape(-1), dtype=self.wp.int32, device=self.device)
        vertex_vec3_array = self.wp.from_torch(vertex_tensor, dtype=self.wp.vec3)
        self.wp_mesh = self.wp.Mesh(points=vertex_vec3_array, indices=faces_wp_int32_array)
        self.mesh_ids = self.wp.array([self.wp_mesh.id], dtype=self.wp.uint64, device=self.device)

        # Prepare sensor config and buffers
        self.sensor_cfg = self.LidarConfig()
        self.sensor_cfg.sensor_type = self.sensor_type

        num_envs = self.num_envs
        num_sensors = int(getattr(self.sensor_cfg, "num_sensors", 1))
        v_lines = int(getattr(self.sensor_cfg, "vertical_line_num", 128))
        h_lines = int(getattr(self.sensor_cfg, "horizontal_line_num", 512))

        self.lidar_tensor = torch.zeros((num_envs, num_sensors, v_lines, h_lines, 3), device=self.device)
        self.sensor_dist_tensor = torch.zeros((num_envs, num_sensors, v_lines, h_lines), device=self.device)
        self.sensor_pos_tensor = torch.zeros((num_envs, 3), device=self.device)
        # LidarSensor expects XYZW ordering
        self.sensor_quat_tensor_xyzw = torch.zeros((num_envs, 4), device=self.device)

        self.warp_tensor_dict = {
            "sensor_dist_tensor": self.sensor_dist_tensor,
            "device": str(self.device),
            "num_envs": num_envs,
            "num_sensors": num_sensors,
            "sensor_pos_tensor": self.sensor_pos_tensor,
            "sensor_quat_tensor": self.sensor_quat_tensor_xyzw,
            "mesh_ids": self.mesh_ids,
        }

        self.sensor = self.LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self._backend_ready = True

    def _init_backend_isaacsim(self):
        """Initialize LiDAR point cloud backend for Isaac Sim using Isaac Lab's LidarSensor.

        This attaches a LiDAR sensor (Livox/MID-360 pattern by default) to the specified
        robot link across all environments using the env-regex prim path. Point cloud is
        requested in the sensor (local) frame to allow consistent downstream transforms.
        """
        # Isaac Lab LiDAR imports (as in the user-provided code snippet)
        from isaaclab.sensors import LidarSensor, LidarSensorCfg
        from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg

        # Resolve a usable link name on this robot: prefer requested link; otherwise pick a base-like name or first body
        robot_name = self.robots[0].name
        body_names = self.handler.scene.articulations[robot_name].body_names

        resolved_link = None
        if body_names:
            # exact match or suffix match (support nested namespaces)
            for bn in body_names:
                if bn == self.link_name or bn.endswith("/" + self.link_name) or bn.split("/")[-1] == self.link_name:
                    resolved_link = bn
                    break
            if resolved_link is None:
                # If not present as a physics body (common when fixed joints are merged), try to find the prim in the USD stage
                # and mount the sensor to that Xform path instead.
                import omni
                from pxr import Usd

                stage = omni.usd.get_context().get_stage()
                # Find any env prim under /World/envs and search for robot + link under it
                envs_prim = stage.GetPrimAtPath("/World/envs")
                subpath = None
                if envs_prim and envs_prim.IsValid():
                    for env_prim in envs_prim.GetChildren():
                        if not env_prim.GetName().startswith("env_"):
                            continue
                        robot_prim_path = f"{env_prim.GetPath().pathString}/{robot_name}"
                        robot_prim = stage.GetPrimAtPath(robot_prim_path)
                        if not robot_prim or not robot_prim.IsValid():
                            continue
                        for p in Usd.PrimRange(robot_prim):
                            if p.GetName() == self.link_name:
                                full_path = p.GetPath().pathString
                                # compute relative subpath under robot prim
                                subpath = full_path[len(robot_prim_path) + 1 :]
                                break
                        if subpath is not None:
                            break
                if subpath is not None and len(subpath) > 0:
                    resolved_link = subpath
            if resolved_link is None:
                # heuristics for a reasonable base link name
                for cand in ("base", "trunk", "root", "chassis"):
                    for bn in body_names:
                        if bn == cand or bn.endswith("/" + cand) or bn.split("/")[-1] == cand:
                            resolved_link = bn
                            break
                    if resolved_link is not None:
                        break
                if resolved_link is None:
                    resolved_link = body_names[0]
        else:
            resolved_link = self.link_name

        self._resolved_link_name = resolved_link

        # Attach LiDAR to the resolved link across envs via env-regex path
        prim_path = f"/World/envs/env_.*/{robot_name}/{self._resolved_link_name}"

        # Configure a Livox/MID-360-like sensor, aligned in world with point cloud in local frame
        # Keep params conservative for performance; adjust as needed by caller
        lidar_cfg = LidarSensorCfg(
            prim_path=prim_path,
            offset=LidarSensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            ray_alignment="world",
            pattern_cfg=LivoxPatternCfg(sensor_type=self.sensor_type, samples=24000),
            mesh_prim_paths=["/World/ground", "/World/static"],
            # optionally include dynamic scene meshes: robot and default scene under each env
            dynamic_env_mesh_prim_paths=[f"/World/envs/env_.*/{robot_name}/.*", "/World/envs/env_.*/.*/geometry/mesh"],
            max_distance=20.0,
            min_range=0.2,
            return_pointcloud=True,            # request point cloud output
            pointcloud_in_world_frame=False,   # get local-frame points and transform ourselves
            enable_sensor_noise=False,
            update_frequency=25.0,
            debug_vis=False,
        )

        self._isaacsim_lidar = LidarSensor(lidar_cfg)
        # Register into scene sensors so it updates with scene.update()
        self.handler.scene.sensors["lidar_pointcloud"] = self._isaacsim_lidar
        self._backend_ready = True

    def isaacgym_call(self, robot_name: str):
        """IsaacGym-specific LiDAR point cloud computation."""
        # Refresh tensors and read rigid body state tensor (xyzw from IsaacGym)
        self.handler.gym.refresh_rigid_body_state_tensor(self.handler.sim)
        rb_states = self.handler._rigid_body_states  # (N_total_bodies, 13)

        # Resolve global rigid body indices for the target link once
        if not hasattr(self, "_gym_link_gidxs"):
            gidxs = []
            for i in range(self.num_envs):
                gidx = self.handler._env_rigid_body_global_indices[i]["robot"][self.link_name]
                gidxs.append(gidx)
            self._gym_link_gidxs = gidxs

        link_states = rb_states[self._gym_link_gidxs, :]
        pos_w = link_states[:, 0:3]
        quat_xyzw = link_states[:, 3:7]
        # Convert to (w,x,y,z)
        quat_wxyz = convert_quat(quat_xyzw, to="wxyz")

        return self._compute_lidar_points(robot_name, pos_w, quat_wxyz)

    def mujoco_call(self, robot_name: str):
        """MuJoCo-specific LiDAR point cloud computation."""
        # Resolve body id using cached names from handler (avoid mj_name2id signature mismatch)
        if not hasattr(self, "_mj_link_bid"):
            bid = None
            body_names = self.handler.body_names
            for i, bn in enumerate(body_names):
                if bn == self.link_name or bn.endswith("/" + self.link_name) or bn.split("/")[-1] == self.link_name:
                    bid = i
                    break
            if bid is None:
                warnings.warn(f"LidarPointCloud: link '{self.link_name}' not found in MuJoCo body names.")
                return {robot_name: None}
            self._mj_link_bid = int(bid)

        # MuJoCo xquat is (w,x,y,z)
        pos_np = self.handler.physics.data.xpos[self._mj_link_bid]
        quat_np = self.handler.physics.data.xquat[self._mj_link_bid]
        pos_w = torch.as_tensor(pos_np, device=self.device, dtype=torch.float32).view(1, 3)
        quat_wxyz = torch.as_tensor(quat_np, device=self.device, dtype=torch.float32).view(1, 4)

        return self._compute_lidar_points(robot_name, pos_w, quat_wxyz)

    def isaacsim_call(self, robot_name: str):
        """Isaac Sim LiDAR point cloud query using Isaac Lab LidarSensor.

        Returns a dict containing local and world point clouds for the target robot.
        """

        # Ensure the sensor produces up-to-date data; if scene already updates it, this is a no-op
        self._isaacsim_lidar.update(dt=0.0)

        data = getattr(self._isaacsim_lidar, "data", None)
        if data is None:
            warnings.warn("LidarPointCloud(isaacsim): LidarSensor has no data. Returning None.")
            return {robot_name: None}

        # Helper to pull a candidate attribute from sensor data
        def _get_first_attr(obj: object, names: list[str]):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return None

        # Try common field names for point cloud in local/world frames
        pts_local = _get_first_attr(
            data,
            [
                "pointcloud",  # common
                "point_cloud",
                "points_local",
                "points_l",
            ],
        )
        pts_world = _get_first_attr(
            data,
            [
                "pointcloud_world",
                "point_cloud_world",
                "points_world",
                "points_w",
            ],
        )

        # Convert to torch tensors on the handler's device
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(self.handler.device)
            return torch.as_tensor(x, device=self.handler.device)

        pts_local = _to_tensor(pts_local)
        pts_world = _to_tensor(pts_world)

        # If only local points are available, compute world points using the link pose (and optical center offset)
        if pts_world is None and pts_local is not None:
            # Resolve pose for the resolved link across envs
            art = self.handler.scene.articulations[robot_name]
            body_names = art.body_names
            # find index for resolved link
            link_idx = None
            for i, bn in enumerate(body_names):
                if (
                    bn == self._resolved_link_name
                    or bn.endswith("/" + self._resolved_link_name)
                    or bn.split("/")[-1] == self._resolved_link_name
                ):
                    link_idx = i
                    break
            if link_idx is None:
                link_idx = 0
            body_state = art.data.body_state_w  # (E, B, 13)
            pos_w = body_state[:, link_idx, 0:3]
            quat_wxyz = body_state[:, link_idx, 3:7]

            # Apply optical center offset if requested
            if self.apply_optical_center_offset and self.optical_center_offset_z != 0.0:
                offset_local = torch.tensor(
                    [0.0, 0.0, self.optical_center_offset_z], device=pos_w.device, dtype=pos_w.dtype
                ).view(1, 3)
                pos_w = pos_w + quat_apply(quat_wxyz, offset_local.repeat(pos_w.shape[0], 1))

            # pts_local may be (E, N, 3) or (N, 3) duplicated across envs; normalize to (E, N, 3)
            if pts_local.ndim == 2 and pts_local.size(-1) == 3:
                # broadcast same pattern to all envs
                pts_local = pts_local.unsqueeze(0).repeat(self.num_envs, 1, 1)
            elif pts_local.ndim == 3 and pts_local.shape[-1] == 3:
                # expected shape (E, N, 3)
                pass
            else:
                # Unknown shape; return None rather than crashing
                warnings.warn(
                    f"LidarPointCloud(isaacsim): Unsupported local point shape {tuple(pts_local.shape)}. Returning None."
                )
                return {robot_name: None}

            # Transform to world: R(q)*p + t
            E, N, _ = pts_local.shape
            quat_rep = quat_wxyz.unsqueeze(1).repeat(1, N, 1).view(-1, 4)
            vec = pts_local.reshape(-1, 3)
            rot = quat_apply(quat_rep, vec).view(E, N, 3)
            pos_rep = pos_w.unsqueeze(1).repeat(1, N, 1)
            pts_world = rot + pos_rep

        # If only world points are available, compute local points using inverse rotation
        if pts_local is None and pts_world is not None:
            art = self.handler.scene.articulations[robot_name]
            body_names = art.body_names
            link_idx = None
            for i, bn in enumerate(body_names):
                if (
                    bn == self._resolved_link_name
                    or bn.endswith("/" + self._resolved_link_name)
                    or bn.split("/")[-1] == self._resolved_link_name
                ):
                    link_idx = i
                    break
            if link_idx is None:
                link_idx = 0
            body_state = art.data.body_state_w
            pos_w = body_state[:, link_idx, 0:3]
            quat_wxyz = body_state[:, link_idx, 3:7]

            # Normalize pts_world to (E, N, 3)
            if pts_world.ndim == 2 and pts_world.size(-1) == 3:
                pts_world = pts_world.unsqueeze(0).repeat(self.num_envs, 1, 1)
            elif pts_world.ndim == 3 and pts_world.shape[-1] == 3:
                pass
            else:
                warnings.warn(
                    f"LidarPointCloud(isaacsim): Unsupported world point shape {tuple(pts_world.shape)}. Returning None."
                )
                return {robot_name: None}

            # Inverse rotation: local = R(q)^T * (p_w - t)
            E, N, _ = pts_world.shape
            p_rel = pts_world - pos_w.unsqueeze(1)
            # Conjugate quaternion for inverse rotation
            qw, qx, qy, qz = quat_wxyz.unbind(-1)
            quat_inv = torch.stack([qw, -qx, -qy, -qz], dim=-1)
            quat_rep = quat_inv.unsqueeze(1).repeat(1, N, 1).view(-1, 4)
            vec = p_rel.reshape(-1, 3)
            pts_local = quat_apply(quat_rep, vec).view(E, N, 3)

        # If neither points tensor is available, return None
        if pts_local is None and pts_world is None:
            warnings.warn("LidarPointCloud(isaacsim): No point cloud fields found in sensor data.")
            return {robot_name: None}

        return {
            robot_name: {
                "points_local": pts_local,
                "points_world": pts_world,
                "dist": None,  # not exposing raw distance buffer here
                "link": getattr(self, "_resolved_link_name", self.link_name),
            }
        }

    def _compute_lidar_points(self, robot_name: str, pos_w: torch.Tensor, quat_wxyz: torch.Tensor):
        """Common logic to compute LiDAR points from pose information."""
        # Apply optical center offset in the sensor's local +Z (after model-specific mounting rotations)
        if self.apply_optical_center_offset and self.optical_center_offset_z != 0.0:
            offset_local = torch.tensor([0.0, 0.0, self.optical_center_offset_z], device=pos_w.device).view(-1, 3)
            if offset_local.shape[0] != pos_w.shape[0]:
                offset_local = offset_local.repeat(pos_w.shape[0], 1)
            pos_w = pos_w + quat_apply(quat_wxyz, offset_local)

        # Update sensor pose buffers (LidarSensor expects XYZW)
        self.sensor_pos_tensor[:, :pos_w.shape[1]] = pos_w
        quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
        self.sensor_quat_tensor_xyzw[:, :quat_xyzw.shape[1]] = quat_xyzw

        # Run LiDAR update
        lidar_tensor_local, dist_tensor = self.sensor.update()

        # Compute world coordinates from local points
        # lidar_tensor_local: (E, S, V, H, 3)
        E, S, V, H, _ = lidar_tensor_local.shape
        pts_local = lidar_tensor_local.view(E, -1, 3)
        # Expand pose to match points
        quat_rep = quat_wxyz.unsqueeze(1).repeat(1, pts_local.shape[1], 1).view(-1, 4)
        vec = pts_local.view(-1, 3)
        rot = quat_apply(quat_rep, vec).view(E, -1, 3)
        pos_rep = pos_w.unsqueeze(1).repeat(1, pts_local.shape[1], 1)
        pts_world = rot + pos_rep
        pts_world = pts_world.view(E, S, V, H, 3)

        return {
            robot_name: {
                "points_local": lidar_tensor_local,
                "points_world": pts_world,
                "dist": dist_tensor,
                "link": self.link_name,
            }
        }

    def __call__(self):
        if not self.enabled:
            return {self.robots[0].name: None}

        robot_name = self.robots[0].name
        sim_type = self.simulator

        if sim_type == "isaacgym":
            return self.isaacgym_call(robot_name)
        elif sim_type == "mujoco":
            return self.mujoco_call(robot_name)
        elif sim_type == "isaacsim":
            return self.isaacsim_call(robot_name)
        else:
            warnings.warn(f"LidarPointCloud: simulator '{sim_type}' not supported for LiDAR pose fetch.")
            return {robot_name: None}
