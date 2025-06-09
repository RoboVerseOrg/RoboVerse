from __future__ import annotations

import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
from loguru import logger as log

from metasim.cfg.objects import (
    ArticulationObjCfg,
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
    _FileBasedMixin,
)
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


class IsaacgymHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []
        self._robot_names = {self.robot.name}
        self._robot_init_pose = self.robot.default_position
        self._robot_init_quat = self.robot.default_orientation
        self._cameras = scenario.cameras

        self.gym = None
        self.sim = None
        self.viewer = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_envs: int = scenario.num_envs
        self._episode_length_buf = [0 for _ in range(self.num_envs)]

        # asset related
        self._asset_dict_dict: dict = {}  # dict of object link index dict
        self._articulated_asset_dict_dict: dict = {}  # dict of articulated object link index dict
        self._articulated_joint_dict_dict: dict = {}  # dict of articulated object joint index dict
        self._robot_link_dict: dict = {}  # dict of robot link index dict
        self._robot_joint_dict: dict = {}  # dict of robot joint index dict
        self._joint_info: dict = {}  # dict of joint names of each env
        self._num_joints: int = 0
        self._body_info: dict = {}  # dict of body names of each env
        self._num_bodies: int = 0

        # environment related pointers
        self._envs: list = []
        self._obj_handles: list = []  # 2 dim list: list in list, each list contains object handles of each env
        self._articulated_obj_handles: list = []  # 2 dim list: list in list, each list contains articulated object handles of each env
        self._robot_handles: list = []  # 2 dim list: list of robot handles of each env

        # environment related tensor indices
        self._env_rigid_body_global_indices: list = []  # 2 dim list: list in list, each list contains global indices of each env

        # will update after refresh
        self._root_states: torch.Tensor | None = None
        self._dof_states: torch.Tensor | None = None
        self._rigid_body_states: torch.Tensor | None = None
        self._robot_dof_state: torch.Tensor | None = None

        # control related
        self._robot_num_dof: int  # number of robot dof
        self._obj_num_dof: int = 0  # number of object dof
        self._actions: torch.Tensor | None = None
        self._action_scale: torch.Tensor | None = (
            None  # for configuration: desire_pos = action_scale * action + default_pos
        )
        self._default_dof_pos: torch.Tensor | None = (
            None  # for the configuration: desire_pos = action_scale * action + default_pos
        )
        self._action_offset: bool = False  # for configuration: desire_pos = action_scale * action + default_pos
        self._p_gains: torch.Tensor | None = None  # parameter for PD controller in for pd effort control
        self._d_gains: torch.Tensor | None = None
        self._torque_limits: torch.Tensor | None = None
        self._effort: torch.Tensor | None = None  # output of pd controller, used for effort control
        self._pos_ctrl_dof_dix = []  # joint index in dof state, built-in position control mode
        self._manual_pd_on: bool = False  # turn on maunual pd controller if effort joint exist

    def launch(self) -> None:
        ## IsaacGym Initialization
        self._init_gym()
        self._make_envs()
        self._set_up_camera()
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self._root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_states = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._rigid_body_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self._robot_dof_state = self._dof_states.view(self._num_envs, -1, 2)[:, self._obj_num_dof :]

    def _init_gym(self) -> None:
        physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()
        # configure sim
        # TODO move more params into sim_params cfg
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        if self.scenario.sim_params.dt is not None:
            # IsaacGym has a different dt definition than IsaacLab, see https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#simulation-config
            sim_params.dt = self.scenario.sim_params.dt * self.scenario.decimation
        sim_params.substeps = self.scenario.decimation
        sim_params.use_gpu_pipeline = self.scenario.sim_params.use_gpu_pipeline
        sim_params.physx.solver_type = self.scenario.sim_params.solver_type
        sim_params.physx.num_position_iterations = self.scenario.sim_params.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.scenario.sim_params.num_velocity_iterations
        sim_params.physx.rest_offset = self.scenario.sim_params.rest_offset
        sim_params.physx.contact_offset = self.scenario.sim_params.contact_offset
        sim_params.physx.friction_offset_threshold = self.scenario.sim_params.friction_offset_threshold
        sim_params.physx.friction_correlation_distance = self.scenario.sim_params.friction_correlation_distance
        sim_params.physx.num_threads = self.scenario.sim_params.num_threads
        sim_params.physx.use_gpu = self.scenario.sim_params.use_gpu
        sim_params.physx.bounce_threshold_velocity = self.scenario.sim_params.bounce_threshold_velocity

        compute_device_id = 0
        graphics_device_id = 0
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

    def _set_up_camera(self) -> None:
        self._depth_tensors = []
        self._rgb_tensors = []
        self._seg_tensors = []
        self._vinv_mats = []
        self._proj_mats = []
        self._camera_handles = []
        self._env_origin = []
        for i_env in range(self.num_envs):
            self._depth_tensors.append([])
            self._rgb_tensors.append([])
            self._seg_tensors.append([])
            self._vinv_mats.append([])
            self._proj_mats.append([])
            self._env_origin.append([])

            origin = self.gym.get_env_origin(self._envs[i_env])
            self._env_origin[i_env] = [origin.x, origin.y, origin.z]
            for cam_cfg in self.cameras:
                camera_props = gymapi.CameraProperties()
                camera_props.width = cam_cfg.width
                camera_props.height = cam_cfg.height
                camera_props.horizontal_fov = cam_cfg.horizontal_fov
                camera_props.near_plane = cam_cfg.clipping_range[0]
                camera_props.far_plane = cam_cfg.clipping_range[1]
                camera_props.enable_tensors = True
                camera_handle = self.gym.create_camera_sensor(self._envs[i_env], camera_props)
                self._camera_handles.append(camera_handle)

                camera_eye = gymapi.Vec3(*cam_cfg.pos)
                camera_lookat = gymapi.Vec3(*cam_cfg.look_at)
                self.gym.set_camera_location(camera_handle, self._envs[i_env], camera_eye, camera_lookat)

                camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_DEPTH
                )
                camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_COLOR
                )
                camera_tensor_rgb_seg = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_SEGMENTATION
                )
                torch_cam_depth_tensor = gymtorch.wrap_tensor(camera_tensor_depth)
                torch_cam_rgb_tensor = gymtorch.wrap_tensor(camera_tensor_rgb)
                torch_cam_rgb_seg_tensor = gymtorch.wrap_tensor(camera_tensor_rgb_seg)

                cam_vinv = torch.inverse(
                    torch.tensor(self.gym.get_camera_view_matrix(self.sim, self._envs[i_env], camera_handle))
                ).to(self.device)
                cam_proj = torch.tensor(
                    self.gym.get_camera_proj_matrix(self.sim, self._envs[i_env], camera_handle),
                    device=self.device,
                )

                self._depth_tensors[i_env].append(torch_cam_depth_tensor)
                self._rgb_tensors[i_env].append(torch_cam_rgb_tensor)
                self._seg_tensors[i_env].append(torch_cam_rgb_seg_tensor)
                self._vinv_mats[i_env].append(cam_vinv)
                self._proj_mats[i_env].append(cam_proj)

    def _load_object_asset(self, object: BaseObjCfg) -> None:
        asset_root = "."
        if isinstance(object, PrimitiveCubeCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.create_box(self.sim, object.size[0], object.size[1], object.size[2], asset_options)
        elif isinstance(object, PrimitiveSphereCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.create_sphere(self.sim, object.radius, asset_options)

        elif isinstance(object, ArticulationObjCfg):
            asset_path = object.mjcf_path if object.isaacgym_read_mjcf else object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)
            self._articulated_asset_dict_dict[object.name] = self.gym.get_asset_rigid_body_dict(asset)
            self._articulated_joint_dict_dict[object.name] = self.gym.get_asset_dof_dict(asset)
        elif isinstance(object, RigidObjCfg):
            asset_path = object.mjcf_path if object.isaacgym_read_mjcf else object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = object.fix_base_link
            asset_options.disable_gravity = False
            asset_options.flip_visual_attachments = False
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)

        asset_link_dict = self.gym.get_asset_rigid_body_dict(asset)
        self._asset_dict_dict[object.name] = asset_link_dict
        self._obj_num_dof += self.gym.get_asset_dof_count(asset)
        return asset

    def _load_robot_assets(self) -> None:
        asset_root = "."
        robot_asset_file = self.robot.mjcf_path if self.robot.isaacgym_read_mjcf else self.robot.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = self.robot.fix_base_link
        asset_options.disable_gravity = not self.robot.enabled_gravity
        asset_options.flip_visual_attachments = self.robot.isaacgym_flip_visual_attachments
        asset_options.collapse_fixed_joints = self.robot.collapse_fixed_joints
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # Defaults are set to free movement and will be updated based on the configuration in actuator_cfg below.
        asset_options.replace_cylinder_with_capsule = self.scenario.sim_params.replace_cylinder_with_capsule
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
        # configure robot dofs
        robot_num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self._robot_num_dof = robot_num_dofs

        self._action_scale = torch.tensor(self.scenario.control.action_scale, device=self.device)
        self._action_offset = self.scenario.control.action_offset

        self._p_gains = torch.zeros(
            self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._d_gains = torch.zeros(
            self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._torque_limits = torch.zeros(
            self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)
        num_actions = 0
        default_dof_pos = []
        self._manual_pd_on = any(mode == "effort" for mode in self.robot.control_type.values())

        dof_names = self.gym.get_asset_dof_names(robot_asset)
        for i, dof_name in enumerate(dof_names):
            # get config
            i_actuator_cfg = self.robot.actuators[dof_name]
            i_control_mode = self.robot.control_type[dof_name] if dof_name in self.robot.control_type else "position"

            # task default position from cfg if exist, otherwise use 0.3*(uppper + lower) as default
            if not i_actuator_cfg.is_ee:
                default_dof_pos_i = (
                    self.robot.default_joint_positions[dof_name]
                    if dof_name in self.robot.default_joint_positions
                    else robot_mids[i]
                )
                default_dof_pos.append(default_dof_pos_i)
            # for end effector, always use open as default position
            else:
                default_dof_pos.append(robot_upper_limits[i])

            # pd control effort mode
            if i_control_mode == "effort":
                self._p_gains[:, i] = i_actuator_cfg.stiffness
                self._d_gains[:, i] = i_actuator_cfg.damping
                torque_limit = (
                    i_actuator_cfg.torque_limit
                    if i_actuator_cfg.torque_limit is not None
                    else torch.tensor(robot_dof_props["effort"][i], dtype=torch.float, device=self.device)
                )
                self._torque_limits[:, i] = self.scenario.control.torque_limit_scale * torque_limit
                robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props["stiffness"][i] = 0.0
                robot_dof_props["damping"][i] = 0.0

            # built-in position mode
            elif i_control_mode == "position":
                robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                if i_actuator_cfg.stiffness is not None:
                    robot_dof_props["stiffness"][i] = i_actuator_cfg.stiffness
                if i_actuator_cfg.damping is not None:
                    robot_dof_props["damping"][i] = i_actuator_cfg.damping
                self._pos_ctrl_dof_dix.append(i + self._obj_num_dof)
            else:
                log.error(f"Unknown actuator control mode: {i_control_mode}, only support effort and position")
                raise ValueError

            if i_actuator_cfg.fully_actuated:
                num_actions += 1

        self._default_dof_pos = torch.tensor(default_dof_pos, device=self.device).unsqueeze(0)
        self.actions = torch.zeros([self._num_envs, num_actions], device=self.device)

        # # get link index of panda hand, which we will use as end effector
        self._robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self._robot_joint_dict = self.gym.get_asset_dof_dict(robot_asset)

        return robot_asset, robot_dof_props

    def _make_envs(
        self,
    ) -> None:
        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        log.info("Creating %d environments" % self.num_envs)

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*self._robot_init_pose)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # get object and robot asset
        obj_assets_list = [self._load_object_asset(obj) for obj in self.objects]
        robot_asset, robot_dof_props = self._load_robot_assets()

        #### Joint Info ####
        for art_obj_name, art_obj_joint_dict in self._articulated_joint_dict_dict.items():
            num_joints = len(art_obj_joint_dict)
            joint_names_ = []
            for joint_i in range(num_joints):
                for joint_name, joint_idx in art_obj_joint_dict.items():
                    if joint_idx == joint_i:
                        joint_names_.append(joint_name)
            assert len(joint_names_) == num_joints
            joint_info_ = {}
            joint_info_["names"] = joint_names_
            joint_info_["local_indices"] = art_obj_joint_dict
            art_obj_joint_dict_global = {k_: v_ + self._num_joints for k_, v_ in art_obj_joint_dict.items()}
            joint_info_["global_indices"] = art_obj_joint_dict_global
            self._num_joints += num_joints
            self._joint_info[art_obj_name] = joint_info_

        # robot
        num_joints = len(self._robot_joint_dict)
        joint_names_ = []
        for joint_i in range(num_joints):
            for joint_name, joint_idx in self._robot_joint_dict.items():
                if joint_idx == joint_i:
                    joint_names_.append(joint_name)

        assert len(joint_names_) == num_joints
        joint_info_ = {}
        joint_info_["names"] = joint_names_
        joint_info_["local_indices"] = self._robot_joint_dict
        joint_info_["global_indices"] = {k_: v_ + self._num_joints for k_, v_ in self._robot_joint_dict.items()}
        self._joint_info[self.robot.name] = joint_info_
        self._num_joints += num_joints

        ###################
        #### Body Info ####
        for obj_name, asset_dict in self._asset_dict_dict.items():
            num_bodies = len(asset_dict)
            rigid_body_names = []
            for i in range(num_bodies):
                for rigid_body_name, rigid_body_idx in asset_dict.items():
                    if rigid_body_idx == i:
                        rigid_body_names.append(rigid_body_name)
            assert len(rigid_body_names) == num_bodies
            body_info_ = {}
            body_info_["name"] = rigid_body_names
            body_info_["local_indices"] = asset_dict
            body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in asset_dict.items()}
            self._body_info[obj_name] = body_info_
            self._num_bodies += num_bodies

        num_bodies = len(self._robot_link_dict)
        rigid_body_names = []
        for i in range(num_bodies):
            for rigid_body_name, rigid_body_idx in self._robot_link_dict.items():
                if rigid_body_idx == i:
                    rigid_body_names.append(rigid_body_name)

        assert len(rigid_body_names) == num_bodies
        rigid_body_info_ = {}
        rigid_body_info_["name"] = rigid_body_names
        rigid_body_info_["local_indices"] = self._robot_link_dict
        rigid_body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in self._robot_link_dict.items()}
        self._body_info[self.robot.name] = rigid_body_info_
        self._num_bodies += num_bodies

        #################

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            ##  state update  ##
            self._envs.append(env)
            self._obj_handles.append([])
            self._env_rigid_body_global_indices.append({})
            self._articulated_obj_handles.append([])
            ####################

            # carefully set each object
            for obj_i, obj_asset in enumerate(obj_assets_list):
                # add object
                obj_pose = gymapi.Transform()
                obj_pose.p.x = obj_i * 0.2  # place to any position, will update immediately at reset stage
                obj_pose.p.y = 0.0
                obj_pose.p.z = 0.0
                obj_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
                obj_handle = self.gym.create_actor(env, obj_asset, obj_pose, "object", i, 0)

                if isinstance(self.objects[obj_i], _FileBasedMixin):
                    self.gym.set_actor_scale(env, obj_handle, self.objects[obj_i].scale[0])
                elif isinstance(self.objects[obj_i], PrimitiveCubeCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], PrimitiveSphereCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], RigidObjCfg):
                    pass
                elif isinstance(self.objects[obj_i], ArticulationObjCfg):
                    self._articulated_obj_handles[-1].append(obj_handle)
                else:
                    log.error("Unknown object type")
                    raise NotImplementedError
                self._obj_handles[-1].append(obj_handle)

                object_rigid_body_indices = {}
                for rigid_body_name, rigid_body_idx in self._asset_dict_dict[self.objects[obj_i].name].items():
                    rigid_body_idx = self.gym.find_actor_rigid_body_index(
                        env, obj_handle, rigid_body_name, gymapi.DOMAIN_SIM
                    )
                    object_rigid_body_indices[rigid_body_name] = rigid_body_idx

                self._env_rigid_body_global_indices[-1][self.objects[obj_i].name] = object_rigid_body_indices

            # # carefully add robot
            robot_handle = self.gym.create_actor(env, robot_asset, robot_pose, "robot", i, 2)
            self.gym.set_actor_scale(env, robot_handle, self.robot.scale[0])
            self._robot_handles.append(robot_handle)
            # set dof properties
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            robot_rigid_body_indices = {}
            for rigid_body_name, rigid_body_idx in self._robot_link_dict.items():
                rigid_body_idx = self.gym.find_actor_rigid_body_index(
                    env, robot_handle, rigid_body_name, gymapi.DOMAIN_SIM
                )
                robot_rigid_body_indices[rigid_body_name] = rigid_body_idx

            self._env_rigid_body_global_indices[-1]["robot"] = robot_rigid_body_indices

        # GET initial state, copy for reset later
        self._initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        ###### set VEWIER camera ######
        # point camera at middle env
        if not self.headless:  # TODO: update a default viewer
            cam_pos = gymapi.Vec3(1, 1, 1)
            cam_target = gymapi.Vec3(-1, -1, -0.5)
            middle_env = self._envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        ################################

    def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj_id, obj in enumerate(self.objects):
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
                body_ids_reindex = self._get_body_ids_reindex(obj.name)
                state = ObjectState(
                    root_state=self._root_states.view(self.num_envs, -1, 13)[:, obj_id, :],
                    body_names=self.get_body_names(obj.name),
                    body_state=self._rigid_body_states.view(self.num_envs, -1, 13)[:, body_ids_reindex, :],
                    joint_pos=self._dof_states.view(self.num_envs, -1, 2)[:, joint_reindex, 0],
                    joint_vel=self._dof_states.view(self.num_envs, -1, 2)[:, joint_reindex, 1],
                )
            else:
                state = ObjectState(
                    root_state=self._root_states.view(self.num_envs, -1, 13)[:, obj_id, :],
                )
            object_states[obj.name] = state

        # FIXME some RL task need joint state as dof_pos - default_dof_pos, not absolute dof_pos. see https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L216 for further details
        robot_states = {}
        for robot_id, robot in enumerate([self.robot]):
            joint_reindex = self.get_joint_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)
            state = RobotState(
                # HACK: robot is always after objects
                root_state=self._root_states.view(self.num_envs, -1, 13)[:, len(self.objects) + robot_id, :],
                body_names=self.get_body_names(robot.name),
                body_state=self._rigid_body_states.view(self.num_envs, -1, 13)[:, body_ids_reindex, :],
                joint_pos=self._dof_states.view(self.num_envs, -1, 2)[:, joint_reindex, 0],
                joint_vel=self._dof_states.view(self.num_envs, -1, 2)[:, joint_reindex, 1],
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=self._effort if self._manual_pd_on else None,
            )
            robot_states[robot.name] = state

        camera_states = {}
        self.gym.start_access_image_tensors(self.sim)
        for cam_id, cam in enumerate(self.cameras):
            state = CameraState(
                rgb=torch.stack([self._rgb_tensors[env_id][cam_id][..., :3] for env_id in env_ids]),
                depth=torch.stack([self._depth_tensors[env_id][cam_id] for env_id in env_ids]),
            )
            camera_states[cam.name] = state
        self.gym.end_access_image_tensors(self.sim)

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={})

    @property
    def episode_length_buf(self) -> list[int]:
        return self._episode_length_buf

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def _get_action_array_all(self, actions: list[Action]):
        action_array_list = []

        for action_data in actions:
            flat_vals = []
            for joint_i, joint_name in enumerate(self._joint_info[self.robot.name]["names"]):
                if self.robot.actuators[joint_name].fully_actuated:
                    flat_vals.append(
                        action_data[self.robot.name]["dof_pos_target"][joint_name]
                    )  # TODO: support other actions
                else:
                    flat_vals.append(0.0)  # place holder for under-actuated joints

            action_array = torch.tensor(flat_vals, dtype=torch.float32, device=self.device).unsqueeze(0)

            action_array_list.append(action_array)
        action_array_all = torch.cat(action_array_list, dim=0)
        return action_array_all

    def set_dof_targets(self, obj_name: str, actions: list[Action]):
        self._actions_cache = actions
        action_input = torch.zeros_like(self._dof_states[:, 0])
        action_array_all = self._get_action_array_all(actions)
        robot_dim = action_array_all.shape[1]

        assert (
            action_input.shape[0] % self._num_envs == 0
        )  # WARNING: obj dim(env0), robot dim(env0), obj dim(env1), robot dim(env1) ...
        chunk_size = action_input.shape[0] // self._num_envs
        robot_dim_index = [
            i * chunk_size + offset
            for i in range(self.num_envs)
            for offset in range(chunk_size - robot_dim, chunk_size)
        ]
        action_input[robot_dim_index] = action_array_all.float().to(self.device).reshape(-1)

        # if any effort joint exist, set pd controller's target position for later effort calculation
        if self._manual_pd_on:
            actions_reshape = action_input.view(self._num_envs, self._obj_num_dof + self._robot_num_dof)
            self.actions = actions_reshape[:, self._obj_num_dof :]
            # and set position target for position actuator if any exist
            if len(self._pos_ctrl_dof_dix) > 0:
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

        # directly set position target
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

    def refresh_render(self) -> None:
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh cameras and viewer
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # TODO add keyboard callback(mostly likely push v) to stop rendering in render mode
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def _simulate_one_physics_step(self, action):
        # for pd control joints by effort api, update torque and step the physics
        if self._manual_pd_on:
            self._apply_pd_control(action)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for position control joints, just step the physics
        else:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

    def simulate(self) -> None:
        # Step the physics
        self._simulate_one_physics_step(self.actions)
        # Refresh tensors
        if not self._manual_pd_on:
            self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh cameras and viewer
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)

        # self.gym.sync_frame_time(self.sim)

    def _compute_effort(self, actions):
        """Compute effort from actions"""
        # scale the actions (generally output from policy)
        action_scaled = self._action_scale * actions
        robot_dof_pos = self._robot_dof_state[..., 0]
        robot_dof_vel = self._robot_dof_state[..., 1]
        if self._action_offset:
            _effort = (
                self._p_gains * (action_scaled + self._default_dof_pos - robot_dof_pos) - self._d_gains * robot_dof_vel
            )
        else:
            _effort = self._p_gains * (action_scaled - robot_dof_pos) - self._d_gains * robot_dof_vel
        self._effort = torch.clip(_effort, -self._torque_limits, self._torque_limits)
        effort = self._effort.to(torch.float32)
        return effort

    def _apply_pd_control(self, actions):
        """
        Compute torque using pd controller for effort actuator and set torque.
        """
        effort = self._compute_effort(actions)

        # NOTE: effort passed set_dof_actuation_force_tensor() must have the same dimension as the number of DOFs, even if some DOFs are not actionable.
        if self._obj_num_dof > 0:
            obj_force_placeholder = torch.zeros((self._num_envs, self._obj_num_dof), device=self.device)
            effort = torch.cat((obj_force_placeholder, effort), dim=1)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort))

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None):
        ## Support setting status only for specified env_ids
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert len(states) == self.num_envs, (
            f"The length of the state list ({len(states)}) must match the length of num_envs ({self.num_envs})."
        )

        pos_list = []
        rot_list = []
        q_list = []
        states_flat = [{**states[i]["objects"], **states[i]["robots"]} for i in env_ids]

        # Prepare state data for specified env_ids
        env_indices = {env_id: i for i, env_id in enumerate(env_ids)}

        for i in range(self.num_envs):
            if i not in env_indices:
                continue

            state_idx = env_indices[i]
            state = states_flat[state_idx]

            pos_list_i = []
            rot_list_i = []
            q_list_i = []
            for obj in self.objects:
                obj_name = obj.name
                pos = np.array(state[obj_name].get("pos", [0.0, 0.0, 0.0]))
                rot = np.array(state[obj_name].get("rot", [1.0, 0.0, 0.0, 0.0]))
                obj_quat = [rot[1], rot[2], rot[3], rot[0]]  # IsaacGym convention

                pos_list_i.append(pos)
                rot_list_i.append(obj_quat)
                if isinstance(obj, ArticulationObjCfg):
                    obj_joint_q = np.zeros(len(self._articulated_joint_dict_dict[obj_name]))
                    articulated_joint_dict = self._articulated_joint_dict_dict[obj_name]
                    for joint_name, joint_idx in articulated_joint_dict.items():
                        if "dof_pos" in state[obj_name]:
                            obj_joint_q[joint_idx] = state[obj_name]["dof_pos"][joint_name]
                        else:
                            log.warning(f"No dof_pos for {joint_name} in {obj_name}")
                            obj_joint_q[joint_idx] = 0.0
                    q_list_i.append(obj_joint_q)

            pos_list_i.append(np.array(state[self.robot.name].get("pos", [0.0, 0.0, 0.0])))
            rot = np.array(state[self.robot.name].get("rot", [1.0, 0.0, 0.0, 0.0]))
            robot_quat = [rot[1], rot[2], rot[3], rot[0]]
            rot_list_i.append(robot_quat)

            robot_dof_state_i = np.zeros(len(self._robot_joint_dict))
            if "dof_pos" in state[self.robot.name]:
                for joint_name, joint_idx in self._robot_joint_dict.items():
                    robot_dof_state_i[joint_idx] = state[self.robot.name]["dof_pos"][joint_name]
            else:
                for joint_name, joint_idx in self._robot_joint_dict.items():
                    robot_dof_state_i[joint_idx] = (
                        self.robot.joint_limits[joint_name][0] + self.robot.joint_limits[joint_name][1]
                    ) / 2

            q_list_i.append(robot_dof_state_i)
            pos_list.append(pos_list_i)
            rot_list.append(rot_list_i)
            q_list.append(q_list_i)

        self._set_actor_root_state(pos_list, rot_list, env_ids)
        self._set_actor_joint_state(q_list, env_ids)

        self.gym.simulate(self.sim)  # FIXME: update the state, but has the side effect of stepping the physics
        # Refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # reset all env_id action to default
        self.actions[env_ids] = 0.0

    def _set_actor_root_state(self, position_list, rotation_list, env_ids):
        new_root_states = self._root_states.clone()
        actor_indices = []

        # Only modify the positions and rotations for the specified env_ids
        for i, env_id in enumerate(env_ids):
            env_offset = env_id * (len(self.objects) + 1)  # objects + robot
            for j in range(len(self.objects) + 1):
                actor_idx = env_offset + j
                new_root_states[actor_idx, :3] = torch.tensor(
                    position_list[i][j], dtype=torch.float32, device=self.device
                )
                new_root_states[actor_idx, 3:7] = torch.tensor(
                    rotation_list[i][j], dtype=torch.float32, device=self.device
                )
                new_root_states[actor_idx, 7:13] = torch.zeros(6, dtype=torch.float32, device=self.device)
            actor_indices.extend(range(env_offset, env_offset + len(self.objects) + 1))

        # Convert the actor indices to a tensor
        root_reset_actors_indices = torch.tensor(actor_indices, dtype=torch.int32, device=self.device)

        # Use indexed setting to set the root state
        res = self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(new_root_states),
            gymtorch.unwrap_tensor(root_reset_actors_indices),
            len(root_reset_actors_indices),
        )
        assert res

        return

    def _set_actor_joint_state(self, joint_pos_list, env_ids):
        new_dof_states = self._dof_states.clone()

        # Calculate the indices of DOFs in the tensor
        dof_indices = []
        new_dof_pos_values = []

        for i, env_id in enumerate(env_ids):
            # Get the joint positions for this environment
            flat_vals = []
            for obj_joints in joint_pos_list[i]:
                flat_vals.extend(obj_joints)

            # Calculate the indices of DOFs in the global DOF tensor
            dof_start_idx = env_id * self._num_joints
            for j, val in enumerate(flat_vals):
                dof_idx = dof_start_idx + j
                dof_indices.append(dof_idx)
                new_dof_pos_values.append(val)

        # Update the DOF positions for the specified indices
        dof_indices_tensor = torch.tensor(dof_indices, dtype=torch.int64, device=self.device)
        new_dof_pos_tensor = torch.tensor(new_dof_pos_values, dtype=torch.float32, device=self.device)

        # Update the positions and velocities (set velocities to 0)
        new_dof_states[dof_indices_tensor, 0] = new_dof_pos_tensor
        new_dof_states[dof_indices_tensor, 1] = 0.0

        # Apply the updated DOF state
        res = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(new_dof_states))
        assert res

        return

    def close(self) -> None:
        try:
            self.gym.destroy_sim(self.sim)
            self.gym.destroy_viewer(self.viewer)
            self.gym = None
            self.sim = None
            self.viewer = None
        except Exception as e:
            log.error(f"Error closing IsaacGym environment: {e}")
            pass

    ############################################################
    ## Utils
    ############################################################
    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = list(self._joint_info[obj_name]["global_indices"].keys())
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            body_names = self._body_info[obj_name]["name"]
            if sort:
                body_names.sort()
            return body_names
        else:
            return []

    def _get_body_ids_reindex(self, obj_name: str) -> list[int]:
        return [self._body_info[obj_name]["global_indices"][bn] for bn in self.get_body_names(obj_name)]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return self._device


# TODO: try to align handler API and use GymWrapper instead
IsaacgymEnv: type[EnvWrapper[IsaacgymHandler]] = GymEnvWrapper(IsaacgymHandler)
