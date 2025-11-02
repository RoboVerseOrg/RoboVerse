"""Implemention of Pybullet Handler.

This file contains the implementation of PybulletHandler, which is a subclass of BaseSimHandler.
PybulletHandler is used to handle the simulation environment using Pybullet.
Currently using Pybullet 3.2.6
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pybullet as p
import pybullet_data
import torch
from loguru import logger as log

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import Action, DictEnvState
from metasim.utils.math import convert_quat
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, adapt_actions_to_dict

# Optional: RoboSplatter imports for GS background rendering
try:
    from robo_splatter.models.camera import Camera as SplatCamera

    ROBO_SPLATTER_AVAILABLE = True
except ImportError:
    ROBO_SPLATTER_AVAILABLE = False
    log.warning("RoboSplatter not available. GS background rendering will be disabled.")


PYBULLET_DEFAULT_POSITION_GAIN = 0.1
PYBULLET_DEFAULT_VELOCITY_GAIN = 1.0
PYBULLET_DEFAULT_JOINT_MAX_VEL = 1.0
PYBULLET_DEFAULT_JOINT_MAX_TORQUE = 1.0

DEPTH_EPSILON = 1e-8


class SinglePybulletHandler(BaseSimHandler):
    """Pybullet Handler class."""

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        """Initialize the Pybullet Handler.

        Args:
            scenario: The scenario configuration
            num_envs: Number of environments
            headless: Whether to run the simulation in headless mode
        """
        super().__init__(scenario, optional_queries)
        self._actions_cache: list[Action] = []

    def _get_camera_params(self, view_matrix, projection_matrix, width, height):
        """Get camera intrinsics and extrinsics directly from PyBullet matrices.

        Args:
            view_matrix: PyBullet view matrix (16-element list, column-major)
            projection_matrix: PyBullet projection matrix (16-element list)
            width: image width
            height: image height

        Returns:
            Ks: (3, 3) intrinsic matrix
            c2w: (4, 4) camera-to-world transformation matrix
        """
        # Extrinsics: convert view matrix (world-to-camera) to c2w
        view_mat_np = np.array(view_matrix).reshape(4, 4).T
        c2w = np.linalg.inv(view_mat_np)

        # Intrinsics: extract from projection matrix
        proj_mat = np.array(projection_matrix).reshape(4, 4).T
        fx = proj_mat[0, 0] * width / 2.0
        fy = proj_mat[1, 1] * height / 2.0
        cx = width / 2.0
        cy = height / 2.0
        Ks = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return Ks, c2w

    def _convert_depth_buffer(self, depth_buffer: np.ndarray, near_plane: float, far_plane: float) -> np.ndarray:
        """Convert PyBullet depth buffer [0,1] to metric depth using near/far planes.

        PyBullet returns a non-linear depth buffer z_b in [0,1]. The eye-space
        distance (metric depth) is computed as:
            depth = (far * near) / (far - (far - near) * z_b)
        Args:
            depth_buffer: The depth buffer from PyBullet
            near_plane: The near plane of the camera
            far_plane: The far plane of the camera

        Returns:
            The metric depth
        """
        denom = far_plane - (far_plane - near_plane) * depth_buffer
        depth_metric = (far_plane * near_plane) / np.clip(denom, a_min=DEPTH_EPSILON, a_max=None)
        return depth_metric.astype(np.float32, copy=False)

    def _build_pybullet(self):
        self.client = p.connect(p.DIRECT if self.headless else p.GUI)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1.0 / 240.0,
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.scenario.gravity)

        # add plane
        self.plane_id = p.loadURDF("plane.urdf")
        # add agents
        self.object_ids = {}
        self.object_joint_order = {}
        self.camera_ids = {}
        for camera in self.cameras:
            # Camera parameters
            width, height = camera.width, camera.height

            # Camera position and target
            cam_eye = camera.pos  # Camera position in world coordinates
            cam_target = camera.look_at  # Where the camera is looking at
            cam_up = [0, 0, 1]  # Up vector

            # Compute view matrix
            view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
            # Projection matrix (perspective camera)
            fov = camera.vertical_fov  # Field of view in degrees
            aspect = width / height  # Aspect ratio
            near_plane = camera.clipping_range[0]  # Near clipping plane
            far_plane = camera.clipping_range[1]  # Far clipping plane
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

            self.camera_ids[camera.name] = (width, height, view_matrix, projection_matrix, near_plane, far_plane)

        for object in [*self.objects, *self.robots]:
            if isinstance(object, (ArticulationObjCfg, RobotCfg)):
                pos = np.asarray(object.default_position)
                rot = convert_quat(np.asarray(object.default_orientation), to="xyzw")
                object_file = object.urdf_path
                useFixedBase = object.fix_base_link
                flags = 0
                flags = flags | p.URDF_USE_SELF_COLLISION
                flags = flags | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                flags = flags | p.URDF_MAINTAIN_LINK_ORDER
                if True:  # object.collapse_fixed_joints :
                    flags = flags | p.URDF_MERGE_FIXED_LINKS

                curr_id = p.loadURDF(
                    object_file,
                    pos,
                    rot,
                    useFixedBase=useFixedBase,
                    flags=flags,
                    globalScaling=object.scale[0],
                )
                self.object_ids[object.name] = curr_id

                num_joints = p.getNumJoints(curr_id)
                cur_joint_names = [p.getJointInfo(curr_id, i)[1].decode("utf-8") for i in range(num_joints)]
                self.object_joint_order[object.name] = cur_joint_names

                # assert(num_joints == len(agent.dof.low))
                # assert(num_joints == len(agent.dof.high))
                # print(num_joints, agent.dof.init)
                # print(p.getJointStates(agent.instance, range(num_joints)))
                # assert(num_joints == len(object.dof.init))

                jointIndices = range(num_joints)

                if isinstance(object, RobotCfg):
                    # p.setJointMotorControlMultiDofArray(
                    #     curr_id,
                    #     jointIndices,
                    #     p.POSITION_CONTROL,
                    #     # targetPositions = agent.dof.target
                    # )
                    for j in range(num_joints):
                        joint_config = object.actuators[cur_joint_names[j]]
                        default_target = object.default_joint_positions.get(cur_joint_names[j], 0.0)
                        p.setJointMotorControl2(
                            bodyIndex=curr_id,
                            jointIndex=j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=default_target,
                            positionGain=joint_config.stiffness or PYBULLET_DEFAULT_POSITION_GAIN,
                            velocityGain=joint_config.damping or PYBULLET_DEFAULT_VELOCITY_GAIN,
                            maxVelocity=joint_config.velocity_limit or PYBULLET_DEFAULT_JOINT_MAX_VEL,
                            force=joint_config.torque_limit or PYBULLET_DEFAULT_JOINT_MAX_TORQUE,
                        )

                if hasattr(object, "default_joint_positions"):
                    for j in range(num_joints):
                        default_pos = object.default_joint_positions[cur_joint_names[j]]
                        p.resetJointState(bodyUniqueId=curr_id, jointIndex=j, targetValue=default_pos)

                ### TODO:
                # Add dof properties for articulated objects
                # if agent.dof.drive_mode == "position":
                #     p.setJointMotorControlMultiDofArray(
                #         agent.instance,
                #         jointIndices,
                #         p.POSITION_CONTROL,
                #         targetPositions = agent.dof.target
                #     )
                # elif agent.dof.drive_mode == "velocity":
                #     p.setJointMotorControlMultiDofArray(
                #         agent.instance,
                #         jointIndices,
                #         p.VELOCITY_CONTROL,
                #         targetVelocities = agent.dof.target
                #     )
                # elif agent.dof.drive_mode == "torque":
                #     p.setJointMotorControlMultiDofArray(
                #         agent.instance,
                #         jointIndices,
                #         p.TORQUE_CONTROL,
                #         forces = agent.dof.target
                #     )
                # p.resetJointStatesMultiDof(
                #     agent.instance,
                #     jointIndices,
                #     targetValues=[[x] for x in agent.dof.init],
                #     # targetVelocities=[0 for i in range(num_joints)]
                # )
                # p.resetBasePositionAndOrientation(agent.instance, pos, wxyz_to_xyzw(rot))

            elif isinstance(object, PrimitiveCubeCfg):
                box_dimensions = object.half_size
                pos = np.asarray(object.default_position)
                rot = convert_quat(np.asarray(object.default_orientation), to="xyzw")
                box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_dimensions)
                curr_id = p.createMultiBody(
                    baseMass=object.mass,
                    baseCollisionShapeIndex=box_id,
                    basePosition=pos,
                    baseOrientation=rot,
                )
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                radius = object.radius
                pos = np.asarray(object.default_position)
                rot = convert_quat(np.asarray(object.default_orientation), to="xyzw")
                sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                curr_id = p.createMultiBody(
                    baseMass=object.mass,
                    baseCollisionShapeIndex=sphere_id,
                    basePosition=pos,
                    baseOrientation=rot,
                )
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                useFixedBase = object.fix_base_link
                pos = np.asarray(object.default_position)
                rot = convert_quat(np.asarray(object.default_orientation), to="xyzw")
                curr_id = p.loadURDF(
                    object.urdf_path,
                    pos,
                    rot,
                    useFixedBase=useFixedBase,
                    globalScaling=object.scale[0],
                )
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            else:
                raise ValueError(f"Object type {object} not supported")

            # p.resetBaseVelocity(curr_id, agent.vel, agent.ang_vel)
            ### TODO:
            # Add rigid body properties for objects
            # p.changeDynamics(
            #     agent.instance,
            #     -1,
            #     linearDamping=agent.rigid_shape_property.linear_damping,
            #     angularDamping=agent.rigid_shape_property.angular_damping,
            # )

        ### TODO:
        # Viewer configuration
        if not self.headless:
            camera_pos = np.array([1.5, -1.5, 1.5])
            camera_target = np.array([0.0, 0.0, 0.0])
            # if self.viewer_params.viewer_rot != None :
            #     camera_z = np.array([0.0, 0.0, 1.0])
            #     camera_rot = np.array(self.viewer_params.viewer_rot)
            #     camera_rot = camera_rot / np.linalg.norm(camera_rot)
            #     camera_target = camera_pos + quat_apply(camera_rot, camera_z)
            # else :
            #     camera_target = np.array(self.viewer_params.target_pos)
            direction_vector = camera_target - camera_pos
            yaw = -math.atan2(direction_vector[0], direction_vector[1])
            # Compute roll (Rotation around z axis)
            pitch = math.atan2(direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
            roll = 0
            p.resetDebugVisualizerCamera(
                cameraDistance=np.sqrt(np.sum(direction_vector**2)),
                cameraYaw=yaw * 180 / np.pi,
                cameraPitch=pitch * 180 / np.pi,
                cameraTargetPosition=camera_target,
            )

        # Prepare list of debug points and lines
        self.debug_points = []
        self.debug_lines = []

    def _apply_action(self, action, object):
        p.setJointMotorControlArray(
            object, range(action.shape[0]), controlMode=p.POSITION_CONTROL, targetPositions=action
        )

    def set_dof_targets(self, actions: list[Action] | TensorState):
        """Set the target joint positions for the object.

        Args:
            obj_name: The name of the object
            target: The target joint positions
        """
        # For multi-env version, rewrite this function
        actions = adapt_actions_to_dict(self, actions)
        self._actions_cache = actions

        for obj_name, action in actions.items():
            action_arr = np.array([action["dof_pos_target"][name] for name in self.object_joint_order[obj_name]])
            self._apply_action(action_arr, self.object_ids[obj_name])

    def _simulate(self):
        """Step the simulation."""
        # # Rewrite this function for multi-env version
        # action = actions[0]

        # action_arr = np.array([action['dof_pos_target'][name] for name in self.object_joint_order[self.robot.name]])
        # self._apply_action(action_arr, self.object_ids[self.robot.name])

        for i in range(self.scenario.decimation):
            p.stepSimulation()

    def refresh_render(self):
        """Refresh the render."""
        # XXX: This implementation is not correct, but just work for compatibility
        # calvin also use this to refresh the render, seems no better way?
        p.stepSimulation()

    def launch(self) -> None:
        """Launch the simulation."""
        super().launch()
        self._build_pybullet()
        if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
            self._build_gs_background()
        self.already_disconnect = False

    def close(self):
        """Close the simulation."""
        if not self.already_disconnect:
            p.disconnect(self.client)
            self.already_disconnect = True

    ############################################################
    ## Set states
    ############################################################
    def _set_states(self, init_states, env_ids=None):
        """Set the initial states of the environment.

        Args:
            init_states: The initial states
            env_ids: The environment ids
        """
        # For multi-env version, rewrite this function

        states_flat = [state["objects"] | state["robots"] for state in init_states]
        for name, val in states_flat[0].items():
            assert name in self.object_ids
            # Reset joint state
            obj_id = self.object_ids[name]

            if isinstance(self.object_dict[name], ArticulationObjCfg):
                joint_names = self.object_joint_order[name]
                for i, joint_name in enumerate(joint_names):
                    p.resetJointState(obj_id, i, val["dof_pos"][joint_name])

            # Reset base position and orientation
            p.resetBasePositionAndOrientation(obj_id, val["pos"], convert_quat(val["rot"], to="xyzw"))

    ############################################################
    ## Get states
    ############################################################
    def _get_states(self, env_ids=None) -> list[DictEnvState]:
        """Get the states of the environment.

        Returns:
            The states of the environment
        """
        # For multi-env version, rewrite this function

        object_states = {}
        for obj in self.objects:
            obj_id = self.object_ids[obj.name]
            pos, rot = p.getBasePositionAndOrientation(obj_id)
            rot = convert_quat(np.array(rot), to="wxyz")
            pos = torch.tensor(pos, dtype=torch.float32)
            rot = torch.tensor(rot, dtype=torch.float32)
            lin_vel = torch.zeros_like(pos)  # TODO
            ang_vel = torch.zeros_like(pos)  # TODO
            root_state = torch.cat([pos, rot, lin_vel, ang_vel]).unsqueeze(0)
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=root_state,
                    body_names=None,
                    body_state=None,  # TODO
                    joint_pos=torch.tensor([p.getJointState(obj_id, i)[0] for i in joint_reindex]).unsqueeze(0),
                    joint_vel=torch.tensor([p.getJointState(obj_id, i)[1] for i in joint_reindex]).unsqueeze(0),
                )
            else:
                state = ObjectState(root_state=root_state)
            object_states[obj.name] = state

        robot_states = {}
        for robot in self.robots:
            joint_reindex = self.get_joint_reindex(robot.name)
            obj_id = self.object_ids[robot.name]
            pos, rot = p.getBasePositionAndOrientation(obj_id)
            rot = convert_quat(np.array(rot), to="wxyz")
            pos = torch.tensor(pos, dtype=torch.float32)
            rot = torch.tensor(rot, dtype=torch.float32)
            lin_vel = torch.zeros_like(pos)  # TODO
            ang_vel = torch.zeros_like(pos)  # TODO
            root_state = torch.cat([pos, rot, lin_vel, ang_vel]).unsqueeze(0)
            state = RobotState(
                root_state=root_state,
                body_names=None,
                body_state=None,  # TODO
                joint_pos=torch.tensor([p.getJointState(obj_id, i)[0] for i in joint_reindex]).unsqueeze(0),
                joint_vel=torch.tensor([p.getJointState(obj_id, i)[1] for i in joint_reindex]).unsqueeze(0),
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=None,  # TODO
            )
            robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            width, height, view_matrix, projection_matrix, near_plane, far_plane = self.camera_ids[camera.name]

            # Get PyBullet simulation rendering
            img_arr = p.getCameraImage(width, height, view_matrix, projection_matrix, lightAmbientCoeff=0.5)
            rgb_img = np.reshape(img_arr[2], (height, width, 4))
            depth_buffer = np.reshape(img_arr[3], (height, width))
            depth_img = self._convert_depth_buffer(depth_buffer, near_plane, far_plane)
            segmentation_mask = np.reshape(img_arr[4], (height, width))

            if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
                from metasim.utils.gs_util import alpha_blend_rgba

                # Extract camera parameters from PyBullet
                Ks, c2w = self._get_camera_params(view_matrix, projection_matrix, width, height)

                # Render GS background
                gs_cam = SplatCamera.init_from_pose_list(
                    pose_list=c2w,
                    camera_intrinsic=Ks,
                    image_height=height,
                    image_width=width,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                gs_result = self.gs_background.render(gs_cam)
                gs_result.to_numpy()

                # Create foreground mask: exclude background (-1) and ground plane
                foreground_mask = (segmentation_mask > -1) & (segmentation_mask != self.plane_id).astype(np.uint8)

                # Blend RGB: foreground objects over GS background
                sim_color = rgb_img[:, :, :3]  # rgba to rgb
                foreground = np.concatenate([sim_color, foreground_mask[..., None] * 255], axis=-1)
                background = gs_result.rgb.squeeze(0)
                blended_rgb = alpha_blend_rgba(foreground, background)
                rgb = torch.from_numpy(np.array(blended_rgb.copy()))

                # Compose depth: use simulation depth for foreground, GS depth for background
                bg_depth = gs_result.depth.squeeze(0)
                if bg_depth.ndim == 3 and bg_depth.shape[-1] == 1:
                    bg_depth = bg_depth[..., 0]
                depth_comp = np.where(foreground_mask, depth_img, bg_depth)
                depth = torch.from_numpy(depth_comp.astype(np.float32, copy=False))
            else:
                # Original PyBullet rendering without GS background
                rgb = torch.from_numpy(rgb_img[:, :, :3])
                depth = torch.from_numpy(depth_img)

            state = CameraState(
                rgb=rgb.unsqueeze(0),
                depth=depth.unsqueeze(0),
            )
            camera_states[camera.name] = state

        extras = self.get_extra()  # extra observations
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    ############################################################
    ## Utils
    ############################################################
    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = deepcopy(self.object_joint_order[obj_name])
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def robot(self):
        return self.robots[0]


PybulletHandler = SinglePybulletHandler  # TODO: support parallel
