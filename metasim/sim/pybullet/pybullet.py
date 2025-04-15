"""Implemention of Pybullet Handler.

This file contains the implementation of PybulletHandler, which is a subclass of BaseSimHandler.
PybulletHandler is used to handle the simulation environment using Pybullet.
Currently using Pybullet 3.2.6
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pybullet_data
import torch

from metasim.cfg.objects import ArticulationObjCfg, BaseObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.math import convert_quat
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


class SinglePybulletHandler(BaseSimHandler):
    """Pybullet Handler class."""

    def __init__(self, scenario: ScenarioCfg):
        """Initialize the Pybullet Handler.

        Args:
            scenario: The scenario configuration
            num_envs: Number of environments
            headless: Whether to run the simulation in headless mode
        """
        super().__init__(scenario)
        self._actions_cache: list[Action] = []

    def _build_pybullet(self):
        self.client = p.connect(p.GUI)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 60.0, numSolverIterations=300)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

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

            self.camera_ids[camera.name] = (width, height, view_matrix, projection_matrix)

        for object in [*self.objects, self.robot]:
            if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)):
                pos = np.array([0, 0, 0])
                rot = np.array([1, 0, 0, 0])
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
                    convert_quat(rot, to="xyzw"),
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

                if isinstance(object, BaseRobotCfg):
                    p.setJointMotorControlMultiDofArray(
                        curr_id,
                        jointIndices,
                        p.POSITION_CONTROL,
                        # targetPositions = agent.dof.target
                    )

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
                box_dimensions = [x * s for x, s in zip(object.half_size, object.scale)]
                pos = np.array([0, 0, 0])
                rot = np.array([1, 0, 0, 0])
                box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_dimensions)
                curr_id = p.createMultiBody(
                    baseMass=object.mass,
                    baseCollisionShapeIndex=box_id,
                    basePosition=pos,
                    baseOrientation=convert_quat(rot, to="xyzw"),
                )
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                radius = object.radius * object.scale[0]
                pos = np.array([0, 0, 0])
                rot = np.array([1, 0, 0, 0])
                sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                curr_id = p.createMultiBody(
                    baseMass=object.mass,
                    baseCollisionShapeIndex=sphere_id,
                    basePosition=pos,
                    baseOrientation=convert_quat(rot, to="xyzw"),
                )
                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                useFixedBase = object.fix_base_link
                pos = np.array([0, 0, 0])
                rot = np.array([1, 0, 0, 0])
                curr_id = p.loadURDF(
                    object.urdf_path,
                    pos,
                    convert_quat(rot, to="xyzw"),
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

    def set_dof_targets(self, obj_name, actions: list[Action]):
        """Set the target joint positions for the object.

        Args:
            obj_name: The name of the object
            target: The target joint positions
        """
        # For multi-env version, rewrite this function
        self._actions_cache = actions
        action = actions[0]
        action_arr = np.array([action["dof_pos_target"][name] for name in self.object_joint_order[obj_name]])
        self._apply_action(action_arr, self.object_ids[obj_name])

    def simulate(self):
        """Step the simulation."""
        # # Rewrite this function for multi-env version
        # action = actions[0]

        # action_arr = np.array([action['dof_pos_target'][name] for name in self.object_joint_order[self.robot.name]])
        # self._apply_action(action_arr, self.object_ids[self.robot.name])

        for i in range(self.scenario.decimation):
            p.stepSimulation()

        # return None, None, np.zeros(1), np.zeros(1), np.zeros(1)

    def refresh_render(self):
        """Refresh the render."""
        # XXX: This implementation is not correct, but just work for compatibility
        # calvin also use this to refresh the render, seems no better way?
        p.stepSimulation()

    def launch(self) -> None:
        """Launch the simulation."""
        self._build_pybullet()
        self.already_disconnect = False

    def close(self):
        """Close the simulation."""
        if not self.already_disconnect:
            p.disconnect(self.client)
            self.already_disconnect = True

    ############################################################
    ## Set states
    ############################################################
    def set_states(self, init_states, env_ids=None):
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
    def get_states(self, env_ids=None) -> list[EnvState]:
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
        for robot in [self.robot]:
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
            width, height, view_matrix, projection_matrix = self.camera_ids[camera.name]
            img_arr = p.getCameraImage(
                width,
                height,
                view_matrix,
                projection_matrix,
                lightAmbientCoeff=0.5,
            )
            rgb_img = np.reshape(img_arr[2], (height, width, 4))
            depth_img = np.reshape(img_arr[3], (height, width))
            segmentation_mask = np.reshape(img_arr[4], (height, width))
            state = CameraState(
                rgb=torch.from_numpy(rgb_img[:, :, :3]).unsqueeze(0),
                depth=torch.from_numpy(depth_img).unsqueeze(0),
            )
            camera_states[camera.name] = state

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states)

    ############################################################
    ## Utils
    ############################################################
    def get_object_joint_names(self, obj: BaseObjCfg) -> list[str]:
        if isinstance(obj, ArticulationObjCfg):
            return self.object_joint_order[obj.name]
        else:
            return []

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


PybulletHandler = SinglePybulletHandler  # TODO: support parallel

PybulletEnv: type[EnvWrapper[PybulletHandler]] = GymEnvWrapper(PybulletHandler)  # type: ignore
