"""Implemention of Sapien Handler.

This file contains the implementation of Sapien2Handler, which is a subclass of BaseSimHandler.
Sapien2Handler is used to handle the simulation environment using Sapien.
Currently using Sapien 2.2
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import sapien
import sapien.core as sapien_core
import sapien.physx as physx
import torch
from loguru import logger as log
from packaging.version import parse as parse_version
from sapien.utils import Viewer
from scipy.spatial.transform import Rotation as R

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import (
    ArticulationObjCfg,
    NonConvexRigidObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import Action, DictEnvState
from metasim.utils.math import quat_from_euler_np
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, adapt_actions_to_dict

from .sapien2 import _load_init_pose

# Optional: RoboSplatter imports for GS background rendering
try:
    from robo_splatter.models.camera import Camera as SplatCamera

    ROBO_SPLATTER_AVAILABLE = True
except ImportError:
    ROBO_SPLATTER_AVAILABLE = False
    log.warning("RoboSplatter not available. GS background rendering will be disabled.")


__all__ = [
    "Sapien3Handler",
    "load_actor_from_urdf",
]


def load_actor_from_urdf(
    scene: sapien.Scene,
    file_path: str,
    pose: sapien.Pose | None = None,
    use_static: bool = False,
    update_mass: bool = False,
    scale: float | np.ndarray = 1.0,
) -> sapien.pysapien.Entity:
    def _get_local_pose(origin_tag: ET.Element | None) -> sapien.Pose:
        local_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if origin_tag is not None:
            xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
            qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
            local_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])

        return local_pose

    tree = ET.parse(file_path)
    root = tree.getroot()
    node_name = root.get("name")
    file_dir = os.path.dirname(file_path)

    visual_mesh = root.find(".//visual/geometry/mesh")
    visual_file = visual_mesh.get("filename").replace("package://", "")
    visual_scale = visual_mesh.get("scale", "1.0 1.0 1.0")
    visual_scale = np.array([float(x) for x in visual_scale.split()]) * np.array(scale)

    collision_mesh = root.find(".//collision/geometry/mesh")
    collision_file = collision_mesh.get("filename").replace("package://", "")
    collision_scale = collision_mesh.get("scale", "1.0 1.0 1.0")
    collision_scale = np.array([float(x) for x in collision_scale.split()]) * np.array(scale)

    visual_pose = _get_local_pose(root.find(".//visual/origin"))
    collision_pose = _get_local_pose(root.find(".//collision/origin"))

    visual_file = os.path.join(file_dir, visual_file).replace("package://", "")
    collision_file = os.path.join(file_dir, collision_file).replace("package://", "")
    mu1 = root.find(".//collision/gazebo/mu1")
    mu2 = root.find(".//collision/gazebo/mu2")
    static_fric = mu1.text if mu1 is not None else 0.5
    dynamic_fric = mu2.text if mu2 is not None else 0.5

    material = physx.PhysxMaterial(
        static_friction=np.clip(float(static_fric), 0.1, 0.7),
        dynamic_friction=np.clip(float(dynamic_fric), 0.1, 0.6),
        restitution=0.05,
    )
    builder = scene.create_actor_builder()

    body_type = "static" if use_static else "dynamic"
    builder.set_physx_body_type(body_type)
    builder.add_multiple_convex_collisions_from_file(
        collision_file,
        material=material,
        scale=collision_scale,
        # decomposition="coacd",
        # decomposition_params=dict(
        #     threshold=0.05, max_convex_hull=64, verbose=False
        # ),
        pose=collision_pose,
    )

    builder.add_visual_from_file(
        visual_file,
        scale=visual_scale,
        pose=visual_pose,
    )
    if pose is None:
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
    builder.set_initial_pose(pose)
    actor = builder.build(name=node_name)

    if update_mass and hasattr(actor.components[1], "mass"):
        node_mass = float(root.find(".//inertial/mass").get("value"))
        actor.components[1].set_mass(node_mass)

    return actor


class Sapien3Handler(BaseSimHandler):
    """Sapien3 Handler class."""

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        assert parse_version(sapien.__version__) >= parse_version("3.0.0a0"), "Sapien3 is required"
        assert parse_version(sapien.__version__) < parse_version("4.0.0"), "Sapien3 is required"
        log.warning("Sapien3 is still under development, some metasim apis yet don't have sapien3 support")
        super().__init__(scenario, optional_queries)
        self.headless = scenario.headless
        self._actions_cache: list[Action] = []

    def _build_sapien(self):
        self.engine = sapien_core.Engine()  # Create a physical simulation engine
        self.renderer = sapien_core.SapienRenderer()  # Create a renderer

        scene_config = sapien_core.SceneConfig()
        # scene_config.default_dynamic_friction = self.physical_params.dynamic_friction
        # scene_config.default_static_friction = self.physical_params.static_friction
        # scene_config.contact_offset = self.physical_params.contact_offset
        # scene_config.default_restitution = self.physical_params.restitution
        # scene_config.enable_pcm = True
        # scene_config.solver_iterations = self.sim_params.num_position_iterations
        # scene_config.solver_velocity_iterations = self.sim_params.num_velocity_iterations
        scene_config.gravity = np.array(self.scenario.gravity)
        # scene_config.bounce_threshold = self.sim_params.bounce_threshold

        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100)
        ground_material = self.renderer.create_material()
        ground_material.base_color = np.array([202, 164, 114, 256]) / 256
        ground_material.specular = 0.5
        self.scene.add_ground(altitude=0, render_material=ground_material)

        self.loader = self.scene.create_urdf_loader()

        # Add agents
        self.object_ids: dict[str, sapien_core.Entity] = {}
        self.link_ids: dict[str, list[sapien.physx.PhysxArticulationLinkComponent]] = {}
        self._previous_dof_pos_target: dict[str, np.ndarray] = {}
        self._previous_dof_vel_target: dict[str, np.ndarray] = {}
        self._previous_dof_torque_target: dict[str, np.ndarray] = {}
        self.object_joint_order = {}
        self.camera_ids = {}

        for camera in self.cameras:
            # Create a camera entity in the scene
            camera_id = self.scene.add_camera(
                name=camera.name,
                width=camera.width,
                height=camera.height,
                fovy=np.deg2rad(camera.vertical_fov),
                near=camera.clipping_range[0],
                far=camera.clipping_range[1],
            )
            pos = np.array(camera.pos)
            look_at = np.array(camera.look_at)
            direction_vector = look_at - pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
            roll = 0
            camera_id.set_pose(sapien_core.Pose(p=pos, q=quat_from_euler_np(roll, -pitch, yaw)))
            self.camera_ids[camera.name] = camera_id

            # near, far = 0.1, 100
            # width, height = 640, 480
            # camera_id = self.scene.add_camera(
            #     name="camera",
            #     width=width,
            #     height=height,
            #     fovy=np.deg2rad(35),
            #     near=near,
            #     far=far,
            # )
            # camera_id.set_pose(sapien.Pose(p=[2, 0, 0], q=[0, 0, -1, 0]))
            # self.camera_ids[camera.name] = camera_id

        for object in [*self.objects, self.robot]:
            if isinstance(object, (ArticulationObjCfg, RobotCfg)):
                self.loader.fix_root_link = object.fix_base_link
                self.loader.scale = object.scale[0]
                file_path = object.urdf_path
                curr_id = self.loader.load(file_path)
                curr_id.set_root_pose(_load_init_pose(object))

                self.object_ids[object.name] = curr_id

                active_joints = curr_id.get_active_joints()
                # num_joints = len(active_joints)
                cur_joint_names = []
                for id, joint in enumerate(active_joints):
                    joint_name = joint.get_name()
                    cur_joint_names.append(joint_name)
                self.object_joint_order[object.name] = cur_joint_names

                ### TODO
                # Change dof properties
                ###

                if isinstance(object, RobotCfg):
                    active_joints = curr_id.get_active_joints()
                    for id, joint in enumerate(active_joints):
                        stiffness = object.actuators[joint.get_name()].stiffness
                        damping = object.actuators[joint.get_name()].damping
                        if stiffness is not None and damping is not None:
                            joint.set_drive_property(stiffness, damping)
                else:
                    active_joints = curr_id.get_active_joints()
                    for id, joint in enumerate(active_joints):
                        joint.set_drive_property(0, 0)

                if hasattr(object, "default_joint_positions") and object.default_joint_positions:
                    qpos_list = []
                    for i, joint_name in enumerate(cur_joint_names):
                        qpos_list.append(object.default_joint_positions[joint_name])
                    curr_id.set_qpos(qpos_list)

                # if agent.dof.init:
                #     robot.set_qpos(agent.dof.init)

                # if agent.dof.target:
                #     robot.set_drive_target(agent.dof.target)

            elif isinstance(object, PrimitiveCubeCfg):
                actor_builder = self.scene.create_actor_builder()
                # material = get_material(self.scene, agent.rigid_shape_property)
                actor_builder.add_box_collision(
                    half_size=object.half_size,
                    density=object.density,
                    # material=material,
                )
                actor_builder.add_box_visual(
                    half_size=object.half_size,
                    # color=object.color if object.color else [1.0, 1.0, 0.0],
                    material=sapien_core.render.RenderMaterial(
                        base_color=list(object.color[:3]) + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                box = actor_builder.build(name="box")  # Add a box
                box.set_pose(_load_init_pose(object))
                # box.set_damping(agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping)
                # if agent.vel:
                #     box.set_velocity(agent.vel)
                # if agent.ang_vel:
                #     box.set_angular_velocity(agent.ang_vel)
                # box.set_damping(agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping)
                # if agent.fix_base_link:
                #     box.lock_motion()
                # agent.instance = box
                self.object_ids[object.name] = box
                self.object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                actor_builder = self.scene.create_actor_builder()
                # material = get_material(self.scene, agent.rigid_shape_property)
                actor_builder.add_sphere_collision(radius=object.radius, density=object.density)
                actor_builder.add_sphere_visual(
                    radius=object.radius,
                    material=sapien_core.render.RenderMaterial(
                        base_color=list(object.color[:3]) + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                sphere = actor_builder.build(name="sphere")  # Add a sphere
                sphere.set_pose(_load_init_pose(object))
                # sphere.set_damping(
                #     agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping
                # )
                # if agent.vel:
                #     sphere.set_velocity(agent.vel)
                # if agent.ang_vel:
                #     sphere.set_angular_velocity(agent.ang_vel)
                # if agent.fix_base_link:
                #     sphere.lock_motion()
                # agent.instance = sphere
                self.object_ids[object.name] = sphere
                self.object_joint_order[object.name] = []

            elif isinstance(object, NonConvexRigidObjCfg):
                builder = self.scene.create_actor_builder()
                scene_pose = sapien_core.Pose(p=np.array(object.mesh_pose[:3]), q=np.array(object.mesh_pose[3:]))
                builder.add_nonconvex_collision_from_file(object.usd_path, scene_pose)
                builder.add_visual_from_file(object.usd_path, scene_pose)
                curr_id = builder.build_static(name=object.name)
                curr_id.set_pose(_load_init_pose(object))

                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                self.loader.fix_root_link = object.fix_base_link
                self.loader.scale = object.scale[0]
                file_path = object.urdf_path
                # curr_id: sapien_core.Entity
                # try:
                #     curr_id = self.loader.load(file_path)
                # except Exception as e:
                #     log.warning(f"Error loading {file_path}: {e}")
                #     curr_id_list = self.loader.load_multiple(file_path)
                #     # TODO:
                #     # Don't understand why some urdf are treated as multiple entities
                #     # Needs to figure out a better way to load!
                #     for id in curr_id_list:
                #         if len(id):
                #             curr_id = id
                #             break
                # # builder = self.loader.load_file_as_articulation_builder(file_path)
                # if isinstance(curr_id, list):
                #     ## HACK
                #     curr_id = curr_id[0]
                # curr_id.set_pose(sapien_core.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
                curr_id = load_actor_from_urdf(self.scene, file_path, scale=object.scale)
                curr_id.set_pose(_load_init_pose(object))

                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            if isinstance(object, (ArticulationObjCfg, RobotCfg)):
                self.link_ids[object.name] = self.object_ids[object.name].get_links()
                self._previous_dof_pos_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
                self._previous_dof_vel_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
                self._previous_dof_torque_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
            else:
                self.link_ids[object.name] = []

            # elif agent.type == "capsule":
            #     actor_builder = self.scene.create_actor_builder()
            #     material = get_material(self.scene, agent.rigid_shape_property)
            #     actor_builder.add_capsule_collision(
            #         radius=agent.radius, half_length=agent.length, density=agent.density, material=material
            #     )
            #     actor_builder.add_capsule_visual(
            #         radius=agent.radius, half_length=agent.length, color=agent.color if agent.color else [1.0, 1.0, 1.0]
            #     )
            #     capsule = actor_builder.build(name="capsule")  # Add a capsule
            #     capsule.set_pose(sapien.Pose(p=[*agent.pos], q=np.asarray(agent.rot)))
            #     capsule.set_damping(
            #         agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping
            #     )
            #     if agent.vel:
            #         capsule.set_velocity(agent.vel)
            #     if agent.ang_vel:
            #         capsule.set_angular_velocity(agent.ang_vel)
            #     if agent.fix_base_link:
            #         capsule.lock_motion()
            #     agent.instance = capsule

        # Add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        # self.scene.add_directional_light(
        #     self.sim_params.directional_light_pos, self.sim_params.directional_light_target
        # )

        # Create viewer and adjust camera position
        # if not self.viewer_params.headless:
        if not self.headless:
            self.viewer = Viewer(self.renderer)  # Create a viewer (window)
            self.viewer.set_scene(self.scene)  # Bind the viewer and the scene

        if not self.headless:
            camera_pos = np.array([1.5, -1.5, 1.5])
            camera_target = np.array([0.0, 0.0, 0.0])
            # if self.viewer_params.viewer_rot != None:
            #     camera_z = np.array([0.0, 0.0, 1.0])
            #     camera_rot = np.array(self.viewer_params.viewer_rot)
            #     camera_target = camera_pos + quat_apply(camera_rot, camera_z)
            # else:
            #     camera_target = np.array(self.viewer_params.target_pos)
            direction_vector = camera_target - camera_pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(
                direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
            )  # 计算 roll 角（绕 X 轴的旋转角度）
            roll = 0
            # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
            # The principle axis of the camera is the x-axis
            self.viewer.set_camera_xyz(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2])
            # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
            self.viewer.set_camera_rpy(r=roll, p=pitch, y=-yaw)
            # TODO:
            # UNABLE TO REMOVE THE AXIS AND CAMERA LINES FOR EARLY VERSIONS OF SAPIEN3
            # self.viewer.toggle_axes(show=False)
            # self.viewer.toggle_camera_lines(show=False)

        # self.viewer.set_fovy(self.viewer_params.horizontal_fov)
        # self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=self.viewer_params.fovy / 2) # the /2 is to align with isaac-gym

        # List for debug points
        self.debug_points = []
        self.debug_lines = []

        self.scene.update_render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def _apply_action(self, instance: sapien_core.physx.PhysxArticulation, pos_action=None, vel_action=None):
        qf = instance.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        instance.set_qf(qf)
        if pos_action is not None:
            for joint in instance.get_active_joints():
                joint.set_drive_target(pos_action[joint.get_name()])
        if vel_action is not None:
            for joint in instance.get_active_joints():
                joint.set_drive_velocity_target(vel_action[joint.get_name()])
        # instance.set_drive_target(action)

    def _set_dof_targets(self, targets: list[Action] | TensorState):
        targets = adapt_actions_to_dict(self, targets)

        for obj_name, action in targets.items():
            instance = self.object_ids[obj_name]
            if isinstance(instance, sapien_core.physx.PhysxArticulation):
                pos_target = action.get("dof_pos_target", None)
                vel_target = action.get("dof_vel_target", None)
                jns = self.get_joint_names(obj_name, sort=True)
                if pos_target is not None:
                    self._previous_dof_pos_target[obj_name] = np.array([pos_target[name] for name in jns])
                if vel_target is not None:
                    self._previous_dof_vel_target[obj_name] = np.array([vel_target[name] for name in jns])
                self._apply_action(instance, pos_target, vel_target)

    def _simulate(self):
        for i in range(self.scenario.decimation):
            self.scene.step()
            self.scene.update_render()
            if not self.headless:
                self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def launch(self) -> None:
        super().launch()
        self._build_sapien()
        if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
            self._build_gs_background()

    def close(self):
        if not self.headless:
            self.viewer.close()
        self.scene = None

    def _get_link_states(self, obj_name: str) -> tuple[list, torch.Tensor]:
        link_name_list = []
        link_state_list = []

        if len(self.link_ids[obj_name]) == 0:
            return [], torch.zeros((0, 13), dtype=torch.float32)
        for link in self.link_ids[obj_name]:
            pose = link.get_pose()
            pos = torch.tensor(pose.p)
            rot = torch.tensor(pose.q)
            vel = torch.tensor(link.linear_velocity)
            ang_vel = torch.tensor(link.angular_velocity)
            link_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            link_name_list.append(link.get_name())
            link_state_list.append(link_state)
        link_state_tensor = torch.cat(link_state_list, dim=0)

        # sort the links by name
        sorted_indices = sorted(range(len(link_name_list)), key=lambda i: link_name_list[i])
        link_name_list = [link_name_list[i] for i in sorted_indices]
        link_state_tensor = link_state_tensor[sorted_indices]

        return link_name_list, link_state_tensor

    def _get_states(self, env_ids=None) -> list[DictEnvState]:
        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_ids[obj.name]
            pose = obj_inst.get_pose()
            link_names, link_state = self._get_link_states(obj.name)
            if isinstance(obj, ArticulationObjCfg):
                assert isinstance(obj_inst, sapien_core.physx.PhysxArticulation)
                pos = torch.tensor(pose.p)
                rot = torch.tensor(pose.q)
                vel = torch.tensor(obj_inst.get_root_linear_velocity())
                ang_vel = torch.tensor(obj_inst.get_root_angular_velocity())
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                joint_reindex = self.get_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=root_state,
                    body_names=link_names,
                    body_state=link_state.unsqueeze(0),
                    joint_pos=torch.tensor(obj_inst.get_qpos()[joint_reindex]).unsqueeze(0),
                    joint_vel=torch.tensor(obj_inst.get_qvel()[joint_reindex]).unsqueeze(0),
                )
            else:
                assert isinstance(obj_inst, sapien_core.Entity)
                pos = torch.tensor(pose.p)
                rot = torch.tensor(pose.q)
                vel = torch.tensor(obj_inst.get_components()[1].get_linear_velocity())
                ang_vel = torch.tensor(obj_inst.get_components()[1].get_angular_velocity())
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                state = ObjectState(root_state=root_state)
            object_states[obj.name] = state

        robot_states = {}
        for robot in [self.robot]:
            robot_inst = self.object_ids[robot.name]
            assert isinstance(robot_inst, sapien_core.physx.PhysxArticulation)
            pose = robot_inst.get_pose()
            pos = torch.tensor(pose.p)
            rot = torch.tensor(pose.q)
            vel = torch.tensor(robot_inst.root_linear_velocity)
            ang_vel = torch.tensor(robot_inst.root_angular_velocity)
            root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            joint_reindex = self.get_joint_reindex(robot.name)
            link_names, link_state = self._get_link_states(robot.name)
            pos_target = torch.tensor(self._previous_dof_pos_target[robot.name]).unsqueeze(0)
            vel_target = torch.tensor(self._previous_dof_vel_target[robot.name]).unsqueeze(0)
            effort_target = torch.tensor(self._previous_dof_torque_target[robot.name]).unsqueeze(0)
            state = RobotState(
                root_state=root_state,
                body_names=link_names,
                body_state=link_state.unsqueeze(0),
                joint_pos=torch.tensor(robot_inst.get_qpos()[joint_reindex]).unsqueeze(0),
                joint_vel=torch.tensor(robot_inst.get_qvel()[joint_reindex]).unsqueeze(0),
                joint_pos_target=pos_target,
                joint_vel_target=vel_target,
                joint_effort_target=effort_target,
            )
            robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            cam_inst = self.camera_ids[camera.name]

            if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
                from metasim.utils.gs_util import alpha_blend_rgba

                # Build RoboSplatter camera from SAPIEN pose and scenario intrinsics, then render GS
                gs_cam = SplatCamera.init_from_pose_list(
                    pose_list=cam_inst.get_model_matrix(),
                    camera_intrinsic=cam_inst.get_intrinsic_matrix(),
                    image_height=cam_inst.height,
                    image_width=cam_inst.width,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                gs_result = self.gs_background.render(gs_cam)
                gs_result.to_numpy()

                seg_labels = cam_inst.get_picture("Segmentation")
                label0 = seg_labels[..., 0]
                mask = np.where((label0 > 1), 255, 0).astype(
                    np.uint8
                )  # exclude background and ground plane (typically ID 0 = ground, ID 1 = first object)

                # depth compose: use sim depth where foreground exists, otherwise GS depth
                sim_depth = -cam_inst.get_picture("Position")[..., 2]
                bg_depth = gs_result.depth.squeeze(0)
                if bg_depth.ndim == 3 and bg_depth.shape[-1] == 1:
                    bg_depth = bg_depth[..., 0]

                # foreground mask: label0 > 1 means non-background objects
                depth_comp = np.where(label0 > 1, sim_depth, bg_depth)
                depth = torch.from_numpy(depth_comp.copy())

                # rgb blend
                sim_color = cam_inst.get_picture("Color")
                sim_color = (np.clip(sim_color[..., :3], 0, 1) * 255).astype(np.uint8)
                foreground = np.concatenate([sim_color, mask[..., None]], axis=-1)

                background = gs_result.rgb.squeeze(0)
                blended_rgb = alpha_blend_rgba(foreground, background)
                rgb = torch.from_numpy(np.array(blended_rgb.copy()))

            else:
                rgb = torch.from_numpy(np.array(cam_inst.get_picture("Color").copy()))
                depth = -cam_inst.get_picture("Position")[..., 2]
                depth = torch.from_numpy(depth.copy())

            state = CameraState(rgb=rgb.unsqueeze(0), depth=depth.unsqueeze(0))
            camera_states[camera.name] = state

        extras = self.get_extra()  # extra observations
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    def refresh_render(self):
        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def _set_states(self, states, env_ids=None):
        states_flat = [state["objects"] | state["robots"] for state in states]
        for name, val in states_flat[0].items():
            if name not in self.object_ids:
                continue
            # assert name in self.object_ids
            # Reset joint state
            obj_id = self.object_ids[name]

            if isinstance(self.object_dict[name], ArticulationObjCfg):
                joint_names = self.object_joint_order[name]
                qpos_list = []
                for i, joint_name in enumerate(joint_names):
                    qpos_list.append(val["dof_pos"][joint_name])
                obj_id.set_qpos(np.array(qpos_list))

            # Reset base position and orientation
            obj_id.set_pose(sapien_core.Pose(p=val["pos"], q=val["rot"]))

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = deepcopy(self.object_joint_order[obj_name])
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _get_body_names(self, obj_name, sort=True):
        body_names = deepcopy([link.name for link in self.link_ids[obj_name]])
        if sort:
            return sorted(body_names)
        else:
            return deepcopy(body_names)

    @property
    def robot(self):
        return self.robots[0]
