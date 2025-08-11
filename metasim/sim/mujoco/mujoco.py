from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.robots import BaseRobotCfg

if TYPE_CHECKING:
    from metasim.cfg.scenario import ScenarioCfg

from metasim.constants import TaskType
from metasim.queries.base import BaseQueryType
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.sim.parallel import ParallelSimWrapper
from metasim.types import Action
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState


class MujocoHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self._actions_cache: list[Action] = []

        if scenario.num_envs > 1:
            raise ValueError("MujocoHandler only supports single envs, please run with --num_envs 1.")

        self._mujoco_robot_name = None
        self._robot_num_dof = None
        self._robot_path = self.robot.mjcf_path
        self._gravity_compensation = not self.robot.enabled_gravity

        self.viewer = None
        self.cameras = []
        for camera in scenario.cameras:
            self.cameras.append(camera)
        self._episode_length_buf = 0

        # FIXME: hard code decimation for now
        if self.task is not None and self.task.task_type == TaskType.LOCOMOTION:
            self.decimation = self.scenario.decimation
        else:
            log.warning("Warning: hard coding decimation to 25 for replaying trajectories")
            self.decimation = 25

        self._manual_pd_on = False
        self._p_gains = None
        self._d_gains = None
        self._torque_limits = None
        self._robot_default_dof_pos = None
        self._action_scale = scenario.control.action_scale
        self._action_offset = scenario.control.action_offset
        self._effort_controlled_joints = []
        self._position_controlled_joints = []
        self._current_action = None
        self._current_vel_target = None  # Track velocity targets

    def launch(self) -> None:
        model = self._init_mujoco()
        self.physics = mjcf.Physics.from_mjcf_model(model)
        self.data = self.physics.data

        self.body_names = [self.physics.model.body(i).name for i in range(self.physics.model.nbody)]
        self.robot_body_names = [
            body_name for body_name in self.body_names if body_name.startswith(self._mujoco_robot_name)
        ]

        self._init_torque_control()

        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
            self.viewer.sync()

        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)

    def _init_torque_control(self):
        """Initialize torque control parameters based on robot configuration."""
        joint_names = self.get_joint_names(self.robot.name, sort=True)
        self._robot_num_dof = len(joint_names)

        self._p_gains = np.zeros(self._robot_num_dof)
        self._d_gains = np.zeros(self._robot_num_dof)
        self._torque_limits = np.zeros(self._robot_num_dof)

        self._manual_pd_on = any(mode == "effort" for mode in self.robot.control_type.values())

        default_dof_pos = []

        for i, joint_name in enumerate(joint_names):
            i_actuator_cfg = self.robot.actuators[joint_name]
            i_control_mode = self.robot.control_type.get(joint_name, "position")

            if joint_name in self.robot.default_joint_positions:
                default_pos = self.robot.default_joint_positions[joint_name]
            else:
                joint_id = self.physics.model.joint(f"{self._mujoco_robot_name}{joint_name}").id
                joint_range = self.physics.model.jnt_range[joint_id]
                default_pos = 0.3 * (joint_range[0] + joint_range[1])
            default_dof_pos.append(default_pos)

            if i_control_mode == "effort":
                self._effort_controlled_joints.append(i)
                self._p_gains[i] = i_actuator_cfg.stiffness
                self._d_gains[i] = i_actuator_cfg.damping

                if i_actuator_cfg.torque_limit is not None:
                    torque_limit = i_actuator_cfg.torque_limit
                else:
                    actuator_id = self.physics.model.actuator(f"{self._mujoco_robot_name}{joint_name}").id
                    torque_limit = self.physics.model.actuator_forcerange[actuator_id, 1]

                self._torque_limits[i] = self.scenario.control.torque_limit_scale * torque_limit

            elif i_control_mode == "position":
                self._position_controlled_joints.append(i)
            else:
                log.error(f"Unknown actuator control mode: {i_control_mode}, only support effort and position")
                raise ValueError

        self._robot_default_dof_pos = np.array(default_dof_pos)
        self._current_vel_target = None  # Initialize velocity target tracking

    def _apply_scale_to_mjcf(self, mjcf_model, scale):
        """Apply scale to all geoms, bodies, and sites in the MJCF model."""
        scale_x, scale_y, scale_z = scale

        for geom in mjcf_model.find_all("geom"):
            if hasattr(geom, "size") and geom.size is not None:
                size = list(geom.size)
                if geom.type in ["box", None]:
                    if len(size) >= 3:
                        geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]
                elif geom.type == "sphere":
                    if len(size) >= 1:
                        geom.size = [size[0] * max(scale_x, scale_y, scale_z)]
                elif geom.type == "cylinder":
                    if len(size) >= 2:
                        radius_scale = max(scale_x, scale_y)
                        geom.size = [size[0] * radius_scale, size[1] * scale_z]
                elif geom.type == "capsule":
                    if len(size) >= 2:
                        radius_scale = max(scale_x, scale_y)
                        geom.size = [size[0] * radius_scale, size[1] * scale_z]
                elif geom.type == "ellipsoid":
                    if len(size) >= 3:
                        geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]

            if hasattr(geom, "pos") and geom.pos is not None:
                pos = list(geom.pos)
                if len(pos) >= 3:
                    geom.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

        for body in mjcf_model.find_all("body"):
            if hasattr(body, "pos") and body.pos is not None:
                pos = list(body.pos)
                if len(pos) >= 3:
                    body.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

        for site in mjcf_model.find_all("site"):
            if hasattr(site, "pos") and site.pos is not None:
                pos = list(site.pos)
                if len(pos) >= 3:
                    site.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

            if hasattr(site, "size") and site.size is not None:
                size = list(site.size)
                if len(size) >= 1:
                    site.size = [size[0] * max(scale_x, scale_y, scale_z)]

        for joint in mjcf_model.find_all("joint"):
            if hasattr(joint, "pos") and joint.pos is not None:
                pos = list(joint.pos)
                if len(pos) >= 3:
                    joint.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

    def _set_framebuffer_size(self, mjcf_model, width, height):
        visual_elem = mjcf_model.visual
        global_elem = None
        for child in visual_elem._children:
            if child.tag == "global":
                global_elem = child
                break
        if global_elem is None:
            global_elem = visual_elem.add("global")
        global_elem.offwidth = width
        global_elem.offheight = height

    def _create_primitive_xml(self, obj):
        if isinstance(obj, PrimitiveCubeCfg):
            size_str = f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}"
            type_str = "box"
        elif isinstance(obj, PrimitiveCylinderCfg):
            size_str = f"{obj.radius} {obj.height}"
            type_str = "cylinder"
        elif isinstance(obj, PrimitiveSphereCfg):
            size_str = f"{obj.radius}"
            type_str = "sphere"
        else:
            raise ValueError("Unknown primitive type")

        rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
        xml = f"""
        <mujoco model="{obj.name}_model">
        <worldbody>
            <body name="{type_str}_body" pos="{0} {0} {0}">
            <geom name="{type_str}_geom" type="{type_str}" size="{size_str}" rgba="{rgba_str}"/>
            </body>
        </worldbody>
        </mujoco>
        """
        return xml.strip()

    def _init_mujoco(self) -> mjcf.RootElement:
        mjcf_model = mjcf.RootElement()
        if self.scenario.sim_params.dt is not None:
            mjcf_model.option.timestep = self.scenario.sim_params.dt

        ## Optional: Add ground grid
        # mjcf_model.asset.add('texture', name="texplane", type="2d", builtin="checker", width=512, height=512, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3])
        # mjcf_model.asset.add('material', name="matplane", reflectance="0.", texture="texplane", texrepeat=[1, 1], texuniform=True)

        camera_max_width = 640
        camera_max_height = 480
        for camera in self.cameras:
            direction = np.array([
                camera.look_at[0] - camera.pos[0],
                camera.look_at[1] - camera.pos[1],
                camera.look_at[2] - camera.pos[2],
            ])
            direction = direction / np.linalg.norm(direction)
            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)

            camera_params = {
                "pos": f"{camera.pos[0]} {camera.pos[1]} {camera.pos[2]}",
                "mode": "fixed",
                "fovy": camera.vertical_fov,
                "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
            }
            mjcf_model.worldbody.add("camera", name=f"{camera.name}_custom", **camera_params)
            camera_max_width = max(camera_max_width, camera.width)
            camera_max_height = max(camera_max_height, camera.height)

        if camera_max_width > 640 or camera_max_height > 480:
            self._set_framebuffer_size(mjcf_model, camera_max_width, camera_max_height)

        if self.scenario.try_add_table:
            mjcf_model.asset.add(
                "texture",
                name="texplane",
                type="2d",
                builtin="checker",
                width=512,
                height=512,
                rgb1=[0, 0, 0],
                rgb2=[1.0, 1.0, 1.0],
            )
            mjcf_model.asset.add(
                "material", name="matplane", reflectance="0.2", texture="texplane", texrepeat=[1, 1], texuniform=True
            )
            ground = mjcf_model.worldbody.add(
                "geom",
                type="plane",
                pos="0 0 0",
                size="100 100 0.001",
                quat="1 0 0 0",
                condim="3",
                conaffinity="15",
                material="matplane",
            )
        self.object_body_names = []
        self.mj_objects = {}
        for obj in self.objects:
            if isinstance(obj, (PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg)):
                xml_str = self._create_primitive_xml(obj)
                obj_mjcf = mjcf.from_xml_string(xml_str)
            else:
                obj_mjcf = mjcf.from_path(obj.mjcf_path)

            if hasattr(obj, "scale") and obj.scale != (1.0, 1.0, 1.0):
                self._apply_scale_to_mjcf(obj_mjcf, obj.scale)

            obj_attached = mjcf_model.attach(obj_mjcf)
            if not obj.fix_base_link:
                obj_attached.add("freejoint")
            self.object_body_names.append(obj_attached.full_identifier)
            self.mj_objects[obj.name] = obj_mjcf

        robot_xml = mjcf.from_path(self._robot_path)

        if hasattr(self.robot, "scale") and self.robot.scale != (1.0, 1.0, 1.0):
            self._apply_scale_to_mjcf(robot_xml, self.robot.scale)

        robot_attached = mjcf_model.attach(robot_xml)
        if not self.robot.fix_base_link:
            robot_attached.add("freejoint")
        self.robot_attached = robot_attached
        self.mj_objects[self.robot.name] = robot_xml
        self._mujoco_robot_name = robot_xml.full_identifier
        return mjcf_model

    def _get_actuator_states(self, obj_name):
        """Get actuator states (targets and forces)."""
        actuator_states = {
            "dof_pos_target": {},
            "dof_vel_target": {},
            "dof_torque": {},
        }

        for actuator_id in range(self.physics.model.nu):
            actuator = self.physics.model.actuator(actuator_id)
            if actuator.name.startswith(self._mujoco_robot_name):
                clean_name = actuator.name[len(self._mujoco_robot_name) :]

                actuator_states["dof_pos_target"][clean_name] = float(
                    self.physics.data.ctrl[actuator_id].item()
                )  # Hardcoded to position control
                actuator_states["dof_vel_target"][clean_name] = None
                actuator_states["dof_torque"][clean_name] = float(self.physics.data.actuator_force[actuator_id].item())

        return actuator_states

    def _pack_state(self, body_ids: list[int]):
        """
        Pack pos(3), quat(4), lin_vel_world(3), ang_vel(3) for one-env MuJoCo.

        Args:
            body_ids: list of body IDs, e.g. [root_id] or [root_id] + body_ids_reindex

        Returns:
            root_np: numpy (13,)      — the first body
            body_np: numpy (n_body,13)     — n_body bodies
        """
        data = self.physics.data
        pos = data.xpos[body_ids]
        quat = data.xquat[body_ids]

        # angular ω (world) & v @ subtree_com
        w = data.cvel[body_ids, 0:3]
        v = data.cvel[body_ids, 3:6]

        # compute world‐frame linear velocity at body origin
        offset = data.xpos[body_ids] - data.subtree_com[body_ids]
        lin_world = v + np.cross(w, offset)

        full = np.concatenate([pos, quat, lin_world, w], axis=1)
        root_np = full[0]
        return root_np, full  # root, bodies

    def _get_states(self, env_ids: list[int] | None = None) -> list[dict]:
        object_states = {}
        for obj in self.objects:
            model_name = self.mj_objects[obj.name].model

            obj_body_id = self.physics.model.body(f"{model_name}/").id
            if isinstance(obj, ArticulationObjCfg):
                joint_names = self.get_joint_names(obj.name, sort=True)
                body_ids_reindex = self._get_body_ids_reindex(obj.name)

                root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)

                state = ObjectState(
                    root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                    body_names=self.get_body_names(obj.name),
                    body_state=torch.from_numpy(body_np).float().unsqueeze(0),  # (1,n_body,13)
                    joint_pos=torch.tensor([
                        self.physics.data.joint(f"{model_name}/{jn}").qpos.item() for jn in joint_names
                    ]).unsqueeze(0),
                    joint_vel=torch.tensor([
                        self.physics.data.joint(f"{model_name}/{jn}").qvel.item() for jn in joint_names
                    ]).unsqueeze(0),
                )
            else:
                root_np, _ = self._pack_state([obj_body_id])

                state = ObjectState(
                    root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                )
            object_states[obj.name] = state

        robot_states = {}
        for robot in [self.robot]:
            model_name = self.mj_objects[robot.name].model
            obj_body_id = self.physics.model.body(f"{model_name}/").id
            joint_names = self.get_joint_names(robot.name, sort=True)
            actuator_reindex = self._get_actuator_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)

            root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)

            state = RobotState(
                body_names=self.get_body_names(robot.name),
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),  # (1,n_body,13)
                joint_pos=torch.tensor([
                    self.physics.data.joint(f"{model_name}/{jn}").qpos.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_vel=torch.tensor([
                    self.physics.data.joint(f"{model_name}/{jn}").qvel.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_pos_target=torch.from_numpy(self.physics.data.ctrl[actuator_reindex]).unsqueeze(0),
                joint_vel_target=torch.from_numpy(self._current_vel_target).unsqueeze(0)
                if self._current_vel_target is not None
                else None,
                joint_effort_target=torch.from_numpy(self.physics.data.actuator_force[actuator_reindex]).unsqueeze(0),
            )
            robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_id = f"{camera.name}_custom"  # XXX: hard code camera id for now
            camera_states[camera.name] = {}
            if "rgb" in camera.data_types:
                rgb = self.physics.render(width=camera.width, height=camera.height, camera_id=camera_id, depth=False)
                rgb = torch.from_numpy(rgb.copy()).unsqueeze(0)
            if "depth" in camera.data_types:
                depth = self.physics.render(width=camera.width, height=camera.height, camera_id=camera_id, depth=True)
                depth = torch.from_numpy(depth.copy()).unsqueeze(0)
            state = CameraState(rgb=rgb, depth=depth)
            camera_states[camera.name] = state
        extras = self.get_extra()

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={}, extras=extras)

    def _set_root_state(self, obj_name, obj_state, zero_vel=False):
        """Set root position and rotation."""
        if "pos" not in obj_state and "rot" not in obj_state:
            return

        if obj_name == self.robot.name:
            if not self.robot.fix_base_link:
                root_joint = self.physics.data.joint(self._mujoco_robot_name)
                root_joint.qpos[:3] = obj_state.get("pos", [0, 0, 0])
                root_joint.qpos[3:7] = obj_state.get("rot", [1, 0, 0, 0])
                if zero_vel:
                    root_joint.qvel[:6] = 0
            else:
                root_body = self.physics.named.model.body_pos[self._mujoco_robot_name]
                root_body_quat = self.physics.named.model.body_quat[self._mujoco_robot_name]
                root_body[:] = obj_state.get("pos", [0, 0, 0])
                root_body_quat[:] = obj_state.get("rot", [1, 0, 0, 0])
        else:
            model_name = self.mj_objects[obj_name].model + "/"
            try:
                obj_joint = self.physics.data.joint(model_name)
                obj_joint.qpos[:3] = obj_state["pos"]
                obj_joint.qpos[3:7] = obj_state["rot"]
                if zero_vel:
                    obj_joint.qvel[:6] = 0
            except KeyError:
                obj_body = self.physics.named.model.body_pos[model_name]
                obj_body_quat = self.physics.named.model.body_quat[model_name]
                obj_body[:] = obj_state["pos"]
                obj_body_quat[:] = obj_state["rot"]

    def _set_joint_state(self, obj_name, obj_state, zero_vel=False):
        """Set joint positions."""
        if "dof_pos" not in obj_state:
            return

        for joint_name, joint_pos in obj_state["dof_pos"].items():
            full_joint_name = (
                f"{self._mujoco_robot_name}{joint_name}" if obj_name == self.robot.name else f"{obj_name}/{joint_name}"
            )
            joint = self.physics.data.joint(full_joint_name)
            joint.qpos = joint_pos
            if zero_vel:
                joint.qvel = 0
            try:
                actuator = self.physics.model.actuator(full_joint_name)
                self.physics.data.ctrl[actuator.id] = joint_pos
            except KeyError:
                pass

    def _set_states(self, states, env_ids=None, zero_vel=True):
        if len(states) > 1:
            raise ValueError("MujocoHandler only supports single env state setting")

        states_flat = [state["objects"] | state["robots"] for state in states]
        for obj_name, obj_state in states_flat[0].items():
            self._set_root_state(obj_name, obj_state, zero_vel)
            self._set_joint_state(obj_name, obj_state, zero_vel)
        self.physics.forward()

    def _disable_robotgravity(self):
        gravity_vec = np.array([0.0, 0.0, -9.81])

        self.physics.data.xfrc_applied[:] = 0
        for body_name in self.robot_body_names:
            body_id = self.physics.model.body(body_name).id
            force_vec = -gravity_vec * self.physics.model.body(body_name).mass
            self.physics.data.xfrc_applied[body_id, 0:3] = force_vec
            self.physics.data.xfrc_applied[body_id, 3:6] = 0

    def _compute_effort(self, actions):
        """Compute effort from actions using PD controller."""
        action_scaled = self._action_scale * actions
        joint_names = self.get_joint_names(self.robot.name, sort=True)
        robot_dof_pos = np.array([
            self.physics.data.joint(f"{self._mujoco_robot_name}{jn}").qpos[0] for jn in joint_names
        ])
        robot_dof_vel = np.array([
            self.physics.data.joint(f"{self._mujoco_robot_name}{jn}").qvel[0] for jn in joint_names
        ])

        if self._action_offset:
            effort = (
                self._p_gains * (action_scaled + self._robot_default_dof_pos - robot_dof_pos)
                - self._d_gains * robot_dof_vel
            )
        else:
            effort = self._p_gains * (action_scaled - robot_dof_pos) - self._d_gains * robot_dof_vel

        effort = np.clip(effort, -self._torque_limits, self._torque_limits)

        return effort

    def _apply_pd_control(self, actions):
        """Apply torque control using computed efforts."""
        effort = self._compute_effort(actions)

        joint_names = self.get_joint_names(self.robot.name, sort=True)
        for i in self._effort_controlled_joints:
            joint_name = joint_names[i]
            actuator_id = self.physics.model.actuator(f"{self._mujoco_robot_name}{joint_name}").id
            self.physics.data.ctrl[actuator_id] = effort[i]

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        self._actions_cache = actions

        # Extract velocity targets if present
        vel_targets = actions[0][obj_name].get("dof_vel_target", None)
        if vel_targets:
            joint_names = self.get_joint_names(self.robot.name, sort=True)
            self._current_vel_target = np.zeros(self._robot_num_dof)
            for i, joint_name in enumerate(joint_names):
                if joint_name in vel_targets:
                    self._current_vel_target[i] = vel_targets[joint_name]
        else:
            self._current_vel_target = None

        if self._manual_pd_on:
            joint_targets = actions[0][obj_name]["dof_pos_target"]
            joint_names = self.get_joint_names(self.robot.name, sort=True)

            self._current_action = np.zeros(self._robot_num_dof)
            for i, joint_name in enumerate(joint_names):
                if joint_name in joint_targets:
                    self._current_action[i] = joint_targets[joint_name]

            for i in self._position_controlled_joints:
                joint_name = joint_names[i]
                if joint_name in joint_targets:
                    actuator = self.physics.data.actuator(f"{self._mujoco_robot_name}{joint_name}")
                    actuator.ctrl = joint_targets[joint_name]
        else:
            joint_targets = actions[0][obj_name]["dof_pos_target"]
            for joint_name, target_pos in joint_targets.items():
                actuator = self.physics.data.actuator(f"{self._mujoco_robot_name}{joint_name}")
                actuator.ctrl = target_pos

    def set_actions(self, obj_name: str, actions):
        self._actions_cache = actions
        if self._manual_pd_on:
            self._current_action = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        else:
            self.physics.data.ctrl[:] = actions.detach().to(dtype=torch.float32, device="cpu").numpy()

    def refresh_render(self) -> None:
        self.physics.forward()  # Recomputes the forward dynamics without advancing the simulation.
        if self.viewer is not None:
            self.viewer.sync()

    def _simulate(self):
        if self._gravity_compensation:
            self._disable_robotgravity()

        # Apply torque control if manual PD is enabled
        if self._manual_pd_on:
            for _ in range(self.decimation):
                self._apply_pd_control(self._current_action)
                self.physics.step()
        else:
            self.physics.step(self.decimation)

        if not self.headless:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    ############################################################
    ## Utils
    ############################################################
    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg) or isinstance(
            self.object_dict[obj_name], BaseRobotCfg
        ):
            if obj_name == self.robot.name:
                prefix = self._mujoco_robot_name
            else:
                prefix = obj_name + "/"

            joint_names = [
                self.physics.model.joint(joint_id).name
                for joint_id in range(self.physics.model.njnt)
                if self.physics.model.joint(joint_id).name.startswith(prefix)
            ]

            if obj_name == self.robot.name:
                joint_names = [name[len(prefix) :] for name in joint_names]
            else:
                joint_names = [name.split("/")[-1] for name in joint_names]

            joint_names = [name for name in joint_names if name != ""]
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _get_actuator_names(self, robot_name: str) -> list[str]:
        assert isinstance(self.object_dict[robot_name], BaseRobotCfg)
        actuator_names = [self.physics.model.actuator(i).name for i in range(self.physics.model.nu)]

        robot_actuator_names = []
        for name in actuator_names:
            if name.startswith(self._mujoco_robot_name):
                joint_name = name[len(self._mujoco_robot_name) :]
                if joint_name:
                    robot_actuator_names.append(joint_name)

        all_joint_names = self.get_joint_names(robot_name)

        actuated_joint_names = []
        for joint_name in all_joint_names:
            if joint_name in self.robot.actuators:
                if self.robot.actuators[joint_name].fully_actuated is not False:
                    actuated_joint_names.append(joint_name)

        assert set(robot_actuator_names) == set(actuated_joint_names), (
            f"Actuator names {robot_actuator_names} do not match joint names {actuated_joint_names}"
        )

        return robot_actuator_names

    def _get_actuator_reindex(self, robot_name: str) -> list[int]:
        assert isinstance(self.object_dict[robot_name], BaseRobotCfg)
        origin_actuator_names = self._get_actuator_names(robot_name)
        sorted_actuator_names = sorted(origin_actuator_names)
        return [origin_actuator_names.index(name) for name in sorted_actuator_names]

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            names = [self.physics.model.body(i).name for i in range(self.physics.model.nbody)]
            names = [name.split("/")[-1] for name in names if name.split("/")[0] == obj_name]
            names = [name for name in names if name != ""]
            if sort:
                names.sort()
            return names
        else:
            return []

    def _get_body_ids_reindex(self, obj_name: str) -> list[int]:
        """
        Charlie: needs to be taken down
        Get the reindexed body ids for a given object. Reindex means the body reordered by the returned ids will be sorted by their names alphabetically.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: body ids in the order that making body names sorted alphabetically, length is number of bodies.

        Example:
            Suppose `obj_name = "h1"`, and the model has bodies:

            id 0: `"h1/"`
            id 1: `"h1/torso"`
            id 2: `"h1/left_leg"`
            id 3: `"h1/right_leg"`
            id 4: `"cube1/"`
            id 5: `"cube2/"`

            This function will return: `[2, 3, 1]`
        """
        assert isinstance(self.object_dict[obj_name], ArticulationObjCfg)
        if not hasattr(self, "_body_ids_reindex_cache"):
            self._body_ids_reindex_cache = {}
        if obj_name not in self._body_ids_reindex_cache:
            model_name = self.mj_objects[obj_name].model
            body_ids_origin = [
                bi
                for bi in range(self.physics.model.nbody)
                if self.physics.model.body(bi).name.split("/")[0] == model_name
                and self.physics.model.body(bi).name != f"{model_name}/"
            ]
            body_ids_reindex = [body_ids_origin[i] for i in self.get_body_reindex(obj_name)]
            self._body_ids_reindex_cache[obj_name] = body_ids_reindex
        return self._body_ids_reindex_cache[obj_name]

    ############################################################
    ## Misc
    ############################################################
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def episode_length_buf(self) -> list[int]:
        return [self._episode_length_buf]

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


MujocoParallelHandler = ParallelSimWrapper(MujocoHandler)
MujocoEnv: type[EnvWrapper[MujocoHandler]] = GymEnvWrapper(MujocoParallelHandler)
