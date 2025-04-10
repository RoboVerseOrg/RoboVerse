from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log

from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import TaskType
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, Obs


class MujocoHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []

        if scenario.num_envs > 1:
            raise ValueError("MujocoHandler only supports single envs, please run with --num_envs 1.")

        self._robot = scenario.robot
        self._mujoco_robot_name = None
        self._robot_num_dof = None
        self._robot_path = self._robot.mjcf_path
        self._gravity_compensation = not self._robot.enabled_gravity

        self.viewer = None
        self.cameras = []
        for camera in scenario.cameras:
            self.cameras.append(camera)
        self._episode_length_buf = 0

        self.replay_traj = False
        self.use_task_decimation = False

        # FIXME: hard code decimation for now
        if self.use_task_decimation:
            self.decimation = self.scenario.decimation
        elif self.replay_traj:
            log.warning("Warning: hard coding decimation to 1 for object states")
            self.decimation = 1
        elif self.task is not None and self.task.task_type == TaskType.LOCOMOTION:
            self.decimation = self.scenario.decimation
        else:
            log.warning("Warning: hard coding decimation to 25 for replaying trajectories")
            self.decimation = 25

    def launch(self) -> None:
        model = self._init_mujoco()
        self.physics = mjcf.Physics.from_mjcf_model(model)
        self.data = self.physics.data

        self.body_names = [self.physics.model.body(i).name for i in range(self.physics.model.nbody)]
        self.robot_body_names = [
            body_name for body_name in self.body_names if body_name.startswith(self._mujoco_robot_name)
        ]

        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
            self.viewer.sync()

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
                "resolution": f"{camera.width} {camera.height}",
            }
            mjcf_model.worldbody.add("camera", name=f"{camera.name}_custom", **camera_params)
            camera_max_width = max(camera_max_width, camera.width)
            camera_max_height = max(camera_max_height, camera.height)

        for child in mjcf_model.visual._children:
            if child.tag == "global":
                child.offwidth = camera_max_width
                child.offheight = camera_max_height

        # Add ground grid, light, and skybox
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
        object_paths = []
        for obj in self.objects:
            object_paths.append(obj.mjcf_path)
        for i, (obj, obj_path) in enumerate(zip(self.objects, object_paths)):
            if isinstance(obj, (PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg)):
                xml_str = self._create_primitive_xml(obj)
                obj_mjcf = mjcf.from_xml_string(xml_str)
            else:
                obj_mjcf = mjcf.from_path(obj_path)
            obj_attached = mjcf_model.attach(obj_mjcf)
            if not obj.fix_base_link:
                obj_attached.add("freejoint")
            self.object_body_names.append(obj_attached.full_identifier)
            self.mj_objects[obj.name] = obj_mjcf

        robot_xml = mjcf.from_path(self._robot_path)
        robot_attached = mjcf_model.attach(robot_xml)
        if not self._robot.fix_base_link:
            robot_attached.add("freejoint")
        self.robot_attached = robot_attached
        self.mj_objects[self._robot.name] = robot_xml
        self._mujoco_robot_name = robot_xml.full_identifier
        return mjcf_model

    def _get_root_state(self, obj, obj_name):
        """Get root position, rotation, velocity for an object or robot."""
        if obj == self._robot:
            if not self._robot.fix_base_link:
                root_joint = self.physics.data.joint(self._mujoco_robot_name)
                return {
                    "pos": torch.tensor(root_joint.qpos[:3], dtype=torch.float32),
                    "rot": torch.tensor(root_joint.qpos[3:7], dtype=torch.float32),
                    "vel": torch.tensor(root_joint.qvel[:3], dtype=torch.float32),
                    "ang_vel": torch.tensor(root_joint.qvel[3:6], dtype=torch.float32),
                }
            else:
                root_body_id = self.physics.model.body(self._mujoco_robot_name).id
                return {
                    "pos": torch.tensor(self.physics.data.xpos[root_body_id], dtype=torch.float32),
                    "rot": torch.tensor(self.physics.data.xquat[root_body_id], dtype=torch.float32),
                    "vel": torch.tensor(self.physics.data.cvel[root_body_id][:3], dtype=torch.float32),
                    "ang_vel": torch.tensor(self.physics.data.cvel[root_body_id][3:6], dtype=torch.float32),
                }
        else:
            model_name = self.mj_objects[obj_name].model + "/"
            try:
                obj_joint = self.physics.data.joint(model_name)
                return {
                    "pos": torch.tensor(obj_joint.qpos[:3], dtype=torch.float32),
                    "rot": torch.tensor(obj_joint.qpos[3:7], dtype=torch.float32),
                    "vel": torch.tensor(obj_joint.qvel[:3], dtype=torch.float32),
                    "ang_vel": torch.tensor(obj_joint.qvel[3:6], dtype=torch.float32),
                }
            except KeyError:
                obj_body_id = self.physics.model.body(model_name).id
                return {
                    "pos": torch.tensor(self.physics.data.xpos[obj_body_id], dtype=torch.float32),
                    "rot": torch.tensor(self.physics.data.xquat[obj_body_id], dtype=torch.float32),
                    "vel": torch.tensor(self.physics.data.cvel[obj_body_id][:3], dtype=torch.float32),
                    "ang_vel": torch.tensor(self.physics.data.cvel[obj_body_id][3:6], dtype=torch.float32),
                }

    def _get_joint_states(self, obj_name):
        """Get joint positions and velocities."""
        joint_states = {"dof_pos": {}, "dof_vel": {}}

        for joint_id in range(self.physics.model.njnt):
            joint_name = self.physics.model.joint(joint_id).name
            if joint_name.startswith(f"{obj_name}/") or (
                obj_name == self._robot.name and joint_name.startswith(self._mujoco_robot_name)
            ):
                clean_joint_name = joint_name.split("/")[-1]
                if clean_joint_name == "":
                    continue
                joint = self.physics.data.joint(joint_name)

                if len(joint.qpos.shape) == 0 or joint.qpos.shape == (1,):
                    joint_states["dof_pos"][clean_joint_name] = joint.qpos.item()
                    joint_states["dof_vel"][clean_joint_name] = joint.qvel.item()
                else:  # free joint
                    joint_states["dof_pos"][clean_joint_name] = torch.tensor(joint.qpos.copy(), dtype=torch.float32)
                    joint_states["dof_vel"][clean_joint_name] = torch.tensor(joint.qvel.copy(), dtype=torch.float32)

        return joint_states

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

    def get_states(self, env_ids: list[int] | None = None) -> list[dict]:
        state = {"objects": {}, "robots": {}, "cameras": {}}

        ## Root states -- for both robot and object
        for obj in self.objects + [self._robot]:
            object_type = "robots" if obj == self._robot else "objects"
            state[object_type][obj.name] = {}
            state[object_type][obj.name].update(self._get_root_state(obj, obj.name))
            state[object_type][obj.name].update(self._get_joint_states(obj.name))
            ## Actuator states -- for robot
            if obj == self._robot:
                ## TODO: read from simulator instead of cache
                state[object_type][obj.name].update(self._get_actuator_states(obj.name))
                joint_names = [
                    self.physics.model.joint(joint_id).name.split("/")[-1]
                    for joint_id in range(self.physics.model.njnt)
                    if self.physics.model.joint(joint_id).name.startswith(self._mujoco_robot_name)
                ]
                if self.actions_cache:
                    state[object_type][obj.name]["dof_pos_target"] = {
                        joint_name: self.actions_cache[0]["dof_pos_target"][joint_name] for joint_name in joint_names
                    }
                else:
                    state[object_type][obj.name]["dof_pos_target"] = None

        ## Body states -- for both robot and object
        object_names = [obj.name for obj in self.objects]
        robot_names = [self._robot.name]
        for obj in self.objects:
            state["objects"][obj.name]["body"] = {}
        for obj in [self._robot]:
            state["robots"][obj.name]["body"] = {}
        for body_id in range(self.physics.model.nbody):
            body = self.physics.model.body(body_id)
            body_full_name = body.name  # e.g. h1/left_shoulder_pitch_link
            robot_name_body_belong = body_full_name.split("/")[0]  # e.g. h1
            body_base_name = body_full_name.split("/")[-1]  # e.g. left_shoulder_pitch_link
            if robot_name_body_belong in robot_names:
                state["robots"][robot_name_body_belong]["body"][body_base_name] = {
                    # Position and orientation in world frame
                    "pos": torch.tensor(
                        self.physics.data.xpos[body_id], dtype=torch.float32
                    ),  # Position in world frame (x,y,z)
                    "rot": torch.tensor(
                        self.physics.data.xquat[body_id], dtype=torch.float32
                    ),  # Orientation in world frame (quaternion)
                    # Linear and angular velocity in world frame
                    "vel": torch.tensor(
                        self.physics.data.cvel[body_id][:3], dtype=torch.float32
                    ),  # Linear velocity in world frame (x,y,z)
                    "ang_vel": torch.tensor(
                        self.physics.data.cvel[body_id][3:6], dtype=torch.float32
                    ),  # Angular velocity in world frame (x,y,z)
                    # Contact forces and torques in local frame
                    "contact_force": torch.tensor(
                        self.physics.data.cfrc_ext[body_id][:3], dtype=torch.float32
                    ),  # Contact force in local frame (x,y,z)
                    "contact_torque": torch.tensor(
                        self.physics.data.cfrc_ext[body_id][3:6], dtype=torch.float32
                    ),  # Contact torque in local frame (x,y,z)
                }
            elif robot_name_body_belong in object_names:
                state["objects"][robot_name_body_belong]["body"][body_base_name] = {
                    # Position and orientation in world frame
                    "pos": torch.tensor(
                        self.physics.data.xpos[body_id], dtype=torch.float32
                    ),  # Position in world frame (x,y,z)
                    "rot": torch.tensor(
                        self.physics.data.xquat[body_id], dtype=torch.float32
                    ),  # Orientation in world frame (quaternion)
                    # Linear and angular velocity in world frame
                    "vel": torch.tensor(
                        self.physics.data.cvel[body_id][:3], dtype=torch.float32
                    ),  # Linear velocity in world frame (x,y,z)
                    "ang_vel": torch.tensor(
                        self.physics.data.cvel[body_id][3:6], dtype=torch.float32
                    ),  # Angular velocity in world frame (x,y,z)
                    # Contact forces and torques in local frame
                    "contact_force": torch.tensor(
                        self.physics.data.cfrc_ext[body_id][:3], dtype=torch.float32
                    ),  # Contact force in local frame (x,y,z)
                    "contact_torque": torch.tensor(
                        self.physics.data.cfrc_ext[body_id][3:6], dtype=torch.float32
                    ),  # Contact torque in local frame (x,y,z)
                }

        for camera in self.cameras:
            camera_id = f"{camera.name}_custom"  # XXX: hard code camera id for now
            state["cameras"][camera.name] = {}

            if "rgb" in camera.data_types:
                rgb_img = self.physics.render(
                    width=camera.width,
                    height=camera.height,
                    camera_id=camera_id,
                    depth=False,
                )
                state["cameras"][camera.name].update({"rgb": torch.from_numpy(rgb_img.copy())})
            if "depth" in camera.data_types:
                depth_img = self.physics.render(
                    width=camera.width,
                    height=camera.height,
                    camera_id=camera_id,
                    depth=True,
                )
                state["cameras"][camera.name].update({"depth": torch.from_numpy(depth_img.copy())})

        return [state]

    def _set_root_state(self, obj_name, obj_state, zero_vel=False):
        """Set root position and rotation."""
        if "pos" not in obj_state and "rot" not in obj_state:
            return

        if obj_name == self._robot.name:
            if not self._robot.fix_base_link:
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
                f"{self._mujoco_robot_name}{joint_name}" if obj_name == self._robot.name else f"{obj_name}/{joint_name}"
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

    def set_states(self, states, env_ids=None, zero_vel=True):
        if len(states) > 1:
            raise ValueError("MujocoHandler only supports single env state setting")

        states_flat = [state["objects"] | state["robots"] for state in states]
        for obj_name, obj_state in states_flat[0].items():
            self._set_root_state(obj_name, obj_state, zero_vel)
            self._set_joint_state(obj_name, obj_state, zero_vel)

    def _disable_robotgravity(self):
        gravity_vec = np.array([0.0, 0.0, -9.81])

        self.physics.data.xfrc_applied[:] = 0
        for body_name in self.robot_body_names:
            body_id = self.physics.model.body(body_name).id
            force_vec = -gravity_vec * self.physics.model.body(body_name).mass
            self.physics.data.xfrc_applied[body_id, 0:3] = force_vec
            self.physics.data.xfrc_applied[body_id, 3:6] = 0

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        self._actions_cache = actions
        joint_targets = actions[0]["dof_pos_target"]

        if self.replay_traj:
            for joint_name, target_pos in joint_targets.items():
                joint = self.physics.data.joint(f"{self._mujoco_robot_name}{joint_name}")
                joint.qpos = target_pos
        else:
            for joint_name, target_pos in joint_targets.items():
                actuator = self.physics.data.actuator(f"{self._mujoco_robot_name}{joint_name}")
                actuator.ctrl = target_pos

    def refresh_render(self) -> None:
        self.physics.forward()  # Recomputes the forward dynamics without advancing the simulation.
        if self.viewer is not None:
            self.viewer.sync()

    def simulate(self):
        if self._gravity_compensation:
            self._disable_robotgravity()

        self.physics.step(self.decimation)
        if not self.headless:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    ############################################################
    ## Utils
    ############################################################
    def get_observation(self) -> Obs | None:
        states = self.get_states()
        if len(self.cameras) > 0:
            rgbs = [state["cameras"][self.cameras[0].name]["rgb"] for state in states]
            obs = {"rgb": torch.stack(rgbs, dim=0)}
        else:
            obs = {}
        return obs

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


MujocoEnv: type[EnvWrapper[MujocoHandler]] = GymEnvWrapper(MujocoHandler)
