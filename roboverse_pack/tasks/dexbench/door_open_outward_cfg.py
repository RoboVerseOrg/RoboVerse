"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.sensors import ForceTorqueSensorCfg
from metasim.types import DictEnvState
from metasim.utils import configclass, math
from metasim.utils.bidex_util import randomize_rotation
from metasim.utils.state import ObjectState, RobotState, TensorState
from roboverse_pack.robots.franka_allegro_left_cfg import FrankaAllegroHandLeftCfg
from roboverse_pack.robots.franka_allegro_right_cfg import FrankaAllegroHandRightCfg
from roboverse_pack.robots.franka_shadow_left_cfg import FrankaShadowHandLeftCfg
from roboverse_pack.robots.franka_shadow_right_cfg import FrankaShadowHandRightCfg
from roboverse_pack.robots.franka_xhand_left_cfg import FrankaXHandLeftCfg
from roboverse_pack.robots.franka_xhand_right_cfg import FrankaXHandRightCfg

from .base_cfg import BaseRLTaskCfg, SimParamCfg

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from DexterousHands/bidexhands/tasks/shadow_hand_over.py
########################################################


@configclass
class DoorOpenOutwardCfg(BaseRLTaskCfg):
    """class for bidex door open outward tasks."""

    is_testing = False
    episode_length = 250
    traj_filepath = "roboverse_data/trajs/bidex/ShadowHandOpenOutward/v2/initial_state_v2.json"
    device = "cuda:0"
    num_envs = None
    obs_type = "state"
    use_prio = True
    current_object_type = "door"
    current_robot_type = "shadow"
    objects_cfg = {
        "door": ArticulationObjCfg(
            name="door",
            urdf_path="roboverse_data/assets/bidex/objects/urdf/door_mobility/door_mobility.urdf",
            usd_path="roboverse_data/assets/bidex/objects/usd/door_mobility/door_mobility.usd",
            disable_gravity=True,
            fix_base_link=True,
            default_density=500.0,
            override_com=True,
            override_inertia=True,
            use_mesh_materials=True,
            mesh_normal_mode="vertex",
            friction=0.1,
            use_vhacd=True,
            stiffness=100.0,
            damping=100.0,
        ),
    }
    objects = []
    robots = []
    env_spacing = 1.5
    sim_params = SimParamCfg(
        dt=1.0 / 60.0,
        contact_offset=0.002,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.2,
        num_threads=4,
        use_gpu_pipeline=True,
        use_gpu=True,
        substeps=2,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )
    right_goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    left_goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    sensors = []
    r_handle_name = "link_1"  # Name of the right hand handle in the body state
    l_handle_name = "link_2"  # Name of the left hand handle in the body state
    r_handle_idx = None  # Index of the right hand handle in the body state
    l_handle_idx = None  # Index of the left hand handle in the body state
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 1.0  # Scale for force and torque observations
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    action_penalty_scale = 0
    reach_goal_bonus = 250.0
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0
    leave_penalty = 5.0

    def set_sim_params(self, sim_type=None) -> None:
        """Set the simulation parameters based on the simulator type."""
        if sim_type is not None:
            self.sim = sim_type
        if self.sim in ["isaacgym"]:
            self.sim_params.dt = 1.0 / 60.0
            self.sim_params.substeps = 2
            self.sim_params.num_threads = 4
            self.decimation = 1
        elif self.sim in ["isaacsim"]:
            self.sim_params.dt = 1.0 / 60.0
            self.sim_params.substeps = 2
            self.sim_params.num_threads = 4
            self.decimation = 1
            self.env_spacing = 10.0
        else:
            raise ValueError(f"Unknown simulator type: {self.sim}")
        self.dt = self.sim_params.dt

    def set_objects(self) -> None:
        """Set the objects for the door close inward task."""
        self.objects.append(self.objects_cfg[self.current_object_type])
        if self.current_robot_type == "shadow":
            self.robots = [
                FrankaShadowHandRightCfg(
                    use_vhacd=False,
                    hand_controller="dof_pos",
                    isaacgym_read_mjcf=True,
                    name="right_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
                FrankaShadowHandLeftCfg(
                    use_vhacd=False,
                    hand_controller="dof_pos",
                    isaacgym_read_mjcf=True,
                    name="left_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
            ]
            self.robot_init_state = {
                "right_hand": {
                    "pos": torch.tensor([0.88, 0.2, 0.0]),
                    "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                    "dof_pos": {
                        "FFJ1": 0.0,
                        "FFJ2": 0.0,
                        "FFJ3": 0.0,
                        "FFJ4": 0.0,
                        "LFJ1": 0.0,
                        "LFJ2": 0.0,
                        "LFJ3": 0.0,
                        "LFJ4": 0.0,
                        "LFJ5": 0.0,
                        "MFJ1": 0.0,
                        "MFJ2": 0.0,
                        "MFJ3": 0.0,
                        "MFJ4": 0.0,
                        "RFJ1": 0.0,
                        "RFJ2": 0.0,
                        "RFJ3": 0.0,
                        "RFJ4": 0.0,
                        "THJ1": 0.0,
                        "THJ2": 0.0,
                        "THJ3": 0.0,
                        "THJ4": 0.0,
                        "THJ5": 0.0,
                        "WRJ1": 0.0,
                        "WRJ2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": 2.35619445,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.88, -0.2, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
                    "dof_pos": {
                        "FFJ1": 0.0,
                        "FFJ2": 0.0,
                        "FFJ3": 0.0,
                        "FFJ4": 0.0,
                        "LFJ1": 0.0,
                        "LFJ2": 0.0,
                        "LFJ3": 0.0,
                        "LFJ4": 0.0,
                        "LFJ5": 0.0,
                        "MFJ1": 0.0,
                        "MFJ2": 0.0,
                        "MFJ3": 0.0,
                        "MFJ4": 0.0,
                        "RFJ1": 0.0,
                        "RFJ2": 0.0,
                        "RFJ3": 0.0,
                        "RFJ4": 0.0,
                        "THJ1": 0.0,
                        "THJ2": 0.0,
                        "THJ3": 0.0,
                        "THJ4": 0.0,
                        "THJ5": 0.0,
                        "WRJ1": 0.0,
                        "WRJ2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": -0.785398,
                    },
                },
            }
        elif self.current_robot_type == "allegro":
            self.robots = [
                FrankaAllegroHandRightCfg(
                    use_vhacd=False, 
                    hand_controller="dof_pos", 
                    name="right_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
                FrankaAllegroHandLeftCfg(
                    use_vhacd=False, 
                    hand_controller="dof_pos", 
                    name="left_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
            ]
            self.robot_init_state = {
                "right_hand": {
                    "pos": torch.tensor([0.65, 0.2, 0.0]),
                    "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                    "dof_pos": {
                        "joint_0": 0.0,
                        "joint_1": 0.0,
                        "joint_2": 0.0,
                        "joint_3": 0.0,
                        "joint_4": 0.0,
                        "joint_5": 0.0,
                        "joint_6": 0.0,
                        "joint_7": 0.0,
                        "joint_8": 0.0,
                        "joint_9": 0.0,
                        "joint_10": 0.0,
                        "joint_11": 0.0,
                        "joint_12": 0.0,
                        "joint_13": 0.0,
                        "joint_14": 1.64,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": 2.35619445,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.65, -0.2, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
                    "dof_pos": {
                        "joint_0": 0.0,
                        "joint_1": 0.0,
                        "joint_2": 0.0,
                        "joint_3": 0.0,
                        "joint_4": 0.0,
                        "joint_5": 0.0,
                        "joint_6": 0.0,
                        "joint_7": 0.0,
                        "joint_8": 0.0,
                        "joint_9": 0.0,
                        "joint_10": 0.0,
                        "joint_11": 0.0,
                        "joint_12": 0.0,
                        "joint_13": 0.0,
                        "joint_14": 1.64,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": -0.785398,
                    },
                },
            }
        elif self.current_robot_type == "xhand":
            self.robots = [
                FrankaXHandRightCfg(
                    use_vhacd=False, 
                    hand_controller="dof_pos", 
                    name="right_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
                FrankaXHandLeftCfg(
                    use_vhacd=False, 
                    hand_controller="dof_pos", 
                    name="left_hand",
                    arm_translation_scale=0.05,
                    arm_orientation_scale=0.1,
                    arm_controller="ik",
                ),
            ]
            self.robot_init_state = {
                "right_hand": {
                    "pos": torch.tensor([0.58, 0.2, 0.0]),
                    "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                    "dof_pos": {
                        "right_hand_index_bend_joint": 0.0,
                        "right_hand_index_joint1": 0.0,
                        "right_hand_index_joint2": 0.0,
                        "right_hand_mid_joint1": 0.0,
                        "right_hand_mid_joint2": 0.0,
                        "right_hand_pinky_joint1": 0.0,
                        "right_hand_pinky_joint2": 0.0,
                        "right_hand_ring_joint1": 0.0,
                        "right_hand_ring_joint2": 0.0,
                        "right_hand_thumb_bend_joint": 0.0,
                        "right_hand_thumb_rota_joint1": 0.0,
                        "right_hand_thumb_rota_joint2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": 2.35619445,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.58, -0.2, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
                    "dof_pos": {
                        "left_hand_index_bend_joint": 0.0,
                        "left_hand_index_joint1": 0.0,
                        "left_hand_index_joint2": 0.0,
                        "left_hand_mid_joint1": 0.0,
                        "left_hand_mid_joint2": 0.0,
                        "left_hand_pinky_joint1": 0.0,
                        "left_hand_pinky_joint2": 0.0,
                        "left_hand_ring_joint1": 0.0,
                        "left_hand_ring_joint2": 0.0,
                        "left_hand_thumb_bend_joint": 0.0,
                        "left_hand_thumb_rota_joint1": 0.0,
                        "left_hand_thumb_rota_joint2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 3.1415928,
                        "panda_joint7": -0.785398,
                    },
                },
            }
        self.step_actions_shape = 0
        for robot in self.robots:
            self.step_actions_shape += robot.num_joints
        self.action_shape = 0
        for robot in self.robots:
            if robot.hand_controller == "ik":
                self.action_shape += 6 * robot.num_fingertips
            elif robot.hand_controller == "dof_pos":
                self.action_shape += robot.num_actuated_joints - robot.num_arm_joints
            if robot.arm_controller in ["ik"]:
                self.action_shape += 6
            elif robot.arm_controller in ["ik_abs"]:
                self.action_shape += 7
            elif robot.arm_controller in ["dof_pos"]:
                self.action_shape += robot.num_arm_joints
        for name in self.robots[0].fingertips:
            r_name = "right" + name
            self.sensors.append(ForceTorqueSensorCfg(base_link=(self.robots[0].name, name), name=r_name))
        for name in self.robots[1].fingertips:
            l_name = "left" + name
            self.sensors.append(ForceTorqueSensorCfg(base_link=(self.robots[1].name, name), name=l_name))

    def set_init_states(self) -> None:
        """Set the initial states for the door close inward task."""
        self.state_shape = 0
        for robot in self.robots:
            self.state_shape += robot.observation_shape
            self.state_shape += robot.num_fingertips * 6  # fingertip forces
        self.state_shape += self.action_shape
        if self.use_prio:
            self.state_shape += 6
        self.obs_shape = {
            "state": (self.state_shape,),
        }
        if self.obs_type == "state":
            self.cameras = []
            if not self.use_prio:
                raise ValueError("State observation type requires proprioception to be enabled.")
        elif self.obs_type == "rgb":
            assert hasattr(self, "img_h") and hasattr(self, "img_w"), "Image height and width must be set."
            self.cameras = [
                PinholeCameraCfg(
                    name="camera_0", 
                    width=self.img_w, 
                    height=self.img_h, 
                    pos=(1.0, 0.0, 1.6), 
                    look_at=(0.0, -0.0, 0.6)
                )
            ]
            self.obs_shape["rgb"] = (3, self.img_h, self.img_w)
        self.init_states = {
            "objects": {
                self.current_object_type: {
                    "pos": torch.tensor([0, 0.0, 0.75]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "joint_1": 0.0,
                        "joint_2": 0.0,
                    },
                },
            },
            "robots": self.robot_init_state,
        }
        self.robot_dof_default_pos = {}
        self.robot_dof_default_pos_cpu = {}
        for robot in self.robots:
            self.robot_dof_default_pos[robot.name] = torch.tensor(
                list(self.init_states["robots"][robot.name]["dof_pos"].values()),
                dtype=torch.float32,
                device=self.device,
            )
            self.robot_dof_default_pos_cpu[robot.name] = self.robot_dof_default_pos[robot.name].cpu()
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def scale_action_fn(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to the range of the action space.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        step_actions = torch.zeros((self.num_envs, self.step_actions_shape), device=self.device)
        actions_start = 0
        step_actions_start = 0
        for robot in self.robots:
            if robot.arm_controller == "dof_pos":
                dpose = actions[:, actions_start : actions_start + robot.num_arm_joints]
                if robot.use_relative_control:
                    dpose = dpose * self.sim_params.dt * self.decimation
                arm_dof_targets = robot.control_arm(dpose, dpose.shape[0], dpose.device)
                actions_start += robot.num_arm_joints
            elif robot.arm_controller in ["ik"]:
                dpose = actions[:, actions_start : actions_start + 6]
                arm_dof_targets = robot.control_arm(dpose, dpose.shape[0], dpose.device)
                actions_start += 6
            elif robot.arm_controller in ["ik_abs"]:
                pose_action = actions[:, actions_start : actions_start + 7]
                arm_dof_targets = robot.control_arm(pose_action, pose_action.shape[0], pose_action.device)
                actions_start += 7
            step_actions[:, step_actions_start + robot.arm_dof_idx] = arm_dof_targets
            if robot.hand_controller == "ik":
                ft_action = actions[:, actions_start : actions_start + 6 * robot.num_fingertips].view(
                    self.num_envs, robot.num_fingertips, 6
                )
                ft_pos = ft_action[:, :, :3]
                ft_rot = ft_action[:, :, 3:]
                hand_dof_pos = robot.control_hand_ik(ft_pos, ft_rot)
                step_actions[:, step_actions_start + robot.hand_dof_idx] = hand_dof_pos
                actions_start += 6 * robot.num_fingertips
                step_actions_start += robot.num_joints
            elif robot.hand_controller == "dof_pos":
                hand_dof_pos = robot.scale_hand_action(
                    actions[:, actions_start : actions_start + robot.num_actuated_joints - robot.num_arm_joints]
                )
                step_actions[:, step_actions_start + robot.hand_dof_idx] = hand_dof_pos
                actions_start += robot.num_actuated_joints - robot.num_arm_joints
                step_actions_start += robot.num_joints

        return step_actions

    def observation_fn(self, envstates: list[DictEnvState], actions: torch.Tensor, device=None) -> torch.Tensor:
        """Observation function for shadow hand over tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.
            actions (torch.Tensor): Actions taken by the agents in the environment, shape (num_envs, num_actions).
            device (torch.device | None): The device to use for the observations. If None, use the task's device.

        Returns:
            Compute the observations of all environment. The observation is composed of three parts:
            the state values of the left and right hands, and the information of objects and target.
            The state values of the left and right hands were the same for each task, including hand
            joint and finger positions, velocity, and force information. The detail 422-dimensional
            observational space as shown in below:

            Index       Description
            right robot proceptual observation
            right robot fingertip forces
            right robot actions
            left  robot proceptual observation
            left  robot fingertip forces
            left  robot actions
            right handle position
            left handle position
            visual observation, currently RGB image (3 x 256 x 256)
        """
        if device is None:
            device = self.device
        num_envs = envstates.robots[self.robots[0].name].root_state.shape[0]
        if self.num_envs is None:
            self.num_envs = num_envs
        obs = {
            "state": torch.zeros((num_envs, self.state_shape), dtype=torch.float32, device=device),
        }
        state_obs = obs["state"]
        t = 0
        state_obs[:, : self.robots[0].observation_shape] = self.robots[0].observation(use_palm=True)
        t += self.robots[0].observation_shape
        for name in self.robots[0].fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            r_name = "right" + name
            force = envstates.sensors[r_name].force  # (num_envs, 3)
            torque = envstates.sensors[r_name].torque  # (num_envs, 3)
            state_obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        state_obs[:, t : t + self.action_shape // 2] = actions[:, : self.action_shape // 2]  # actions for right hand
        t += self.action_shape // 2
        state_obs[:, t : t + self.robots[1].observation_shape] = self.robots[1].observation(use_palm=True)
        t += self.robots[1].observation_shape
        for name in self.robots[1].fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            l_name = "left" + name
            force = envstates.sensors[l_name].force
            torque = envstates.sensors[l_name].torque
            state_obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        state_obs[:, t : t + self.action_shape // 2] = actions[:, self.action_shape // 2 :]  # actions for left hand
        t += self.action_shape // 2
        if self.r_handle_idx is None:
            self.r_handle_idx = envstates.objects[self.current_object_type].body_names.index(self.r_handle_name)
        if self.l_handle_idx is None:
            self.l_handle_idx = envstates.objects[self.current_object_type].body_names.index(self.l_handle_name)
        if self.use_prio:
            door_right_handle_pos = envstates.objects[self.current_object_type].body_state[:, self.r_handle_idx, :3]
            door_right_handle_rot = envstates.objects[self.current_object_type].body_state[:, self.r_handle_idx, 3:7]
            door_right_handle_pos = door_right_handle_pos + math.quat_apply(
                door_right_handle_rot, self.y_unit_tensor * -0.5
            )
            door_right_handle_pos = door_right_handle_pos + math.quat_apply(
                door_right_handle_rot, self.x_unit_tensor * 0.39
            )
            door_right_handle_pos = door_right_handle_pos + math.quat_apply(
                door_right_handle_rot, self.z_unit_tensor * -0.04
            )
            door_left_handle_pos = envstates.objects[self.current_object_type].body_state[:, self.l_handle_idx, :3]
            door_left_handle_rot = envstates.objects[self.current_object_type].body_state[:, self.l_handle_idx, 3:7]
            door_left_handle_pos = door_left_handle_pos + math.quat_apply(
                door_left_handle_rot, self.y_unit_tensor * -0.5
            )
            door_left_handle_pos = door_left_handle_pos + math.quat_apply(
                door_left_handle_rot, self.x_unit_tensor * -0.39
            )
            door_left_handle_pos = door_left_handle_pos + math.quat_apply(
                door_left_handle_rot, self.z_unit_tensor * -0.04
            )
            state_obs[:, t : t + 3] = door_right_handle_pos  # right handle position
            state_obs[:, t + 3 : t + 6] = door_left_handle_pos  # left handle position
            t += 6
        obs["state"] = state_obs
        if self.obs_type == "rgb":
            obs["rgb"] = (
                envstates.cameras["camera_0"].rgb.permute(0, 3, 1, 2) / 255.0
            )  # (num_envs, H, W, 3) -> (num_envs, 3, H, W)
        return obs

    def reward_fn(
        self,
        envstates: list[DictEnvState],
        actions,
        reset_buf,
        reset_goal_buf,
        episode_length_buf,
        success_buf,
    ) -> torch.Tensor:
        """Compute the reward of all environment. The core function is compute_hand_reward.

        Args:
            envstates (list[EnvState]): States of the environment
            actions (tensor): Actions of agents in the all environment, shape (num_envs, num_actions)
            episode_length_buf (torch.Tensor): The episode length buffer, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
            success_rate (float): The success rate of the environment, used to determine the success condition

        Returns:
            reward (torch.Tensor): The reward of all environments at this time, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
        """
        door_right_handle_pos = envstates.objects[self.current_object_type].body_state[:, self.r_handle_idx, :3]
        door_right_handle_rot = envstates.objects[self.current_object_type].body_state[:, self.r_handle_idx, 3:7]
        door_right_handle_pos = door_right_handle_pos + math.quat_apply(
            door_right_handle_rot, self.y_unit_tensor * -0.5
        )
        door_right_handle_pos = door_right_handle_pos + math.quat_apply(
            door_right_handle_rot, self.x_unit_tensor * 0.39
        )
        door_right_handle_pos = door_right_handle_pos + math.quat_apply(
            door_right_handle_rot, self.z_unit_tensor * -0.04
        )

        door_left_handle_pos = envstates.objects[self.current_object_type].body_state[:, self.l_handle_idx, :3]
        door_left_handle_rot = envstates.objects[self.current_object_type].body_state[:, self.l_handle_idx, 3:7]
        door_left_handle_pos = door_left_handle_pos + math.quat_apply(door_left_handle_rot, self.y_unit_tensor * -0.5)
        door_left_handle_pos = door_left_handle_pos + math.quat_apply(door_left_handle_rot, self.x_unit_tensor * -0.39)
        door_left_handle_pos = door_left_handle_pos + math.quat_apply(door_left_handle_rot, self.z_unit_tensor * -0.04)

        door_dof_pos = envstates.objects[self.current_object_type].joint_pos

        right_hand_reward, right_hand_dist = self.robots[0].reward(door_right_handle_pos)
        left_hand_reward, left_hand_dist = self.robots[1].reward(door_left_handle_pos)
        (reward, reset_buf, reset_goal_buf, success_buf) = compute_task_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            episode_length_buf=episode_length_buf,
            success_buf=success_buf,
            max_episode_length=self.episode_length,
            door_right_handle_pos=door_right_handle_pos,
            door_left_handle_pos=door_left_handle_pos,
            door_dof_pos=door_dof_pos,
            right_hand_reward=right_hand_reward,
            left_hand_reward=left_hand_reward,
            right_hand_dist=right_hand_dist,
            left_hand_dist=left_hand_dist,
            action_penalty_scale=self.action_penalty_scale,
            actions=actions,
            reach_goal_bonus=self.reach_goal_bonus,
        )
        return reward, reset_buf, reset_goal_buf, success_buf

    def goal_reset_fn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the goal position and rotation for the environment.

        Args:
            env_ids (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs_to_reset,).
        """
        pass

    def reset_init_pose_fn(self, init_states: list[DictEnvState], env_ids: torch.Tensor) -> list[DictEnvState]:
        """Reset the initial pose of the environment.

        Args:
            init_states (list[EnvState]): States of the environment
            env_ids (torch.Tensor): The indices of the environments to reset.

        Returns:
            reset_state: The updated states of the environment after resetting.
        """
        if self.reset_dof_pos_noise == 0.0 and self.reset_position_noise == 0.0:
            # If no noise is applied, return the initial states directly
            return deepcopy(init_states)
        if isinstance(init_states, list):
            if getattr(self, "robot_dof_default_pos_cpu", None) is None:
                self.robot_dof_default_pos_cpu = {}
                for robot in self.robots:
                    self.robot_dof_default_pos_cpu[robot.name] = torch.zeros(
                        robot.num_joints, dtype=torch.float32, device="cpu"
                    )
                    for idx, name in enumerate(robot.dof_names):
                        self.robot_dof_default_pos_cpu[robot.name][idx] = self.init_states["robots"][robot.name][
                            "dof_pos"
                        ][name]
            reset_state = deepcopy(init_states)
            num_dofs = self.robots[0].num_joints + self.robots[1].num_joints
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_dofs + 5), device="cpu")

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)
            start_idx = 5
            for i, env_id in enumerate(env_ids):
                for obj_name in reset_state[env_id]["objects"].keys():
                    reset_state[env_id]["objects"][obj_name]["pos"][:3] += (
                        self.reset_position_noise * rand_floats[i, :3]
                    )
                    reset_state[env_id]["objects"][obj_name]["rot"] = new_object_rot[i]

                # reset hand
                if self.reset_dof_pos_noise > 0.0:
                    for robot in self.robots:
                        robot_dof_default_pos = self.robot_dof_default_pos_cpu[robot.name]
                        delta_max = robot.joint_limits_upper.cpu() - robot_dof_default_pos
                        delta_min = robot.joint_limits_lower.cpu() - robot_dof_default_pos
                        rand_delta = (
                            delta_min + (delta_max - delta_min) * rand_floats[i, start_idx : start_idx + robot.num_joints]
                        )
                        dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                        reset_state[env_id]["robots"][robot.name]["dof_pos"] = {
                            name: dof_pos[idx].item() for idx, name in enumerate(robot.dof_names)
                        }
                        start_idx += robot.num_joints

            return reset_state
        elif isinstance(init_states, TensorState):
            reset_state = deepcopy(init_states)  # in sorted order
            num_dofs = self.robots[0].num_joints + self.robots[1].num_joints
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_dofs + 5), device=self.device)

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)
            for obj_id, obj in enumerate(self.objects):
                root_state = reset_state.objects[obj.name].root_state
                root_state[env_ids, :3] += self.reset_position_noise * rand_floats[:, :3]
                obj_state = ObjectState(
                    root_state=root_state,
                )
                if isinstance(obj, ArticulationObjCfg):
                    joint_pos = reset_state.objects[obj.name].joint_pos
                    obj_state.joint_pos = joint_pos
                    obj_state.joint_vel = torch.zeros_like(joint_pos)
                reset_state.objects[obj.name] = obj_state

            if self.reset_dof_pos_noise > 0.0:
                start_idx = 5
                for robot_id, robot in enumerate(self.robots):
                    robot_dof_default_pos = reset_state.robots[robot.name].joint_pos[env_ids]
                    delta_max = robot.joint_limits_upper - robot_dof_default_pos
                    delta_min = robot.joint_limits_lower - robot_dof_default_pos
                    rand_delta = (
                        delta_min + (delta_max - delta_min) * rand_floats[:, start_idx : start_idx + robot.num_joints]
                    )
                    dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                    joint_pos = reset_state.robots[robot.name].joint_pos
                    joint_pos[env_ids.unsqueeze(1), robot.hand_dof_idx.unsqueeze(0)] = dof_pos[:, robot.hand_dof_idx]
                    robot_state = RobotState(
                        root_state=reset_state.robots[robot.name].root_state,
                        joint_pos=joint_pos,
                        joint_vel=torch.zeros_like(joint_pos),
                        body_names=None,
                        body_state=None,
                        joint_force=None,
                        joint_effort_target=None,
                        joint_pos_target=None,
                        joint_vel_target=None,
                    )
                    reset_state.robots[robot.name] = robot_state
                    start_idx += robot.num_joints

            return reset_state
        else:
            raise Exception("Unsupported state type, must be EnvState or TensorState")

    def update_state(self, envstates: TensorState):
        """Update the observation of the environment."""
        for robot in self.robots:
            robot.update_state(envstates)


@torch.jit.script
def compute_task_reward(
    reset_buf,
    reset_goal_buf,
    episode_length_buf,
    success_buf,
    max_episode_length: float,
    door_right_handle_pos,
    door_left_handle_pos,
    door_dof_pos,
    right_hand_reward,
    left_hand_reward,
    right_hand_dist,
    left_hand_dist,
    action_penalty_scale: float,
    actions,
    reach_goal_bonus: float,
):
    """Compute the reward of all environment.

    Args:
        reset_buf (tensor): The reset buffer of all environments at this time, shape (num_envs,)

        reset_goal_buf (tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        episode_length_buf (tensor): The porgress buffer of all environments at this time, shape (num_envs,)

        success_buf (tensor): The success buffer of all environments at this time, shape (num_envs,)

        max_episode_length (float): The max episode length in this environment

        door_right_handle_pos (tensor): The position of the right handle of the cup

        door_left_handle_pos (tensor): The position of the left handle of the cup

        door_dof_pos (tensor): The position of the door dof, shape (num_envs,)

        right_hand_reward (tensor): The reward from the right hand, shape (num_envs,)

        left_hand_reward (tensor): The reward from the left hand, shape (num_envs,)

        right_hand_dist (tensor): The distance from the right hand to the right handle, shape (num_envs,)

        left_hand_dist (tensor): The distance from the left hand to the left handle, shape (num_envs,)

        action_penalty_scale (float): The scale of the action penalty

        actions (tensor): The action buffer of all environments at this time

        reach_goal_bonus (float): The reward given when the object reaches the goal

    """
    action_penalty = torch.sum(actions**2, dim=-1)
    dof_rew = torch.sum(door_dof_pos, dim=-1) * 5  # encourage opening the door
    up_rew = torch.zeros_like(right_hand_reward)
    up_rew = torch.where(
        right_hand_dist < 0.1,
        torch.where(
            left_hand_dist < 0.1,
            torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 10 + dof_rew,
            up_rew,
        ),
        up_rew,
    )

    reward = left_hand_reward + right_hand_reward + up_rew

    success = torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) > 0.7

    # Find out which envs hit the goal and update successes count
    success_buf = torch.where(
        success_buf == 0,
        torch.where(
            success,
            torch.ones_like(success_buf),
            success_buf,
        ),
        success_buf,
    )

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(success == 1, reward + reach_goal_bonus, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(right_hand_dist >= 0.3, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(left_hand_dist >= 0.3, torch.ones_like(resets), resets)

    # Reset because of terminate or fall or success
    resets = torch.where(episode_length_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(success_buf >= 1, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    return reward, resets, goal_resets, success_buf
