"""Stage 1: Simple Approach and Grasp task with hand control.

This task focuses on learning to approach the object with relative pose alignment, grasp it with hand.
Uses saved relative pose for alignment and saved finger positions for grasping.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.utils.math import matrix_from_quat, quat_inv, quat_mul
from roboverse_pack.tasks.pick_place.hand_trajectory import TrajectoryTrackingTaskBase


@register_task("track_grasp_hand_relative", "trackgrasphandrelative")
class TrackGraspHandRelative(TrajectoryTrackingTaskBase):
    """Trajectory tracking task that assumes the object is already grasped.

    The hand should maintain the saved relative pose to the object while following the
    predefined waypoint trajectory. The hand always uses the saved grasp configuration.
    """

    GRASP_DISTANCE_THRESHOLD = 0.04  # Distance threshold for grasp check and hand closing
    GRASP_HISTORY_WINDOW = 10  # Number of frames to check for stable grasp
    APPROACH_DISTANCE_THRESHOLD = 0.05  # Distance threshold to start using saved finger positions
    HAND_MARKER_NAME = "hand_debug_marker"

    # Saved relative pose from hand_info (saved_poses_20251126_101518.py)
    HAND_WHALE_RELATIVE_POS = torch.tensor([-0.072342, 0.036182, 0.000905])  # in object frame
    HAND_WHALE_RELATIVE_ROT = torch.tensor([0.476623, 0.015562, -0.538529, 0.694676])  # hand_quat * inv(whale_quat)

    # Saved finger positions from vega robot (L hand fingers) - saved_poses_20251126_101518.py
    SAVED_FINGER_POSITIONS = {
        "L_ff_j1": -0.551686,
        "L_ff_j2": -0.621416,
        "L_lf_j1": 0.023751,
        "L_lf_j2": 0.027291,
        "L_mf_j1": -0.551840,
        "L_mf_j2": -0.625254,
        "L_rf_j1": 0.014122,
        "L_rf_j2": 0.015942,
        "L_th_j0": 0.693886,
        "L_th_j1": 0.149071,
        "L_th_j2": 0.201562,
    }

    DEFAULT_CONFIG_SIMPLE = deepcopy(TrajectoryTrackingTaskBase.DEFAULT_CONFIG)
    DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"].update({
        "hand_approach": 0.5,
        "grasp_reward": 20.0,
    })
    DEFAULT_CONFIG_SIMPLE["grasp_config"] = {
        "grasp_check_distance": GRASP_DISTANCE_THRESHOLD,
        "hand_close_distance": GRASP_DISTANCE_THRESHOLD,
    }

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="object",
                scale=(0.3, 0.3, 0.3),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_pack/whale_doll/whale_doll.usd",
                urdf_path="roboverse_pack/whale_doll/whale_doll.urdf",
                default_position=(0.104252, -0.076198, 0.846706),
                default_orientation=(0.454115, 0.132146, 0.231502, -0.850132),
            ),
            PrimitiveCubeCfg(
                name="wall",
                size=(0.8, 0.1, 0.3),
                mass=1000.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                enabled_gravity=False,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
            ),
            # Visualization: Trajectory waypoints (5 spheres showing trajectory path)
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_1",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_2",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_3",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_4",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name=HAND_MARKER_NAME,
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
        ],
        robots=["vega"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 250

    def __init__(self, scenario, device=None):
        # Placeholders needed during super().__init__ (reset may be called there)
        self.object_grasped = None
        self.stable_grasp = None
        self._grasp_notified = None
        self._distance_history = None
        self._is_close_enough = None
        self.saved_finger_targets = None

        super().__init__(scenario, device)

        # Override reward functions for this task
        self.reward_functions = [
            self._reward_hand_approach,
            self._reward_trajectory_tracking,
        ]
        self.reward_weights = [
            self.DEFAULT_CONFIG_SIMPLE["reward_config"]["scales"]["hand_approach"],
            1.0,
        ]

        # Get config values
        grasp_config = self.DEFAULT_CONFIG_SIMPLE["grasp_config"]
        self.grasp_check_distance = grasp_config["grasp_check_distance"]
        self.hand_close_distance = grasp_config["hand_close_distance"]

        # Initialize tracking buffers
        self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stable_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._distance_history = torch.full(
            (self.num_envs, self.GRASP_HISTORY_WINDOW),
            float("inf"),
            device=self.device,
        )
        self._is_close_enough = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize saved finger positions tensor
        # Flag ensures saved finger targets re-initialized when needed
        self.saved_finger_targets = None

        # Move relative pose tensors to device
        self.HAND_WHALE_RELATIVE_POS = self.HAND_WHALE_RELATIVE_POS.to(device)
        self.HAND_WHALE_RELATIVE_ROT = self.HAND_WHALE_RELATIVE_ROT.to(device)

    def reset(self, env_ids=None):
        """Reset environment and tracking variables."""
        obs, info = super().reset(env_ids=env_ids)

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )

        # Ensure tracking buffers exist with correct shape
        if self.object_grasped is None or self.object_grasped.shape[0] != self.num_envs:
            self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.stable_grasp is None or self.stable_grasp.shape[0] != self.num_envs:
            self.stable_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._is_close_enough is None or self._is_close_enough.shape[0] != self.num_envs:
            self._is_close_enough = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Reset grasp tracking
        self.object_grasped[env_ids_tensor] = True
        self.stable_grasp[env_ids_tensor] = True
        self._is_close_enough[env_ids_tensor] = True
        if self._grasp_notified is None or self._grasp_notified.shape[0] != self.num_envs:
            self._grasp_notified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        else:
            self._grasp_notified[env_ids_tensor] = False
            self._distance_history[env_ids_tensor] = float("inf")

        # Initialize saved finger positions if not already done
        if self.saved_finger_targets is None and hasattr(self, 'left_hand_finger_joint_names'):
            num_fingers = len(self.left_hand_finger_joint_indices)
            self.saved_finger_targets = torch.zeros((self.num_envs, num_fingers), device=self.device)
            for i, joint_name in enumerate(self.left_hand_finger_joint_names):
                if i < num_fingers and joint_name in self.SAVED_FINGER_POSITIONS:
                    self.saved_finger_targets[:, i] = self.SAVED_FINGER_POSITIONS[joint_name]

        return obs, info

    def step(self, actions):
        """Step with delta control and hand control using relative pose alignment."""
        current_states = self.handler.get_states(mode="tensor")
        
        # Apply delta control
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # Compute relative pose error (distance from saved relative position)
        _, rel_pos, _ = self._get_hand_and_relative_pose(current_states)
        relative_pos_error = torch.norm(
            rel_pos - self.HAND_WHALE_RELATIVE_POS.to(rel_pos.device).unsqueeze(0).expand_as(rel_pos),
            dim=-1,
        )

        # Always use saved finger positions (task starts from a grasped state)
        self._is_close_enough = torch.ones_like(self._is_close_enough, dtype=torch.bool, device=self.device)

        # Apply hand control: use saved finger positions when close, otherwise keep open
        real_actions = self._apply_hand_control(real_actions, current_states)

        # Bypass TrajectoryTrackingTaskBase.step to avoid its hand control logic
        obs, reward, terminated, time_out, info = super(TrajectoryTrackingTaskBase, self).step(real_actions)
        self._last_action = real_actions

        # Update grasp state after step
        updated_states = self.handler.get_states(mode="tensor")
        old_grasped = self.object_grasped.clone()
        self.object_grasped = self._compute_grasp_state(updated_states)

        newly_grasped = self.object_grasped & (~old_grasped)
        newly_released = (~self.object_grasped) & old_grasped

        if newly_grasped.any() and newly_grasped[0]:
            log.info(f"[Env 0] Object grasped! Relative pose error: {relative_pos_error[0].item():.4f}m")
            self._grasp_notified[newly_grasped] = True

        if newly_released.any() and newly_released[0]:
            log.info(f"[Env 0] Object released! Relative pose error: {relative_pos_error[0].item():.4f}m")
            self._grasp_notified[newly_released] = False

        # Terminate episode if object is released
        terminated = terminated | newly_released

        info["grasp_success"] = self.object_grasped
        info["stage"] = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)

        return obs, reward, terminated, time_out, info

    def _apply_hand_control(self, actions, states):
        """Apply hand control: use saved finger positions when close, otherwise keep open."""
        if self.saved_finger_targets is None:
            return actions
            
        # Use saved finger positions when close enough, otherwise use open positions
        finger_targets_open = self.hand_open_q.unsqueeze(0).expand(self.num_envs, -1)  # (B, 11)
        finger_targets_close = self.saved_finger_targets  # (B, 11) - saved positions

        finger_targets = torch.where(
            self._is_close_enough.unsqueeze(-1),
            finger_targets_close,
            finger_targets_open,
        )

        # Apply finger targets to actions
        for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
            if i < finger_targets.shape[1]:
                actions[:, joint_idx] = finger_targets[:, i]

        return actions

    def _get_hand_and_relative_pose(self, states):
        """Match object_layout relative pose computation exactly (batched)."""
        rs = states.robots[self.robot_name]
        body_state = rs.body_state
        if body_state is None:
            raise ValueError("Robot body_state is required to compute hand pose.")
        if not torch.is_tensor(body_state):
            body_state = torch.as_tensor(body_state, device=self.device, dtype=torch.float32)
        else:
            body_state = body_state.to(self.device)

        name_to_index = {name: idx for idx, name in enumerate(rs.body_names)}
        if "L_arm_l7" not in name_to_index:
            raise ValueError("Required link 'L_arm_l7' missing in body_names.")
        link_index = name_to_index["L_arm_l7"]

        link_pos = body_state[:, link_index, 0:3]
        link_quat = body_state[:, link_index, 3:7]

        hand_offset_local = torch.tensor(
            [0.25864, -0.035, -0.03513],
            device=link_pos.device,
            dtype=link_pos.dtype,
        )
        hand_offset = (
            hand_offset_local.view(1, 3, 1)
            .repeat(link_pos.shape[0], 1, 1)
        )
        link_rot = matrix_from_quat(link_quat)
        hand_pos = link_pos + torch.bmm(link_rot, hand_offset).squeeze(-1)

        object_state = states.objects["object"].root_state
        box_pos = object_state[:, 0:3]
        box_quat = object_state[:, 3:7]
        diff_world = hand_pos - box_pos
        object_rot = matrix_from_quat(box_quat)
        rel_pos = torch.bmm(object_rot.transpose(-2, -1), diff_world.unsqueeze(-1)).squeeze(-1)
        rel_rot = quat_mul(link_quat, quat_inv(box_quat))
        return hand_pos, rel_pos, rel_rot

    def _compute_grasp_state(self, states):
        """Compute if object is grasped (requires stable frames based on relative pose error)."""
        _, rel_pos, _ = self._get_hand_and_relative_pose(states)
        relative_pos_error = torch.norm(
            rel_pos - self.HAND_WHALE_RELATIVE_POS.to(rel_pos.device).unsqueeze(0).expand_as(rel_pos),
            dim=-1,
        )

        # Update rolling distance history
        if self._distance_history is None or self._distance_history.shape[0] != self.num_envs:
            self._distance_history = torch.full(
                (self.num_envs, self.GRASP_HISTORY_WINDOW),
                float("inf"),
                device=self.device,
            )
        self._distance_history = torch.roll(self._distance_history, shifts=-1, dims=1)
        self._distance_history[:, -1] = relative_pos_error

        # Object is grasped if relative pose error has been stable (close) for GRASP_HISTORY_WINDOW frames
        self.stable_grasp = (self._distance_history < self.grasp_check_distance).all(dim=1)
        is_grasping = self.stable_grasp

        return is_grasping

    def _get_hand_position(self, states):
        """Get palm center using the same logic as object layout scripts."""
        hand_pos, _, _ = self._get_hand_and_relative_pose(states)
        return hand_pos

    def _get_hand_rotation(self, states):
        """Get hand rotation quaternion from L_arm_l7 link."""
        rs = states.robots[self.robot.name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        name_to_index = {name: idx for idx, name in enumerate(rs.body_names)}
        if "L_arm_l7" not in name_to_index:
            raise ValueError("Required link 'L_arm_l7' missing in body_names.")

        link_index = name_to_index["L_arm_l7"]
        link_quat = body_state[:, link_index, 3:7]  # (B, 4) wxyz

        return link_quat

    def _reward_hand_approach(self, env_states) -> torch.Tensor:
        """Reward for hand approaching the box with relative pose alignment."""
        hand_pos, rel_pos, current_relative_rot = self._get_hand_and_relative_pose(env_states)

        # Reward for matching relative position
        relative_pos_error = torch.norm(
            rel_pos - self.HAND_WHALE_RELATIVE_POS.to(rel_pos.device).unsqueeze(0).expand_as(rel_pos),
            dim=-1
        )
        pos_reward = 2 - torch.tanh(relative_pos_error * 10) -torch.tanh(relative_pos_error)

        # Reward for matching relative rotation
        # Use quaternion distance: 1 - |q1 · q2| (dot product)
        relative_rot_target = self.HAND_WHALE_RELATIVE_ROT.unsqueeze(0).expand(self.num_envs, -1)
        quat_dot = torch.abs(torch.sum(current_relative_rot * relative_rot_target, dim=-1))
        rot_reward = quat_dot  # Higher when quaternions are similar (dot product close to 1)

        # Combined reward
        approach_reward = pos_reward * 0.5 + rot_reward * 0.5

  
        return approach_reward

    def _reward_grasp(self, env_states) -> torch.Tensor:
        """Reward only when stable grasp."""
        if self.stable_grasp is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self.stable_grasp.float()

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.104252, -0.076198, 0.846706]),
                        "rot": torch.tensor([0.454115, 0.132146, 0.231502, -0.850132]),
                    },
                    "wall": {
                        "pos": torch.tensor([0.632921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.660000, -0.250000, 0.399868]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    # Trajectory waypoints (world coordinates)
                    "traj_marker_0": {
                        "pos": torch.tensor([0.400000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.400000, -0.320000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.400000, -0.190000, 1.220000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.400000, -0.070000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.400000, 0.000000, 1.080000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, 0.000000]),
                    },
                    self.HAND_MARKER_NAME: {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.353636, -0.230209, 0.000511]),
                        "rot": torch.tensor([1.000201, 0.000000, -0.000000, -0.000000]),
                        "dof_pos": {
                            # From saved_poses_20251126_101518.py
                            "B_wheel_j1": 3.098256,
                            "B_wheel_j2": 11.410545,
                            "L_arm_j1": 0.132269,
                            "L_arm_j2": -0.005679,
                            "L_arm_j3": -0.039935,
                            "L_arm_j4": -0.007740,
                            "L_arm_j5": 0.792566,
                            "L_arm_j6": -0.068200,
                            "L_arm_j7": 0.003688,
                            "L_ff_j1": -0.551686,
                            "L_ff_j2": -0.621416,
                            "L_lf_j1": 0.023751,
                            "L_lf_j2": 0.027291,
                            "L_mf_j1": -0.551840,
                            "L_mf_j2": -0.625254,
                            "L_rf_j1": 0.014122,
                            "L_rf_j2": 0.015942,
                            "L_th_j0": 0.693886,
                            "L_th_j1": 0.149071,
                            "L_th_j2": 0.201562,
                            "L_wheel_j1": -0.050129,
                            "L_wheel_j2": -2.361024,
                            "R_arm_j1": -0.012039,
                            "R_arm_j2": 0.000147,
                            "R_arm_j3": 0.002277,
                            "R_arm_j4": 0.015811,
                            "R_arm_j5": -0.001019,
                            "R_arm_j6": -0.003106,
                            "R_arm_j7": 0.000569,
                            "R_ff_j1": 0.006052,
                            "R_ff_j2": 0.006850,
                            "R_lf_j1": 0.008481,
                            "R_lf_j2": 0.009723,
                            "R_mf_j1": 0.005918,
                            "R_mf_j2": 0.006720,
                            "R_rf_j1": 0.007117,
                            "R_rf_j2": 0.008026,
                            "R_th_j0": 0.264549,
                            "R_th_j1": 0.078753,
                            "R_th_j2": 0.106434,
                            "R_wheel_j1": 0.081973,
                            "R_wheel_j2": 5.777865,
                            "torso_j1": 0.153865,
                            "torso_j2": 0.016780,
                            "torso_j3": -0.021553,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init
