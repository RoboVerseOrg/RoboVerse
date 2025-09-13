"""Bring a box to a target and orientation."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat
from roboverse_pack.queries.contact import ContactData
from roboverse_pack.queries.site import SiteMat, SitePos

# Default configuration as a global dictionary
DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            # Gripper goes to the box and closes when close enough.
            "gripper_box": 4.0,
            # Box goes to the target mocap.
            "box_target": 8.0,
            # Do not collide the gripper with the floor.
            "no_floor_collision": 0.25,
            # Arm stays close to target pose.
            "robot_target_qpos": 0.3,
        }
    },
    "nconmax": 24 * 8192,
    "njmax": 128,
}


@register_task("mujoco_playground.pick", "pick_rl")
class PandaPickCube(RLTaskEnv):
    """Bring a box to a target.

    Reward shaping and task design adapted from DeepMind's Mujoco Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="box",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveSphereCfg(
                name="target",
                mass=0.01,
                radius=0.02,
                physics=PhysicStateType.XFORM,
                color=(0.7, 1.0, 0.7),
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
            nconmax=24 * 8192,
            njmax=128,
        ),
        decimation=4,
    )
    max_episode_steps = 150

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None  # Initialize _last_action to None
        super().__init__(scenario, device)
        self._action_scale = DEFAULT_CONFIG.get("action_scale", 0.04)
        self.reward_functions = [
            self._reward_box_target,
            self._reward_gripper_box,
            self._reward_no_floor_collision,
            self._reward_robot_target_qpos,
        ]
        self.reward_weights = [
            DEFAULT_CONFIG["reward_config"]["scales"]["box_target"],
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_box"],
            DEFAULT_CONFIG["reward_config"]["scales"]["no_floor_collision"],
            DEFAULT_CONFIG["reward_config"]["scales"]["robot_target_qpos"],
        ]

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges."""
        # Define randomization ranges
        box_pos_range = torch.tensor([[0.4, -0.2, 0.0], [0.8, 0.2, 0.0]], device=self.device)
        target_pos_range = torch.tensor([[0.4, -0.2, 0.2], [0.8, 0.2, 0.4]], device=self.device)

        # Generate random positions within ranges
        box_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (box_pos_range[1] - box_pos_range[0]) + box_pos_range[0]
        )
        target_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (target_pos_range[1] - target_pos_range[0])
            + target_pos_range[0]
        )

        # Keep original quaternions
        box_quat = states.objects["box"].root_state[:, 3:7]
        target_quat = states.objects["target"].root_state[:, 3:7]

        # Set velocities to zero
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Update states
        states.objects["box"].root_state = torch.cat([box_pos, box_quat, zero_vel, zero_ang_vel], dim=-1)
        states.objects["target"].root_state = torch.cat([target_pos, target_quat, zero_vel, zero_ang_vel], dim=-1)

        return states

    def reset(self, env_ids=None):
        """Reset environment and last actions."""
        # Initialize _last_action if not exists (first reset)
        # Reset last action for delta control
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]
        return super().reset(env_ids=env_ids)

    def step(self, actions):
        """Step with delta control."""
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)
        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions
        return obs, reward, terminated, time_out, info

    def _reward_box_target(self, env_states) -> torch.Tensor:
        """Reward for bringing box to target position and orientation."""
        # Get states from tensor state
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["box"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        target_quat = env_states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

        # Get finger joint positions (first two joints are finger joints)

        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)
        pos_err = torch.norm(target_pos - box_pos, dim=-1)  # [num_envs]
        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        target_mat = matrix_from_quat(target_quat)  # [num_envs, 3, 3]

        rot_err = torch.norm(
            target_mat.view(self.num_envs, -1)[:, :6] - box_mat.view(self.num_envs, -1)[:, :6], dim=-1
        )  # [num_envs]
        box_target = 1 - torch.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        reached_box = (torch.norm(box_pos - gripper_pos, dim=-1) < 0.012).float()

        return box_target * reached_box

    def _reward_gripper_box(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the box and closing when close enough."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]

        gripper_pos = env_states.extras["gripper_pos"]  # [num_envs, 3]
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)  # [num_envs]

        # print("gripper_pos[0]:", gripper_pos[0], "gripper_box_dist[0]:", gripper_box_dist[0])

        # Basic approach reward
        approach_reward = 1 - torch.tanh(5 * gripper_box_dist)

        # Get gripper joint positions (finger joints)
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        finger1_pos = robot_joint_pos[:, 0]  # panda_finger_joint1
        finger2_pos = robot_joint_pos[:, 1]  # panda_finger_joint2

        # Calculate gripper closure (0 = fully open, 0.08 = fully closed)
        gripper_closure = 0.08 - finger1_pos - finger2_pos  # [num_envs]

        # Distance-based gripper behavior reward
        # When far: encourage open (low closure), when close: encourage close (high closure)
        distance_threshold = 0.01  # [m]
        far_from_box = (gripper_box_dist > distance_threshold).float()  # [num_envs]
        close_to_box = (gripper_box_dist <= distance_threshold).float()  # [num_envs]

        # When far: reward each finger to be around 0.01 (slightly open)
        finger1_target = 0.03
        finger2_target = 0.03
        finger1_reward = 1.0 - torch.abs(finger1_pos - finger1_target) / 0.03  # [num_envs]
        finger2_reward = 1.0 - torch.abs(finger2_pos - finger2_target) / 0.03  # [num_envs]
        far_finger_reward = 0.5 * (finger1_reward + finger2_reward) / 2.0  # [num_envs]

        # When close: reward for closing (high closure)
        close_reward = gripper_closure / 0.08  # [num_envs]

        # Combine rewards based on distance
        gripper_behavior_reward = (
            far_from_box * far_finger_reward  # Far: reward each finger around 0.01
            + close_to_box * close_reward  # Close: reward closed
        )  # [num_envs]

        return approach_reward + gripper_behavior_reward

    def _reward_no_floor_collision(self, env_states) -> torch.Tensor:
        """Reward for not colliding with the floor."""
        # Get individual floor collision sensor data from extra observations
        extra_data = env_states.extras
        left_collision = extra_data["left_finger_floor_collision"]  # [num_envs]
        right_collision = extra_data["right_finger_floor_collision"]  # [num_envs]
        hand_collision = extra_data["hand_capsule_floor_collision"]  # [num_envs]

        # Check if any of the hand parts are colliding with floor
        # Each sensor returns 1 if collision, 0 if no collision
        any_collision = (left_collision > 0) | (right_collision > 0) | (hand_collision > 0)  # [num_envs]

        # Reward for no collision (1 if no collision, 0 if collision)
        no_floor_collision = (~any_collision).float()

        return no_floor_collision

    def _reward_robot_target_qpos(self, env_states) -> torch.Tensor:
        """Reward for robot staying close to target joint positions."""
        # [num_envs, ARM_joints],first two are finger joints
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos[:, 2:]
        target_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[:, 2:]

        joint_error = torch.norm(robot_joint_pos - target_joint_pos, dim=-1)  # [num_envs]
        return 1 - torch.tanh(joint_error)

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["box"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        target_quat = env_states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)
        gripper_mat = env_states.extras["gripper_mat"]  # (B, 9)
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert quaternions to rotation matrices for box and target
        from metasim.utils.math import matrix_from_quat

        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        target_mat = matrix_from_quat(target_quat)  # [num_envs, 3, 3]

        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]
        target_mat_flat = target_mat.view(self.num_envs, -1)  # [num_envs, 9]

        # Ensure gripper_mat has correct shape [num_envs, 9]
        if gripper_mat.dim() == 3:
            gripper_mat = gripper_mat.view(self.num_envs, -1)  # Reshape to [num_envs, 9]

        box_to_gripper = box_pos - gripper_pos  # [num_envs, 3]
        target_to_box = target_pos - box_pos  # [num_envs, 3]
        target_box_rot_diff = target_mat_flat[:, :6] - box_mat_flat[:, :6]  # [num_envs, 6]

        # Get joint position targets and calculate control error (target - actual)
        robot_joint_pos_target = env_states.robots[self.robot_name].joint_pos_target  # [num_envs, num_joints]
        joint_pos_error = robot_joint_pos_target - robot_joint_pos  # [num_envs, num_joints]

        obs = torch.cat(
            [
                robot_joint_pos,  # Joint positions [num_envs, num_joints]
                robot_joint_vel,  # Joint velocities [num_envs, num_joints]
                gripper_pos,  # Robot position [num_envs, 3]
                gripper_mat[:, 3:],  # Robot orientation (last 6 elements) [num_envs, 6]
                box_mat_flat[:, 3:],  # Box orientation (last 6 elements) [num_envs, 6]
                box_to_gripper,  # Box to gripper vector [num_envs, 3]
                target_to_box,  # Target to box vector [num_envs, 3]
                target_box_rot_diff,  # Target-box orientation difference [num_envs, 6]
                joint_pos_error,  # Joint position error (target - actual) [num_envs, num_joints]
            ],
            dim=-1,
        )  # [num_envs, obs_dim]

        return obs

    def _terminated(self, env_states) -> torch.Tensor:
        """Check if episode should be terminated."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        out_of_bounds = torch.any(torch.abs(box_pos) > 1.0, dim=-1)  # [num_envs]
        out_of_bounds |= box_pos[:, 2] < 0.0  # Box below ground

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]
        nan_joints = torch.isnan(robot_joint_pos).any(dim=-1) | torch.isnan(robot_joint_vel).any(dim=-1)

        nan_objects = torch.isnan(box_pos).any(dim=-1)
        terminated = out_of_bounds | nan_joints | nan_objects

        return terminated

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "box": {
                        "pos": torch.tensor([0.5, 0.0, 0.03]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "target": {
                        "pos": torch.tensor([0.5, 0.0, 0.03]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": 0.3,
                            "panda_joint3": 0.0,
                            "panda_joint4": -1.57079,
                            "panda_joint5": 0.0,
                            "panda_joint6": 2.0,
                            "panda_joint7": -0.7853,
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                        },
                    }
                },
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]

        return init

    def _extra_spec(self) -> dict:
        return {
            "left_finger_floor_collision": ContactData(f"{self.robot_name}/left_finger_pad", "floor"),
            "right_finger_floor_collision": ContactData(f"{self.robot_name}/right_finger_pad", "floor"),
            "hand_capsule_floor_collision": ContactData(f"{self.robot_name}/hand_capsule", "floor"),
            # Gripper site queries for position and orientation
            "gripper_pos": SitePos("gripper"),
            "gripper_mat": SiteMat("gripper"),
        }

    def _get_ee_state(self, states):
        """Return EE state using site queries.

        Returns:
            ee_pos_world: (B, 3) gripper position from site
            ee_mat_world: (B, 9) gripper rotation matrix from site
        """
        # robot_config = self.robot
        # rs = states.robots[robot_config.name]
        # device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        # body_state = (
        #     rs.body_state
        #     if isinstance(rs.body_state, torch.Tensor)
        #     else torch.tensor(rs.body_state, device=device).float()
        # )
        # joint_pos = (
        #     rs.joint_pos
        #     if isinstance(rs.joint_pos, torch.Tensor)
        #     else torch.tensor(rs.joint_pos, device=device).float()
        # )
        # ee_body_index = rs.body_names.index(robot_config.ee_body_name)
        # # Get finger body indices instead of ee_body
        # finger_body_names = ["panda_left_finger", "panda_right_finger"]
        # finger_body_indices = [rs.body_names.index(name) for name in finger_body_names]

        # # Get positions and orientations from both finger bodies
        # finger1_pos = body_state[:, finger_body_indices[0], 0:3]  # (B,3)
        # finger2_pos = body_state[:, finger_body_indices[1], 0:3]  # (B,3)

        # # Calculate mean position and orientation
        # ee_pos_world = (finger1_pos + finger2_pos) / 2.0  # (B,3)
        # ee_quat_world = body_state[:, ee_body_index, 3:7]  # (B,4) wxyz
        # return ee_pos_world, ee_quat_world
        ee_pos_world = states.extras["gripper_pos"]  # (B, 3)
        ee_mat_world = states.extras["gripper_mat"]  # (B, 9)
        return ee_pos_world, ee_mat_world
