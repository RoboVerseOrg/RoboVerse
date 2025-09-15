"""Handover task converted from MuJoCo Playground to RoboVerse."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat
from roboverse_pack.queries.site import SiteMat, SitePos

# Default configuration adapted from MuJoCo Playground
DEFAULT_CONFIG = {
    "action_scale": 0.015,
    "reward_config": {
        "scales": {
            "gripper_box": 1,
            "box_handover": 4,
            "handover_target": 8,
            "no_table_collision": 0.3,
        }
    },
    "nconmax": 24 * 8192,
    "njmax": 88,
}


def logistic_barrier(x: torch.Tensor, x0=0, k=100, direction=1.0):
    """Logistic barrier function for reward shaping."""
    # direction = 1.0: Penalize going to the left.
    return 1 / (1 + torch.exp(-k * direction * (x - x0)))


@register_task("mujoco_playground.handover", "handover_rl")
class HandOver(RLTaskEnv):
    """Dual-arm handover task for ALOHA robot.

    Reward shaping and task design adapted from DeepMind's MuJoCo Playground
    (Apache 2.0 License), re-implemented for RoboVerse.
    """

    scenario = ScenarioCfg(
        objects=[
            # Box to be handed over
            PrimitiveCubeCfg(
                name="box",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),  # Red color for box
            ),
            # Target position for the box
            PrimitiveSphereCfg(
                name="target",
                mass=0.01,
                radius=0.02,
                physics=PhysicStateType.XFORM,
                color=(0.7, 1.0, 0.7),  # Green color for target
            ),
        ],
        robots=["aloha"],  # Dual-arm ALOHA robot
        sim_params=SimParamCfg(
            dt=0.005,
            nconmax=24 * 8192,
            njmax=88,
        ),
        decimation=4,
    )
    max_episode_steps = 250  # 5 seconds

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None
        super().__init__(scenario, device)
        self._action_scale = DEFAULT_CONFIG.get("action_scale", 0.015)

        # Reward thresholds and handover position
        self._left_thresh = -0.1
        self._right_thresh = 0.0
        self._handover_pos = torch.tensor([0.0, 0.0, 0.24], device=self.device)

        # Episode tracking
        self._episode_picked = torch.zeros(self.scenario.num_envs, device=device, dtype=torch.bool)
        self._prev_potential = torch.zeros(self.scenario.num_envs, device=device)

        self.reward_functions = [
            self._reward_gripper_box,
            self._reward_box_handover,
            self._reward_handover_target,
            self._reward_no_table_collision,
        ]
        self.reward_weights = [
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_box"],
            DEFAULT_CONFIG["reward_config"]["scales"]["box_handover"],
            DEFAULT_CONFIG["reward_config"]["scales"]["handover_target"],
            DEFAULT_CONFIG["reward_config"]["scales"]["no_table_collision"],
        ]

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing box and target positions."""
        # mj: box_xy = jp.array([
        # mj:     jax.random.uniform(rng_box_x, (), minval=-0.05, maxval=0.05),
        # mj:     jax.random.uniform(rng_box_y, (), minval=-0.1, maxval=0.1),
        # mj: ])
        box_x_range = torch.tensor([-0.05, 0.05], device=self.device)
        box_y_range = torch.tensor([-0.1, 0.1], device=self.device)

        box_x = torch.rand(self.num_envs, 1, device=self.device) * (box_x_range[1] - box_x_range[0]) + box_x_range[0]
        box_y = torch.rand(self.num_envs, 1, device=self.device) * (box_y_range[1] - box_y_range[0]) + box_y_range[0]
        box_z = torch.zeros(self.num_envs, 1, device=self.device)  # Box on table

        box_pos = torch.cat([box_x, box_y, box_z], dim=-1)

        # mj: target_pos = jp.array([0.20, 0.0, 0.25])
        # mj: target_pos += jax.random.uniform(rng_target, (3,), minval=-0.15, maxval=0.15)
        # mj: target_x = jp.clip(target_pos[0], 0.15, None)  # Saturate log barrier.
        target_base = torch.tensor([0.20, 0.0, 0.25], device=self.device)
        target_jitter = torch.rand(self.num_envs, 3, device=self.device) * 0.3 - 0.15  # [-0.15, 0.15]
        target_pos = target_base.unsqueeze(0) + target_jitter
        target_pos[:, 0] = torch.clamp(target_pos[:, 0], min=0.15)  # Saturate log barrier

        # Set orientations (identity quaternions)
        box_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Zero velocities
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Update object states
        states.objects["box"].root_state = torch.cat([box_pos, box_quat, zero_vel, zero_ang_vel], dim=-1)
        states.objects["target"].root_state = torch.cat([target_pos, target_quat, zero_vel, zero_ang_vel], dim=-1)

        return states

    def reset(self, env_ids=None):
        """Reset environment and episode tracking."""
        if env_ids is None or self._last_action is None:
            # Initialize with home position
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos.clone()
            # Reset episode tracking for all envs
            self._episode_picked.zero_()
            self._prev_potential.zero_()
        else:
            # Reset episode tracking for selected envs
            self._episode_picked[env_ids] = False
            self._prev_potential[env_ids] = 0.0
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids].clone()

        return super().reset(env_ids=env_ids)

    def step(self, actions):
        """Step with delta control and progress tracking."""
        # mj: delta = action * self._config.action_scale
        # mj: ctrl = state.data.ctrl + delta
        # mj: ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)

        obs, _, terminated, time_out, info = super().step(real_actions)

        # Update episode tracking
        box_pos = self.env_states.objects["box"].root_state[:, 0:3]
        picked = box_pos[:, 2] > 0.15
        self._episode_picked = self._episode_picked | picked

        # Calculate potential-based reward
        raw_rewards = self._calculate_raw_rewards(self.env_states)
        potential = sum(w * r for w, r in zip(self.reward_weights, raw_rewards)) / sum(self.reward_weights)

        # Reward progress (clip at zero to not penalize exploration)
        progress_reward = torch.maximum(potential - self._prev_potential, torch.zeros_like(potential))
        self._prev_potential = torch.maximum(potential, self._prev_potential)

        # Additional shaping for end state
        left_gripper_pos = self.env_states.extras["left_gripper_pos"]
        condition = logistic_barrier(left_gripper_pos[:, 0], direction=-1) * logistic_barrier(box_pos[:, 0], 0.10)
        progress_reward += 0.02 * potential * condition

        # Penalty for dropping after picking
        dropped = (box_pos[:, 2] < 0.05) & self._episode_picked
        progress_reward += dropped.float() * -0.1

        # Update actions and return
        self._last_action = real_actions
        return obs, progress_reward, terminated, time_out, info

    def _calculate_raw_rewards(self, env_states):
        """Calculate raw reward components."""
        return [
            self._reward_gripper_box(env_states),
            self._reward_box_handover(env_states),
            self._reward_handover_target(env_states),
            self._reward_no_table_collision(env_states),
        ]

    def _reward_gripper_box(self, env_states) -> torch.Tensor:
        """Reward for grippers approaching the box."""

        # mj: def distance(x, y): return jp.exp(-10 * jp.linalg.norm(x - y))
        def distance(x, y):
            return torch.exp(-10 * torch.norm(x - y, dim=-1))

        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        left_gripper_pos = env_states.extras["left_gripper_pos"]  # [num_envs, 3]
        right_gripper_pos = env_states.extras["right_gripper_pos"]  # [num_envs, 3]

        # mj: pre = jp.where(box[0] < self._left_thresh, 1.0, 0.0)
        # mj: past = jp.where(box[0] >= self._right_thresh, 1.0, 0.0)
        # mj: btwn = (1 - pre) * (1 - past)
        pre = (box_pos[:, 0] < self._left_thresh).float()
        past = (box_pos[:, 0] >= self._right_thresh).float()
        btwn = (1 - pre) * (1 - past)

        # mj: r_lg = distance(box_top, l_gripper) * (pre + btwn)
        # mj: r_rg = distance(box_bottom, r_gripper) * (btwn + past)
        # mj: r_rg_bias = distance(box_bottom, r_gripper) * past
        r_lg = distance(box_pos, left_gripper_pos) * (pre + btwn)
        r_rg = distance(box_pos, right_gripper_pos) * (btwn + past)
        r_rg_bias = distance(box_pos, right_gripper_pos) * past

        return r_lg + r_rg + r_rg_bias

    def _reward_box_handover(self, env_states) -> torch.Tensor:
        """Reward for bringing box to handover position."""
        # mj: box_handover = distance(box, self._handover_pos)
        # mj: hand_handover = distance(l_gripper, self._handover_pos) * past
        # mj: box_handover = jp.maximum(box_handover, hand_handover)

        def distance(x, y):
            return torch.exp(-10 * torch.norm(x - y, dim=-1))

        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        left_gripper_pos = env_states.extras["left_gripper_pos"]  # [num_envs, 3]

        # Check if box is past right threshold
        past = (box_pos[:, 0] >= self._right_thresh).float()

        box_handover = distance(box_pos, self._handover_pos)
        hand_handover = distance(left_gripper_pos, self._handover_pos) * past
        box_handover = torch.maximum(box_handover, hand_handover)

        return box_handover

    def _reward_handover_target(self, env_states) -> torch.Tensor:
        """Reward for bringing box to target position."""
        # mj: box_target = distance(info["target_pos"], box) * (r_rg + r_rg_bias)
        # mj: box_target *= logistic_barrier(l_gripper[0], direction=-1)

        def distance(x, y):
            return torch.exp(-10 * torch.norm(x - y, dim=-1))

        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        left_gripper_pos = env_states.extras["left_gripper_pos"]  # [num_envs, 3]
        right_gripper_pos = env_states.extras["right_gripper_pos"]  # [num_envs, 3]

        # Calculate gripper-box distances for weighting
        r_rg = distance(box_pos, right_gripper_pos)
        past = (box_pos[:, 0] >= self._right_thresh).float()
        r_rg_bias = distance(box_pos, right_gripper_pos) * past

        box_target = distance(target_pos, box_pos) * (r_rg + r_rg_bias)
        box_target *= logistic_barrier(left_gripper_pos[:, 0], direction=-1)

        return box_target

    def _reward_no_table_collision(self, env_states) -> torch.Tensor:
        """Reward for avoiding table collisions."""
        # mj: table_collision = self.hand_table_collision(data)
        # This would need to be implemented with contact detection
        # For now, return a constant reward (no collision)
        return torch.ones(self.num_envs, device=self.device)

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        # mj: obs = jp.concatenate([
        # mj:     data.qpos, data.qvel, (finger_qposadr - box_width),
        # mj:     box_top, box_bottom, left_gripper_pos, left_gripper_mat.ravel()[3:],
        # mj:     right_gripper_pos, right_gripper_mat.ravel()[3:], box_mat.ravel()[3:],
        # mj:     data.xpos[self._box_body] - info["target_pos"],
        # mj:     (info["_steps"].reshape((1,)) / self._config.episode_length).astype(float),
        # mj: ])

        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["box"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]

        left_gripper_pos = env_states.extras["left_gripper_pos"]  # [num_envs, 3]
        left_gripper_mat = env_states.extras["left_gripper_mat"]  # [num_envs, 9]
        right_gripper_pos = env_states.extras["right_gripper_pos"]  # [num_envs, 3]
        right_gripper_mat = env_states.extras["right_gripper_mat"]  # [num_envs, 9]

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert box quaternion to rotation matrix
        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]

        # Ensure gripper matrices have correct shape
        if left_gripper_mat.dim() == 3:
            left_gripper_mat = left_gripper_mat.view(self.num_envs, -1)
        if right_gripper_mat.dim() == 3:
            right_gripper_mat = right_gripper_mat.view(self.num_envs, -1)

        # Finger joint positions (assuming finger joints are the last 4)
        finger_joints = robot_joint_pos[:, -4:]  # [num_envs, 4]
        box_width = 0.04  # Box width for finger positioning
        finger_box_diff = finger_joints - box_width

        # Target relative to box
        target_to_box = target_pos - box_pos  # [num_envs, 3]

        # Episode progress (simplified - would need step counter)
        episode_progress = torch.zeros(self.num_envs, 1, device=self.device)  # Placeholder

        obs = torch.cat(
            [
                robot_joint_pos,  # Joint positions [num_envs, num_joints]
                robot_joint_vel,  # Joint velocities [num_envs, num_joints]
                finger_box_diff,  # Finger joint positions relative to box width [num_envs, 4]
                box_pos,  # Box position [num_envs, 3]
                box_pos,  # Box bottom (same as box position for simplicity) [num_envs, 3]
                left_gripper_pos,  # Left gripper position [num_envs, 3]
                left_gripper_mat[:, 3:],  # Left gripper orientation (last 6 elements) [num_envs, 6]
                right_gripper_pos,  # Right gripper position [num_envs, 3]
                right_gripper_mat[:, 3:],  # Right gripper orientation (last 6 elements) [num_envs, 6]
                box_mat_flat[:, 3:],  # Box orientation (last 6 elements) [num_envs, 6]
                target_to_box,  # Target relative to box [num_envs, 3]
                episode_progress,  # Episode progress [num_envs, 1]
            ],
            dim=-1,
        )  # [num_envs, obs_dim]

        return obs

    def _terminated(self, env_states) -> torch.Tensor:
        """Check if episode should be terminated."""
        # mj: out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        # mj: out_of_bounds |= box_pos[2] < 0.0
        # mj: done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | dropped

        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        out_of_bounds = torch.any(torch.abs(box_pos) > 1.0, dim=-1)  # [num_envs]
        out_of_bounds |= box_pos[:, 2] < 0.0  # Box below ground

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]
        nan_joints = torch.isnan(robot_joint_pos).any(dim=-1) | torch.isnan(robot_joint_vel).any(dim=-1)

        nan_objects = torch.isnan(box_pos).any(dim=-1)

        # Check for dropping after picking
        dropped = (box_pos[:, 2] < 0.05) & self._episode_picked

        terminated = out_of_bounds | nan_joints | nan_objects | dropped

        return terminated

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "box": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),  # Will be randomized in _prepare_states
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "target": {
                        "pos": torch.tensor([0.2, 0.0, 0.25]),  # Will be randomized in _prepare_states
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "aloha": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            # Left arm joints
                            "left/waist": 0.0,
                            "left/shoulder": 0.0,
                            "left/elbow": 0.0,
                            "left/forearm_roll": 0.0,
                            "left/wrist_angle": 0.0,
                            "left/wrist_rotate": 0.0,
                            "left/left_finger": 0.0,
                            "left/right_finger": 0.0,
                            # Right arm joints
                            "right/waist": 0.0,
                            "right/shoulder": 0.0,
                            "right/elbow": 0.0,
                            "right/forearm_roll": 0.0,
                            "right/wrist_angle": 0.0,
                            "right/wrist_rotate": 0.0,
                            "right/left_finger": 0.0,
                            "right/right_finger": 0.0,
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
        """Register extra queries for gripper states."""
        return {
            # Gripper site queries for position and orientation
            "left_gripper_pos": SitePos("left_gripper"),
            "left_gripper_mat": SiteMat("left_gripper"),
            "right_gripper_pos": SitePos("right_gripper"),
            "right_gripper_mat": SiteMat("right_gripper"),
        }
