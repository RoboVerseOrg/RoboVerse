"""Simple reach task: reach a target point and reset to a new random position when reached."""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat


class SimpleReachTask(RLTaskEnv):
    """Simple reach task: reach a target point.
    
    When reaching within 0.1m threshold, agent gets +50 reward and target resets to a new random position.
    """

    DEFAULT_CONFIG = {
        "action_scale": 0.04,
        "reward_config": {
            "scales": {
                "approach": 2.0,
                "reach_bonus": 50.0,
            }
        },
        "reach_threshold": 0.10,
        "randomization": {
            "target_pos_range": {
                "x": [-1.0, 1.0],
                "y": [-1.0, 1.0],
                "z": [0.6, 1.2],
            },
            "robot_pos_noise": 0.0,
            "joint_noise_range": 0.05,
        },
    }

    scenario = ScenarioCfg(
        objects=[
            # Single target marker
            RigidObjCfg(
                name="target_marker",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 200

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None
        self._action_scale = self.DEFAULT_CONFIG.get("action_scale", 0.04)
        self.num_envs = scenario.num_envs

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self.reach_threshold = self.DEFAULT_CONFIG["reach_threshold"]
        self.w_approach = self.DEFAULT_CONFIG["reward_config"]["scales"]["approach"]
        self.reach_bonus = self.DEFAULT_CONFIG["reward_config"]["scales"]["reach_bonus"]

        # Target position (will be randomized)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self._device)

        # Track if target was reached in current step (to avoid multiple rewards)
        self.target_reached_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)

        super().__init__(scenario, device)

        self.reward_functions = [
            self._reward_approach,
        ]
        self.reward_weights = [
            1.0,
        ]

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges."""
        from copy import deepcopy

        states = deepcopy(states)

        rand_config = self.DEFAULT_CONFIG["randomization"]

        # Randomize target position
        pos_range = rand_config["target_pos_range"]
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos[:, 0] = torch.rand(self.num_envs, device=self.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
        target_pos[:, 1] = torch.rand(self.num_envs, device=self.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
        target_pos[:, 2] = torch.rand(self.num_envs, device=self.device) * (pos_range["z"][1] - pos_range["z"][0]) + pos_range["z"][0]

        self.target_pos = target_pos

        marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        states.objects["target_marker"].root_state = torch.cat(
            [target_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
        )

        # Randomize robot position
        robot_pos = states.robots[self.robot_name].root_state[:, 0:3].clone()
        robot_pos_noise_val = rand_config["robot_pos_noise"]
        robot_pos_noise = (torch.rand(self.num_envs, 3, device=self.device) - 0.5) * robot_pos_noise_val
        robot_pos_new = robot_pos + robot_pos_noise
        robot_quat = states.robots[self.robot_name].root_state[:, 3:7].clone()
        robot_vel = states.robots[self.robot_name].root_state[:, 7:].clone()
        states.robots[self.robot_name].root_state = torch.cat([robot_pos_new, robot_quat, robot_vel], dim=-1)

        # Randomize robot joint positions
        robot_joint_pos = states.robots[self.robot_name].joint_pos.clone()
        joint_noise_range = rand_config["joint_noise_range"]
        joint_noise = (torch.rand_like(robot_joint_pos, device=self.device) - 0.5) * 2 * joint_noise_range
        robot_joint_pos_new = robot_joint_pos + joint_noise
        robot_joint_pos_new[:, 0] = torch.clamp(robot_joint_pos_new[:, 0], 0.0, 0.04)
        robot_joint_pos_new[:, 1] = torch.clamp(robot_joint_pos_new[:, 1], 0.0, 0.04)
        robot_joint_pos_new[:, 2:] = torch.clamp(robot_joint_pos_new[:, 2:], -2.8973, 2.8973)
        states.robots[self.robot_name].joint_pos = robot_joint_pos_new

        return states

    def reset(self, env_ids=None):
        """Reset environment and randomize target position."""
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
            env_ids_list = list(range(self.num_envs))
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )
            env_ids_list = env_ids if isinstance(env_ids, list) else list(env_ids)

        # Reset target reached flag
        self.target_reached_this_step[env_ids_tensor] = False

        # Randomize target position for reset environments
        rand_config = self.DEFAULT_CONFIG["randomization"]
        pos_range = rand_config["target_pos_range"]
        
        new_target_pos = torch.zeros(len(env_ids_list), 3, device=self.device)
        new_target_pos[:, 0] = torch.rand(len(env_ids_list), device=self.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
        new_target_pos[:, 1] = torch.rand(len(env_ids_list), device=self.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
        new_target_pos[:, 2] = torch.rand(len(env_ids_list), device=self.device) * (pos_range["z"][1] - pos_range["z"][0]) + pos_range["z"][0]
        
        self.target_pos[env_ids_tensor] = new_target_pos

        # Update marker position
        states = self.handler.get_states()
        marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(len(env_ids_list), -1)
        zero_vel = torch.zeros(len(env_ids_list), 3, device=self.device)
        zero_ang_vel = torch.zeros(len(env_ids_list), 3, device=self.device)
        
        states.objects["target_marker"].root_state[env_ids_tensor] = torch.cat(
            [new_target_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
        )

        obs, info = super().reset(env_ids=env_ids)

        return obs, info

    def step(self, actions):
        """Step with delta control."""
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)

        # Keep gripper open (no grasping needed)
        real_actions[:, 0] = 0.04
        real_actions[:, 1] = 0.04

        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions

        # Check if target is reached and reset to new position
        updated_states = self.handler.get_states(mode="tensor")
        ee_pos, _ = self._get_ee_state(updated_states)
        
        distance_to_target = torch.norm(ee_pos - self.target_pos, dim=-1)
        reached = (distance_to_target < self.reach_threshold) & (~self.target_reached_this_step)
        
        if reached.any():
            # Add bonus reward for reaching
            reward[reached] += self.reach_bonus
            
            # Mark as reached
            self.target_reached_this_step[reached] = True
            
            # Reset target to new random position
            rand_config = self.DEFAULT_CONFIG["randomization"]
            pos_range = rand_config["target_pos_range"]
            
            reached_env_ids = torch.where(reached)[0]
            new_target_pos = torch.zeros(len(reached_env_ids), 3, device=self.device)
            new_target_pos[:, 0] = torch.rand(len(reached_env_ids), device=self.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
            new_target_pos[:, 1] = torch.rand(len(reached_env_ids), device=self.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
            new_target_pos[:, 2] = torch.rand(len(reached_env_ids), device=self.device) * (pos_range["z"][1] - pos_range["z"][0]) + pos_range["z"][0]
            
            self.target_pos[reached] = new_target_pos
            
            # Update marker position
            marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(len(reached_env_ids), -1)
            zero_vel = torch.zeros(len(reached_env_ids), 3, device=self.device)
            zero_ang_vel = torch.zeros(len(reached_env_ids), 3, device=self.device)
            
            updated_states.objects["target_marker"].root_state[reached] = torch.cat(
                [new_target_pos, marker_quat, zero_vel, zero_ang_vel], dim=-1
            )
            
            if reached[0]:
                log.info(f"[Env 0] Target reached! Distance: {distance_to_target[0].item():.4f}m < {self.reach_threshold}m")
                log.info(f"[Env 0] New target position: {new_target_pos[0].cpu().numpy()}")

        return obs, reward, terminated, time_out, info

    def _reward_approach(self, env_states) -> torch.Tensor:
        """Reward for approaching the target."""
        ee_pos, _ = self._get_ee_state(env_states)
        distance = torch.norm(ee_pos - self.target_pos, dim=-1)
        
        # Approach reward: closer is better
        approach_reward = (1 - torch.tanh(distance)) * self.w_approach
        
        return approach_reward

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation."""
        gripper_pos, gripper_quat = self._get_ee_state(env_states)
        gripper_mat = matrix_from_quat(gripper_quat).view(self.num_envs, -1)
        
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel

        # Ensure gripper_mat has correct shape [num_envs, 9]
        if gripper_mat.dim() == 3:
            gripper_mat = gripper_mat.view(self.num_envs, -1)

        target_to_gripper = self.target_pos - gripper_pos
        distance_to_target = torch.norm(target_to_gripper, dim=-1, keepdim=True)

        obs_list = [
            robot_joint_pos,
            robot_joint_vel,
            gripper_pos,
            gripper_mat[:, 3:],
            self.target_pos,
            target_to_gripper,
            distance_to_target,
        ]

        obs = torch.cat(obs_list, dim=-1)

        return obs

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        rand_config = self.DEFAULT_CONFIG["randomization"]
        pos_range = rand_config["target_pos_range"]
        
        init = []
        for _ in range(self.num_envs):
            # Random target position
            target_x = torch.rand(1).item() * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
            target_y = torch.rand(1).item() * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
            target_z = torch.rand(1).item() * (pos_range["z"][1] - pos_range["z"][0]) + pos_range["z"][0]
            
            init.append({
                "objects": {
                    "target_marker": {
                        "pos": torch.tensor([target_x, target_y, target_z]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([-0.025, -0.160, 0.018054]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                        "dof_pos": {
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                            "panda_joint1": 0.0,
                            "panda_joint2": -0.785398,
                            "panda_joint3": 0.0,
                            "panda_joint4": -2.356194,
                            "panda_joint5": 0.0,
                            "panda_joint6": 1.570796,
                            "panda_joint7": 0.785398,
                        },
                    },
                },
            })

        return init

    def _get_ee_state(self, states):
        """Return EE state using site queries.

        Returns:
            ee_pos_world: (B, 3) gripper position from site
            ee_quat_world: (B, 4) gripper rotation quaternion from site
        """
        robot_config = self.robot
        rs = states.robots[robot_config.name]
        device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

        body_state = (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state, device=device).float()
        )

        # Use panda_hand directly for more accurate EE position
        hand_body_index = rs.body_names.index("panda_hand")
        hand_pos = body_state[:, hand_body_index, 0:3]  # (B, 3)
        hand_quat = body_state[:, hand_body_index, 3:7]  # (B, 4) wxyz

        # Add offset from panda_hand to actual gripper center
        from metasim.utils.math import quat_apply

        offset_local = torch.tensor([0.0, 0.0, 0.1034], device=device, dtype=hand_pos.dtype)
        offset_world = quat_apply(hand_quat, offset_local.expand(hand_pos.shape[0], -1))

        ee_pos_world = hand_pos + offset_world
        ee_quat_world = hand_quat

        return ee_pos_world, ee_quat_world


@register_task("pick_place.reach", "pick_place_reach")
class PickPlaceReach(SimpleReachTask):
    """Simple reach task: reach a target point and reset to a new random position when reached."""

    pass
