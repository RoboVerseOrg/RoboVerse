from __future__ import annotations

import gymnasium as gym

ISAACGYM_AVAILABLE = True

import logging
import math
import random
import time

import numpy as np
import torch

from metasim.sim.env_wrapper import GymEnvWrapper
from metasim.types import Obs

log = logging.getLogger(__name__)


class RLEnvWrapper:
    """Wrapper for RL infra, wrapping a GymEnvWrapper."""

    def __init__(self, gym_env: GymEnvWrapper, seed: int | None = None, verbose: bool = False):
        self.env = gym_env
        self.scenario = self.env.handler.scenario

        self._register_configs()
        self._set_up_buffers()

        self.verbose = verbose
        self.step_timings = {}
        self.total_steps = 0
        self.total_time = 0.0

        self.writer = None
        self.global_step = 0
        self.timestep_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if seed is not None:
            self.set_seed(seed)

    def _normalize_quaternion(self, quat):
        """Normalize a quaternion to unit length."""
        w, x, y, z = quat
        magnitude = math.sqrt(w * w + x * x + y * y + z * z)
        if magnitude < 1e-10:  # Avoid division by zero
            return (1.0, 0.0, 0.0, 0.0)
        return (w / magnitude, x / magnitude, y / magnitude, z / magnitude)

    def _get_default_reset_states(self, env_ids: list[int] | None = None):
        """Get default reset states for the environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Get current states to determine the format
        current_states = self.env.handler.get_states()

        # Handle TensorState (IsaacGym, etc.) - these simulators handle reset internally
        if hasattr(current_states, "__class__") and current_states.__class__.__name__ == "TensorState":
            # For tensor-based simulators like IsaacGym, reset is handled differently.
            # The task's reset() method should handle setting robots to default positions.
            # This is because IsaacGym uses GPU tensors and requires special handling.
            # TODO: Consider implementing a unified reset interface for all simulators.
            return None

        # Handle list of states (MuJoCo, SAPIEN, etc.)
        if isinstance(current_states, list):
            reset_states = []
            robot = self._robot

            # For dm_control tasks, return None to let the wrapper handle reset
            if robot is None:
                return None

            for i in range(self.num_envs):
                if i not in env_ids:
                    # Keep current state for environments not being reset
                    reset_states.append(current_states[i])
                else:
                    # Create default state for environments being reset
                    state = {"robots": {}, "objects": {}}

                    # Set robot to default state
                    robot_state = {}

                    # Set default position
                    if hasattr(robot, "default_position"):
                        robot_state["pos"] = list(robot.default_position)
                    else:
                        robot_state["pos"] = [0.0, 0.0, 0.0]

                    # Set default orientation
                    if hasattr(robot, "default_orientation"):
                        robot_state["rot"] = list(robot.default_orientation)
                    else:
                        robot_state["rot"] = [1.0, 0.0, 0.0, 0.0]

                    # Set default velocities to zero
                    robot_state["lin_vel"] = [0.0, 0.0, 0.0]
                    robot_state["ang_vel"] = [0.0, 0.0, 0.0]

                    # Set default joint positions
                    if hasattr(robot, "default_joint_positions"):
                        robot_state["dof_pos"] = dict(robot.default_joint_positions)
                        robot_state["joint_qpos"] = list(robot.default_joint_positions.values())
                    else:
                        robot_state["dof_pos"] = {}
                        robot_state["joint_qpos"] = []

                    # Set default joint velocities to zero
                    if hasattr(robot, "default_joint_positions"):
                        robot_state["dof_vel"] = {name: 0.0 for name in robot.default_joint_positions.keys()}
                        robot_state["joint_qvel"] = [0.0] * len(robot.default_joint_positions)
                    else:
                        robot_state["dof_vel"] = {}
                        robot_state["joint_qvel"] = []

                    state["robots"][robot.name] = robot_state

                    # Set objects to default states
                    for obj in self.scenario.task.objects:
                        obj_state = {}

                        if hasattr(obj, "default_position"):
                            obj_state["pos"] = list(obj.default_position)
                        else:
                            obj_state["pos"] = [0.0, 0.0, 0.0]

                        if hasattr(obj, "default_orientation"):
                            obj_state["rot"] = list(obj.default_orientation)
                        else:
                            obj_state["rot"] = [1.0, 0.0, 0.0, 0.0]

                        obj_state["lin_vel"] = [0.0, 0.0, 0.0]
                        obj_state["ang_vel"] = [0.0, 0.0, 0.0]

                        state["objects"][obj.name] = obj_state

                    reset_states.append(state)

            return reset_states

        # Unknown state format
        return None

    def randomize_initial_states(self, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.env.handler.get_states()

        # Handle TensorState object from IsaacGym
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            # For IsaacGym/TensorState, we don't randomize here as it's handled in the task reset
            return states

        # Handle list of states (for other simulators)
        if isinstance(states, list):
            for state in states:
                if hasattr(self.scenario.robots[0], "default_position"):
                    state["robots"][self.scenario.robots[0].name]["pos"] = self.scenario.robots[0].default_position
                if hasattr(self.scenario.robots[0], "default_orientation"):
                    quat = self.scenario.robots[0].default_orientation
                    state["robots"][self.scenario.robots[0].name]["rot"] = self._normalize_quaternion(quat)
                if hasattr(self.scenario.robots[0], "default_joint_positions"):
                    for joint_name, joint_pos in self.scenario.robots[0].default_joint_positions.items():
                        state["robots"][self.scenario.robots[0].name]["dof_pos"][joint_name] = joint_pos

                for obj in self.scenario.task.objects:
                    if hasattr(obj, "default_position"):
                        state["objects"][obj.name]["pos"] = obj.default_position
                    if hasattr(obj, "default_orientation"):
                        quat = obj.default_orientation
                        state["objects"][obj.name]["rot"] = self._normalize_quaternion(quat)

            if "robot" in self.scenario.task.randomize:
                if "position" in self.scenario.task.randomize["robot"]:
                    new_x = random.uniform(
                        self.scenario.task.randomize["robot"]["position"]["x"][0],
                        self.scenario.task.randomize["robot"]["position"]["x"][1],
                    )
                    new_y = random.uniform(
                        self.scenario.task.randomize["robot"]["position"]["y"][0],
                        self.scenario.task.randomize["robot"]["position"]["y"][1],
                    )
                    new_z = random.uniform(
                        self.scenario.task.randomize["robot"]["position"]["z"][0],
                        self.scenario.task.randomize["robot"]["position"]["z"][1],
                    )
                    state["robots"][self.scenario.robots[0].name]["pos"] = (new_x, new_y, new_z)
                if "orientation" in self.scenario.task.randomize["robot"]:
                    new_x = random.uniform(
                        self.scenario.task.randomize["robot"]["orientation"]["x"][0],
                        self.scenario.task.randomize["robot"]["orientation"]["x"][1],
                    )
                    new_y = random.uniform(
                        self.scenario.task.randomize["robot"]["orientation"]["y"][0],
                        self.scenario.task.randomize["robot"]["orientation"]["y"][1],
                    )
                    new_z = random.uniform(
                        self.scenario.task.randomize["robot"]["orientation"]["z"][0],
                        self.scenario.task.randomize["robot"]["orientation"]["z"][1],
                    )
                    new_w = random.uniform(
                        self.scenario.task.randomize["robot"]["orientation"]["w"][0],
                        self.scenario.task.randomize["robot"]["orientation"]["w"][1],
                    )
                    quat = (new_w, new_x, new_y, new_z)
                    state["robots"][self.scenario.robots[0].name]["rot"] = self._normalize_quaternion(quat)
            if "object" in self.scenario.task.randomize:
                for obj in self.scenario.task.randomize["object"]:
                    if "position" in self.scenario.task.randomize["object"][obj]:
                        new_x = random.uniform(
                            self.scenario.task.randomize["object"][obj]["position"]["x"][0],
                            self.scenario.task.randomize["object"][obj]["position"]["x"][1],
                        )
                        new_y = random.uniform(
                            self.scenario.task.randomize["object"][obj]["position"]["y"][0],
                            self.scenario.task.randomize["object"][obj]["position"]["y"][1],
                        )
                        new_z = random.uniform(
                            self.scenario.task.randomize["object"][obj]["position"]["z"][0],
                            self.scenario.task.randomize["object"][obj]["position"]["z"][1],
                        )
                        state["objects"][obj]["pos"] = (new_x, new_y, new_z)
                    if "orientation" in self.scenario.task.randomize["object"][obj]:
                        new_x = random.uniform(
                            self.scenario.task.randomize["object"][obj]["orientation"]["x"][0],
                            self.scenario.task.randomize["object"][obj]["orientation"]["x"][1],
                        )
                        new_y = random.uniform(
                            self.scenario.task.randomize["object"][obj]["orientation"]["y"][0],
                            self.scenario.task.randomize["object"][obj]["orientation"]["y"][1],
                        )
                        new_z = random.uniform(
                            self.scenario.task.randomize["object"][obj]["orientation"]["z"][0],
                            self.scenario.task.randomize["object"][obj]["orientation"]["z"][1],
                        )
                        new_w = random.uniform(
                            self.scenario.task.randomize["object"][obj]["orientation"]["w"][0],
                            self.scenario.task.randomize["object"][obj]["orientation"]["w"][1],
                        )
                        quat = (new_w, new_x, new_y, new_z)
                        state["objects"][obj]["rot"] = self._normalize_quaternion(quat)

        return states

    def _register_configs(self):
        """Register configurations for the environment based on the wrapped GymEnvWrapper."""
        self.max_episode_length = self.scenario.task.episode_length
        self.handler = self.env.handler
        self.headless = self.scenario.headless
        self.num_envs = self.handler.num_envs
        self._task = self.scenario.task
        # Handle dm_control tasks which don't have robots
        self._robot = self.scenario.robots[0] if self.scenario.robots else None
        self.device = self.handler.device
        self.rgb_observation = len(self.scenario.cameras) > 0
        self.obs_space = self.get_observation_space()

        if self.rgb_observation:
            cam_cfg = self.scenario.cameras[0]
            self.camera_resolution_width = cam_cfg.width
            self.camera_resolution_height = cam_cfg.height

    def get_observation_space(self):
        """Get the observation space for the environment."""
        return self.env.observation_space

    def get_action_space(self):
        """Get the action space for the environment."""
        return self.env.action_space

    def _set_up_buffers(self):
        """Set up buffers for the environment."""
        self.reset_buffer = torch.zeros(self.num_envs, 1, device=self.device)
        self.success_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.rgb_observation:
            cam_cfg = self.scenario.cameras[0]
            self.rgb_buffers = torch.zeros(self.num_envs, 3, cam_cfg.height, cam_cfg.width, device=self.device)

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if hasattr(self.env.handler, "seed"):
            self.env.handler.seed(seed)
        elif hasattr(self.env.handler, "set_seed"):
            self.env.handler.set_seed(seed)
        else:
            log.warning("Could not set seed on underlying handler.")
        return seed

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment forward using the wrapped GymEnv."""
        self.total_steps += 1
        start_time = time.time()

        action_space = self.get_action_space()
        # Ensure action is on the correct device
        if isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            action = torch.tensor(action, device=self.device, dtype=torch.float32)

        if isinstance(action_space, gym.spaces.box.Box):
            action_low = torch.tensor(action_space.low, device=self.device, dtype=torch.float32)
            action_high = torch.tensor(action_space.high, device=self.device, dtype=torch.float32)
            action_center = (action_high + action_low) / 2.0
            action_scale = (action_high - action_low) / 2.0
            processed_action = action_center + action * action_scale
        else:
            log.warning(f"Action space type {type(action_space)} not Box, not scaling actions.")
            processed_action = action

        # For dm_control tasks, directly use the environment's step
        if self._robot is None:
            states, reward, success, timeout, extra = self.env.step(processed_action)
            observation = self.get_observation(states)

            if self.rgb_observation:
                observation["rgb"] = torch.zeros(
                    self.num_envs, 3, self.camera_resolution_height, self.camera_resolution_width, device=self.device
                )

            reward = torch.nan_to_num(reward, nan=0.0)

            reset_mask = success | timeout
            self.reset_buffer[:, 0] = reset_mask.float()
            self.timestep_buffer += 1

            reset_indices = torch.where(reset_mask)[0]
            if len(reset_indices) > 0:
                reset_env_ids = reset_indices.tolist()
                self.reset(env_ids=reset_env_ids)

            self.success_buffer = success

            # Ensure reward and timeout have the right shape for PPO
            if reward.dim() == 1:
                reward = reward.unsqueeze(-1)
            if timeout.dim() == 1:
                timeout = timeout.unsqueeze(-1)

            return observation, reward.to(self.device), success.to(self.device), timeout.to(self.device), extra or {}

        joint_names = list(self._robot.actuators.keys())

        action_dict = []
        for env_idx in range(self.num_envs):
            env_action = {self._robot.name: {"dof_pos_target": {}}}
            for i, joint_name in enumerate(joint_names):
                env_action[self._robot.name]["dof_pos_target"][joint_name] = float(processed_action[env_idx, i])
            action_dict.append(env_action)

        # states, reward, success, timeout, termination = self.env.step(action_dict)
        self.env._episode_length_buf += 1
        self.env.handler.set_dof_targets(self.env.handler.robot.name, action_dict)
        self.env.handler.simulate()
        reward = self.env.handler.task.reward_fn(self.env.handler.get_states(), action_dict).to(self.device)
        termination = (
            self.env.handler.task.termination_fn(self.env.handler.get_states()).to(self.device)
            if hasattr(self.env.handler.task, "termination_fn")
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        success = self.env.handler.checker.check(self.env.handler).to(self.device)
        states = self.env.handler.get_states()
        timeout = (self.env._episode_length_buf >= self.env.handler.scenario.episode_length).to(self.device)
        done = (success | timeout | termination).int()

        reward = reward.unsqueeze(-1)
        timeout = timeout.unsqueeze(-1)
        observation = self.get_observation(states)

        self.timestep_buffer += 1
        self.success_buffer = success.clone()
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.timestep_buffer[reset_env_ids] = 0

        if done.any():
            self.reset(env_ids=list(reset_env_ids.cpu().numpy()))

        end_time = time.time()
        step_time = end_time - start_time
        self.total_time += step_time
        return observation, reward, done, timeout, termination

    def reset(self, env_ids: list[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset environments using the wrapped GymEnv."""
        if env_ids is None:
            reset_indices = list(range(self.num_envs))
        else:
            reset_indices = env_ids

        # Get the default states for reset
        reset_states = self._get_default_reset_states(env_ids=env_ids)

        # Reset the environments with the default states
        if reset_states is not None:
            states, _ = self.env.reset(env_ids=env_ids, states=reset_states)
        else:
            states, _ = self.env.reset(env_ids=env_ids)

        # Call task-specific reset if available
        if hasattr(self._task, "reset"):
            self._task.reset(env_ids=env_ids)

        observation = self.get_observation(states)

        self.reset_buffer[reset_indices] = torch.ones_like(self.reset_buffer[reset_indices])
        self.timestep_buffer[reset_indices] = torch.zeros_like(self.timestep_buffer[reset_indices])
        self.success_buffer[reset_indices] = torch.zeros_like(self.success_buffer[reset_indices])

        # Reset episode length buffer for the reset environments
        for idx in reset_indices:
            self.env._episode_length_buf[idx] = 0

        return observation

    def get_observation(self, states: Obs) -> dict[str, torch.Tensor]:
        """Process the observation from the handler/GymEnv state format into the dict expected by PPO."""

        # Check if the task has its own get_observation method (e.g., AllegroHand)
        if hasattr(self.scenario.task, "get_observation"):
            # For TensorState objects, we need to convert to the format expected by the task
            if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
                # Convert TensorState to list of dicts for task's get_observation
                states_list = []
                for i in range(self.num_envs):
                    state_dict = {
                        "robots": {},
                        "objects": {},
                    }

                    # Extract robot states
                    for robot_name, robot_state in states.robots.items():
                        # Extract and clean position/rotation data
                        pos_data = (
                            robot_state.root_state[i, :3].cpu()
                            if robot_state.root_state is not None
                            else torch.tensor([0.0, 0.0, 0.0])
                        )
                        rot_data = (
                            robot_state.root_state[i, 3:7].cpu()
                            if robot_state.root_state is not None
                            else torch.tensor([1.0, 0.0, 0.0, 0.0])
                        )

                        # Normalize quaternion to prevent NaN
                        rot_norm = torch.norm(rot_data)
                        if rot_norm > 0:
                            rot_data = rot_data / rot_norm
                        else:
                            rot_data = torch.tensor([1.0, 0.0, 0.0, 0.0])

                        state_dict["robots"][robot_name] = {
                            "pos": torch.nan_to_num(pos_data, nan=0.0).numpy(),
                            "rot": rot_data.numpy(),
                            "joint_qpos": torch.nan_to_num(robot_state.joint_pos[i].cpu(), nan=0.0).numpy()
                            if robot_state.joint_pos is not None
                            else np.zeros(16),
                            "joint_qvel": torch.nan_to_num(robot_state.joint_vel[i].cpu(), nan=0.0).numpy()
                            if robot_state.joint_vel is not None
                            else np.zeros(16),
                            "dof_pos": {
                                f"joint_{j}": float(torch.nan_to_num(robot_state.joint_pos[i, j], nan=0.0).item())
                                for j in range(robot_state.joint_pos.shape[1])
                            }
                            if robot_state.joint_pos is not None
                            else {},
                            "dof_vel": {
                                f"joint_{j}": float(torch.nan_to_num(robot_state.joint_vel[i, j], nan=0.0).item())
                                for j in range(robot_state.joint_vel.shape[1])
                            }
                            if robot_state.joint_vel is not None
                            else {},
                        }

                    # Extract object states
                    for obj_name, obj_state in states.objects.items():
                        # Extract and clean object data
                        obj_pos = obj_state.root_state[i, :3].cpu()
                        obj_rot = obj_state.root_state[i, 3:7].cpu()
                        obj_lin_vel = obj_state.root_state[i, 7:10].cpu()
                        obj_ang_vel = obj_state.root_state[i, 10:13].cpu()

                        # Normalize object quaternion
                        obj_rot_norm = torch.norm(obj_rot)
                        if obj_rot_norm > 0:
                            obj_rot = obj_rot / obj_rot_norm
                        else:
                            obj_rot = torch.tensor([1.0, 0.0, 0.0, 0.0])

                        state_dict["objects"][obj_name] = {
                            "pos": torch.nan_to_num(obj_pos, nan=0.0).numpy(),
                            "rot": obj_rot.numpy(),
                            "lin_vel": torch.nan_to_num(obj_lin_vel, nan=0.0).numpy(),
                            "ang_vel": torch.nan_to_num(obj_ang_vel, nan=0.0).numpy(),
                        }

                    states_list.append(state_dict)

                obs = self.scenario.task.get_observation(states_list)
            else:
                obs = self.scenario.task.get_observation(states)

            # Convert to tensor if needed and ensure it's on the right device
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs = obs.to(self.device)

            # Final NaN check
            if torch.isnan(obs).any():
                print(f"Warning: NaN detected in final observation tensor. Shape: {obs.shape}")
                obs = torch.nan_to_num(obs, nan=0.0)

            return {"obs": obs}

        if isinstance(states, list) and len(states) > 0 and isinstance(states[0], dict) and "observations" in states[0]:
            obs_list = []
            for state in states:
                obs = state["observations"]
                if isinstance(obs, (list, tuple)):
                    obs = np.array(obs)
                obs_list.append(obs)

            obs_array = np.array(obs_list)
            obs_tensor = torch.tensor(obs_array, device=self.device, dtype=torch.float32)

            if torch.isnan(obs_tensor).any():
                print(f"Warning: NaN detected in OGBench observations. Shape: {obs_tensor.shape}")
                obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0)

            return {"obs": obs_tensor}

        # Fallback to original implementation for tasks without get_observation
        # XXX: currently only one of "joint_qpos" or "rgb" is supported. If there's a mixture of
        # both, this will raise an error. (also joint_vel is currently not supported)
        obs_mode = ""
        if "joint_qpos" in self.obs_space.spaces.keys():
            obs_shape_from_space = self.obs_space["joint_qpos"].shape
            obs_mode = "joint_qpos"
        elif "rgb" in self.obs_space.spaces.keys():
            obs_shape_from_space = self.obs_space["rgb"].shape
            obs_mode = "rgb"
        else:
            raise ValueError(f"Observation space {self.obs_space} is not supported.")

        obs_tensor = torch.zeros((self.num_envs, *obs_shape_from_space), device=self.device, dtype=torch.float32)

        for i, state in enumerate(states):
            if obs_mode == "joint_qpos":
                obs_tensor[i, :] = torch.tensor(
                    list(state["robots"][self._robot.name]["dof_pos"].values()), device=self.device
                )
            elif obs_mode == "rgb" and "rgb" in state:
                obs_tensor[i, :] = torch.tensor(
                    state["cameras"][0]["rgb"], device=self.device
                )  # TODO: this need to be tested in the future

        return {"obs": obs_tensor}

    def render(self) -> None:
        """Render the environment by delegating to the wrapped GymEnv."""
        self.env.render()

    def close(self) -> None:
        """Close the environment by delegating to the wrapped GymEnv."""
        self.env.close()

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose timing output"""
        self.verbose = verbose
