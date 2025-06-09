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

    def randomize_initial_states(self, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.env.handler.get_states()

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
        self._robot = self.scenario.robots[0]
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
        if isinstance(action_space, gym.spaces.box.Box):
            action_low = torch.tensor(action_space.low, device=self.device, dtype=torch.float32)
            action_high = torch.tensor(action_space.high, device=self.device, dtype=torch.float32)
            action_center = (action_high + action_low) / 2.0
            action_scale = (action_high - action_low) / 2.0
            processed_action = action_center + action * action_scale
        else:
            log.warning(f"Action space type {type(action_space)} not Box, not scaling actions.")
            processed_action = action

        joint_names = list(self._robot.actuators.keys())

        action_dict = []
        for env_idx in range(self.num_envs):
            env_action = {"dof_pos_target": {}}
            for i, joint_name in enumerate(joint_names):
                env_action["dof_pos_target"][joint_name] = float(processed_action[env_idx, i])
            action_dict.append(env_action)

        # states, reward, success, timeout, termination = self.env.step(action_dict)
        self.env._episode_length_buf += 1
        self.env.handler.set_dof_targets(self.env.handler.robot.name, action_dict)
        self.env.handler.simulate()
        reward = self.env.handler.task.reward_fn(self.env.handler.get_states(), action_dict).to(self.device)
        termination = (
            self.env.handler.task.termination_fn(self.env.handler.get_states()).to(self.device)
            if hasattr(self.env.handler.task, "termination_fn")
            else None
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

        states = self.randomize_initial_states(env_ids=env_ids)
        states, _ = self.env.reset(env_ids=env_ids, states=states)

        observation = self.get_observation(states)

        self.reset_buffer[reset_indices] = torch.ones_like(self.reset_buffer[reset_indices])
        self.timestep_buffer[reset_indices] = torch.zeros_like(self.timestep_buffer[reset_indices])
        self.success_buffer[reset_indices] = torch.zeros_like(self.success_buffer[reset_indices])

        return observation

    def get_observation(self, states: Obs) -> dict[str, torch.Tensor]:
        """Process the observation from the handler/GymEnv state format into the dict expected by PPO."""
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
