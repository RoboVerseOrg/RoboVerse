from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.types import Action, EnvState
from metasim.utils.setup_util import get_sim_env_class


class BiDexEnvWrapper:
    """BiDex Environment Wrapper for RL tasks."""

    def __init__(self, scenario: ScenarioCfg, seed: int | None = None):
        """Initialize the BiDex environment wrapper."""
        self.sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert SimType(scenario.sim) == SimType.ISAACGYM, "Currently only support IsaacGym simulator."
        self.num_envs = scenario.num_envs
        self.robots = scenario.robots
        self.task = scenario.task

        env_class = get_sim_env_class(SimType(scenario.sim))
        self.env = env_class(scenario)

        self.init_states = scenario.task.init_states
        if len(self.init_states) < self.num_envs:
            self.init_states = (
                self.init_states * (self.num_envs // len(self.init_states))
                + self.init_states[: self.num_envs % len(self.init_states)]
            )
        else:
            self.init_states = self.init_states[: self.num_envs]

        # FIXME action limit differs with joint limit in locomotion configuration(desire pos = scale*action + default pos)
        # Set up action space based on robot joint limits
        robot_joint_limits = {}
        for robot in scenario.robots:
            robot_joint_limits.update(robot.joint_limits)

        # action space is normalized to [-1, 1]
        self.num_acutated_joints = sum(
            1
            for robot in scenario.robots
            for joint_name in robot.joint_limits
            if robot.actuators[joint_name].fully_actuated
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_acutated_joints,), dtype=np.float32)

        # observation space
        # Create an observation space (398 dimensions) for a single environment, instead of the entire batch (num_envs,398).
        obs_shape = (self.task.obs_shape,)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=obs_shape, dtype=np.float32)
        log.info(f"Observation space: {self.observation_space}")
        log.info(f"Action space: {self.action_space}")

        self.max_episode_steps = self.task.episode_length
        log.info(f"Max episode steps: {self.max_episode_steps}")

        # Episode tracking variables for EpisodeLogCallback
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)

        if seed is not None:
            self.seed(seed)

    def scale_action(self, actions: torch.Tensor) -> list[Action]:
        """Scale actions to the range of the action space.
        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        index = 0
        step_actions = []
        for env in range(self.num_envs):
            action = {}
            for robot in self.robots:
                action[robot.name] = {"dof_pos_target": {}}
                for joint_name in robot.joint_limits.keys():
                    if robot.actuators[joint_name].fully_actuated:
                        action[robot.name]["dof_pos_target"][joint_name] = (
                            0.5
                            * (actions[env][index] + 1.0)
                            * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                            + robot.joint_limits[joint_name][0]
                        )
                        index += 1
            step_actions.append(action)
        return step_actions

    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset(states=self.init_states)

        observations = self.task.observation_fn(
            obs, torch.zeros((self.num_envs, self.num_acutated_joints), device=self.sim_device)
        )

        # Reset episode tracking variables
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)

        log.info("reset now")
        return observations

    def pre_physics_step(self, actions: torch.Tensor):
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """

        step_action = self.scale_action(actions)
        return step_action

    def post_physics_step(self, envstates: list[EnvState], actions: torch.Tensor):
        """Post physics step processing."""
        self.episode_lengths += 1
        self.env._episode_length_buf += 1
        obs = self.task.observation_fn(envstates=envstates, actions=actions)
        (self.episode_rewards, self.episode_reset, self.episode_goal_reset, self.episode_success) = self.task.reward_fn(
            envstates=envstates,
            actions=actions,
            reset_buf=self.episode_reset,
            reset_goal_buf=self.episode_goal_reset,
            episode_length_buf=self.episode_lengths,
            success_buf=self.episode_success,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).

        Returns:
            tuple: A tuple containing the following elements:
                - observations (torch.Tensor, shape=(num_envs, obs_dim)): Observations from the environment.
                - rewards (torch.Tensor, shape=(num_envs,)): Reward values for each environment.
                - dones (torch.Tensor, shape=(num_envs,)): Flags indicating if the episode has ended for each environment (due to termination or truncation).
                - infos (list[dict]): List of additional information for each environment. Each dictionary contains the "TimeLimit.truncated" key,
                                      indicating if the episode was truncated due to timeout.
        """
        step_action = self.pre_physics_step(actions)
        envstates, _, _, _, _ = self.env.step(step_action)
        self.post_physics_step(envstates, actions)

        rewards = deepcopy(self.episode_rewards)
        terminated = self.episode_lengths >= self.max_episode_steps
        not_truncated = terminated.bool() | self.episode_success.bool()
        truncated = self.episode_reset & (~not_truncated)
        dones = deepcopy(self.episode_reset)

        env_ids = self.episode_reset.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.episode_goal_reset.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_goal_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_goal_pose(goal_env_ids)

        if len(env_ids) > 0:
            envstates = self.reset_env(env_ids)

        observations = self.task.observation_fn(envstates=envstates, actions=actions, device=self.sim_device)
        observations = torch.clamp(
            observations,
            torch.tensor(self.observation_space.low, device=self.sim_device),
            torch.tensor(self.observation_space.high, device=self.sim_device),
        )

        info = []
        for i in range(self.num_envs):
            env_info = {}
            env_info["TimeLimit.truncated"] = truncated[i]
            env_info["success"] = int(self.episode_success[i])
            if dones[i]:
                env_info["terminal_observation"] = observations[i]
                env_info["episode_r"] = rewards[i]
                env_info["episode_l"] = self.episode_lengths[i]

            info.append(env_info)

        return observations, rewards, dones, info

    def reset_env(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        self.episode_lengths[env_ids] = 0
        self.episode_rewards[env_ids] = 0.0
        self.episode_success[env_ids] = 0
        self.episode_reset[env_ids] = 0
        self.reset_goal_pose(env_ids)
        env_states, _ = self.env.reset(
            states=self.init_states, env_ids=env_ids.tolist()
        )  # Todo: add randomization to init_states
        return env_states

    def reset_goal_pose(self, env_ids: torch.Tensor):
        """Reset the goal pose for specific environments."""
        self.episode_goal_reset[env_ids] = 0
        # if self.task.goal_reset_fn is not None:
        #     self.task.goal_reset_fn(env_ids=env_ids, envstates=self.env.envstates)
        # else:
        #     log.warning("No goal reset function defined in the task. Skipping goal reset.")

    def close(self):
        """Clean up environment resources."""
        self.env.close()

    def seed(self, seed):
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
