from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class


class MetaSimVecEnv(VectorEnv):
    """Vectorized environment for MetaSim that supports parallel RL training."""

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        sim: str = "isaaclab",
        task_name: str | None = None,
        num_envs: int | None = 4,
    ):
        if scenario is None:
            scenario = ScenarioCfg(task=task_name, robots=["franka"])
            scenario.num_envs = num_envs
            scenario = ScenarioCfg(**vars(scenario))
        self.num_envs = scenario.num_envs
        env_class = get_sim_env_class(SimType(sim))
        env = env_class(scenario)
        self.env: EnvWrapper[BaseSimHandler] = env
        self.render_mode = None  # XXX
        self.scenario = scenario

        # Get candidate states
        self.candidate_init_states, _, _ = get_traj(scenario.task, scenario.robots[0])

        # TODO: modify to give more meaningful space
        self.single_observation_space = spaces.Box(-np.inf, np.inf)
        self.single_action_space = spaces.Box(-np.inf, np.inf)
        self.observation_space = spaces.Box(-np.inf, np.inf)
        self.action_space = spaces.Box(-np.inf, np.inf)

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self, env_ids: list[int] | None = None, seed: int | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        init_states = self.unwrapped._get_default_states(seed)
        self.env.reset(states=init_states, env_ids=env_ids)
        return self.unwrapped._get_obs(), {}

    def step(self, actions: list[dict]):
        """Take a step in the environment."""
        _, _, success, timeout, _ = self.env.step(actions)
        obs = self.unwrapped._get_obs()
        rewards = self.unwrapped._calculate_rewards()
        return obs, rewards, success, timeout, {}

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Helper methods
    ############################################################
    def _get_obs(self):
        """Get current observations for all environments."""
        ## TODO: make this method generalizable
        states = self.env.handler.get_states()
        robot_name = self.scenario.robots[0].name
        robot_cfg = self.scenario.robots[0]
        joint_pos = states.robots[robot_name].joint_pos
        panda_hand_index = states.robots[robot_name].body_names.index(robot_cfg.ee_body_name)
        ee_pos = states.robots[robot_name].body_state[:, panda_hand_index, :3]

        return torch.cat([joint_pos, ee_pos], dim=1)

    def _calculate_rewards(self):
        """Calculate rewards based on distance to origin."""
        states = self.env.handler.get_states()
        tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)

        from dataclasses import _MISSING_TYPE

        if isinstance(self.scenario.task.reward_functions, _MISSING_TYPE):
            return tot_reward

        for reward_fn, weight in zip(self.scenario.task.reward_functions, self.scenario.task.reward_weights):
            tot_reward += weight * reward_fn(states, self.scenario.robots[0].name)
        return tot_reward

    def _get_default_states(self, seed: int | None = None):
        """Generate default reset states."""
        ## TODO: use non-reqeatable random choice when there is enough candidate states?
        return random.Random(seed).choices(self.candidate_init_states, k=self.num_envs)
