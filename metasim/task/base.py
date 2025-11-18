"""A base task env for roboverse."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from metasim.constants import SimType
from metasim.queries.base import BaseQueryType
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, Info, Obs, Reward, Success, Termination, TimeOut
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.setup_util import get_sim_handler_class


class BaseTaskEnv:
    """A base task env for roboverse.

    This env is used to wrap the environment to form a complete task.

    The default scenario config is defined by the class variable "scenario". One can modify it and pass it to the __init__ method.

    To write your own task, you need to inherit this class and override the following methods:
    - _observation
    - _privileged_observation
    - _reward
    - _terminated
    - _time_out
    - _observation_space
    - _action_space
    - _extra_spec

    And use callbacks to modify the environment. The callbacks are:
    - pre_physics_step_callback: Called before the physics step
    - post_physics_step_callback: Called after the physics step
    - reset_callback: Called when the environment is reset
    - close_callback: Called when the environment is closed

    Some methods you usually should not override.
    - step
    - reset
    - close
    """

    max_episode_steps = 100
    traj_filepath = None

    def __init__(
        self,
        scenario: BaseSimHandler | ScenarioCfg | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the task env.

        Args:
            scenario: The scenario configuration. If None, it will use the class variable "scenario".
            device: The device to use for the environment. If None, it will use "cuda" if available, otherwise "cpu".
        """
        self.scenario = scenario
        self.num_envs = self.scenario.num_envs

        if isinstance(self.scenario, BaseSimHandler):
            self.handler = self.scenario
        else:
            self._instantiate_env(self.scenario)
        if self.traj_filepath is not None:
            check_and_download_single(self.traj_filepath)

        self._initial_states = self._get_initial_states()
        self.device = self.handler.device
        self._prepare_callbacks()
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _get_initial_states(self) -> list[dict]:
        """Return per-env initial states (override in subclasses)."""
        return None

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """Instantiate the environment.

        Args:
            scenario: The scenario configuration
        """
        handler_class = get_sim_handler_class(SimType(scenario.simulator))
        self.handler: BaseSimHandler = handler_class(scenario, self.extra_spec)
        self.handler.launch()

    def _prepare_callbacks(self) -> None:
        """Prepare the callbacks for the environment."""
        self.pre_physics_step_callback: list[Callable] = []
        self.post_physics_step_callback: list[Callable] = []
        self.reset_callback: list[Callable] = []
        self.close_callback: list[Callable] = []

    def _observation_space(self) -> gym.Space:
        """Get the observation space of the environment."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    def _action_space(self) -> gym.Space:
        """Get the action space of the environment."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    def _extra_spec(self) -> dict[str, BaseQueryType]:
        """Get the extra spec of the environment."""
        return {}

    def _observation(self, env_states: Obs) -> Obs:
        """Get the observation of the environment."""
        return env_states

    def _privileged_observation(self, env_states: Obs) -> Obs:
        """Get the privileged observation of the environment."""
        return env_states

    def _reward(self, env_states: Obs) -> Reward:
        """Get the reward of the environment."""
        return torch.zeros(self.handler.num_envs, dtype=torch.float32, device=self.device)

    def _terminated(self, env_states: Obs) -> Termination:
        """Get the terminated of the environment."""
        return torch.zeros(self.handler.num_envs, dtype=torch.bool, device=self.device)

    def _time_out(self, env_states) -> torch.Tensor:
        """Timeout flags."""
        return self._episode_steps >= self.max_episode_steps

    def step(self, actions: Action) -> tuple[Obs, Reward, Success, TimeOut, Info | None]:
        """Step the environment.

        Args:
            actions: The actions to take
        """
        # actions = self.__pre_physics_step(actions)
        # env_states, _ = self.__physics_step(actions)
        # obs, priv_obs, reward, terminated, time_out, _ = self.__post_physics_step(env_states)

        # info = {
        #     "privileged_observation": priv_obs,
        # }

        # return obs, reward, terminated, time_out, info
        for callback in self.pre_physics_step_callback:
            callback(actions)

        for robot in self.handler.robots:
            self.handler.set_dof_targets(actions)

        self.handler.simulate()

        env_states = self.handler.get_states()

        for callback in self.post_physics_step_callback:
            callback(env_states)

        # compute reward/termination
        rewards: Reward = self._reward(env_states)
        terminated: Termination = self._terminated(env_states)

        # increment step counter and compute a single unified timeout
        self._episode_steps = self._episode_steps + 1
        timeout: TimeOut = self._time_out(env_states)

        return (
            self._observation(env_states),
            rewards,
            terminated,
            timeout,
            {"privileged_observation": self._privileged_observation(env_states)},
        )

    def reset(
        self,
        states=None,
        env_ids: list[int] | None = None,
    ) -> tuple[Obs, Info | None]:
        """Reset the environment.

        Args:
            env_ids: The environment ids to reset
            states: Optional external states to set for the selected envs. If None, use initial states.

        Returns:
            obs: The observation
            priv_obs: The privileged observation
            info: The info
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        for callback in self.reset_callback:
            callback(env_ids)
        states_to_set = self._initial_states if states is None else states
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        env_states = self.handler.get_states(env_ids=env_ids)
        info = {
            "privileged_observation": self._privileged_observation(env_states),
        }

        # reset episode step counters for reset envs
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._episode_steps[ids] = 0

        return self._observation(env_states), info

    def close(self) -> None:
        """Close the environment."""
        for callback in self.close_callback:
            callback()

        self.handler.close()

    @property
    def observation_space(self) -> gym.Space:
        """Get the observation space of the environment."""
        return self._observation_space()

    @property
    def action_space(self) -> gym.Space:
        """Get the action space of the environment."""
        return self._action_space()

    @property
    def extra_spec(self) -> dict[str, BaseQueryType]:
        """Extra specs are optional queries that are used in handler.get_extra() stage."""
        return self._extra_spec()
