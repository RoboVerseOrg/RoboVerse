"""A base task wrapper for roboverse"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from traitlets import Dict

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, Termination, TimeOut
from metasim.utils.setup_util import get_sim_handler_class
from metasim.utils.state import TensorState


class BaseTaskWrapper:
    def __init__(self, scenario: BaseSimHandler | ScenarioCfg) -> None:
        """
        Initialize the task wrapper.

        Args:
            scenario: The scenario configuration
        """

        if isinstance(scenario, BaseSimHandler):
            self.env = scenario
        else:
            self._instantiate_env(scenario)

        self._prepare_callbacks()

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """
        Instantiate the environment.

        Args:
            scenario: The scenario configuration
        """

        handler_class = get_sim_handler_class(SimType(scenario.sim))
        self.env: BaseSimHandler = handler_class(scenario)
        self.env.launch()

    def _prepare_callbacks(self) -> None:
        """
        Prepare the callbacks for the environment.
        """

        self.pre_physics_step_callback: list[Callable] = []
        self.post_physics_step_callback: list[Callable] = []
        self.reset_callback: list[Callable] = []

    @property
    def observation_space(self) -> gym.Space:
        """
        Get the observation space of the environment.
        """
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    @property
    def action_space(self) -> gym.Space:
        """
        Get the action space of the environment.
        """
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    def observation(self) -> Obs:
        """
        Get the observation of the environment.
        """
        return self.env.get_states()

    def privileged_observation(self) -> Obs:
        """
        Get the privileged observation of the environment.
        """
        return self.env.get_states()

    def reward(self) -> Reward:
        """
        Get the reward of the environment.
        """
        return [0.0] * self.env.num_envs

    def terminated(self) -> Termination:
        """
        Get the terminated of the environment.
        """
        return [False] * self.env.num_envs

    def time_out(self) -> TimeOut:
        """
        Get the time out of the environment.
        """
        return [False] * self.env.num_envs

    def _pre_physics_step(self, actions: Action) -> Dict():
        """
        Pre-physics step, apply transforms to actions and put actions into correct dict format.

        Args:
            actions: The actions to take
        """

        actions_dict = {
            "robots": {
                self.env.robots[0].name: {
                    "dof_pos_target": {
                        joint_name: action
                        for joint_name, action in zip(self.env.get_joint_names(self.env.robots[0].name), actions)
                    }
                }
            }
        }

        for callback in self.pre_physics_step_callback:
            callback(actions_dict)

        return actions_dict

    def _physics_step(self, actions_dict: dict) -> tuple[EnvState, Extra]:
        """
        Physics step.
        """
        # TODO: Use set_states() in new metasim handler
        # self.env.set_states(actions_dict)

        for robot in self.env.robots:
            self.env.set_dof_targets(robot.name, [actions_dict["robots"]])

        self.env.simulate()

        return self.env.get_states(), None

    def _post_physics_step(self, env_states: EnvState) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        """
        Post-physics step.
        """

        for callback in self.post_physics_step_callback:
            callback(env_states)

        return self.env.get_states(), self.reward(), self.terminated(), self.time_out(), None

    def step(self, actions: Action) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        """
        Step the environment.

        Args:
            actions: The actions to take
        """

        actions_dict = self._pre_physics_step(actions)
        env_states, _ = self._physics_step(actions_dict)
        obs, reward, terminated, time_out, _ = self._post_physics_step(env_states)

        return obs, reward, terminated, time_out, None

    def reset(self) -> tuple[TensorState, Extra]:
        """
        Reset the environment.
        """
        for callback in self.reset_callback:
            callback()

        return self.env.get_states(), None

    def close(self) -> None:
        """
        Close the environment.
        """
        self.env.close()
