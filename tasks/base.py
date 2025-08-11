"""A base task env for roboverse."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from metasim.constants import SimType
from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, Info, Obs, Reward, Success, Termination, TimeOut
from metasim.utils.setup_util import get_sim_handler_class
from scenario_cfg.scenario import ScenarioCfg


class BaseTaskEnv:
    """A base task env for roboverse.

    This env is used to wrap the environment to form a complete task.

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

    def __init__(self, scenario: BaseSimHandler | ScenarioCfg) -> None:
        """Initialize the task env.

        Args:
            scenario: The scenario configuration
        """
        if isinstance(scenario, BaseSimHandler):
            self.env = scenario
        else:
            self._instantiate_env(scenario)

        self._prepare_callbacks()

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """Instantiate the environment.

        Args:
            scenario: The scenario configuration
        """
        handler_class = get_sim_handler_class(SimType(scenario.simulator))
        self.env: BaseSimHandler = handler_class(scenario, self.extra_spec)
        self.env.launch()

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
        return [0.0] * self.env.num_envs

    def _terminated(self, env_states: Obs) -> Termination:
        """Get the terminated of the environment."""
        return [False] * self.env.num_envs

    def _time_out(self, env_states: Obs) -> TimeOut:
        """Get the time out of the environment."""
        return [False] * self.env.num_envs

    # def __pre_physics_step(self, actions: Action) -> Dict[str, Any]:
    #     """Pre-physics step, apply transforms to actions and put actions into correct dict format.

    #     Args:
    #         actions: The actions to take
    #     """
    #     for callback in self.pre_physics_step_callback:
    #         callback(actions)

    #     actions_dict = {
    #         "robots": {
    #             self.env.robots[0].name: {
    #                 "dof_pos_target": {
    #                     joint_name: action
    #                     for joint_name, action in zip(self.env.get_joint_names(self.env.robots[0].name), actions)
    #                 }
    #             }
    #         }
    #     }

    #     return actions_dict

    # def __physics_step(self, actions_dict: Dict[str, Any]) -> tuple[Obs, Info | None]:
    #     """Physics step."""
    #     # TODO: Use set_states() in new metasim handler
    #     # self.env.set_states(actions_dict)

    #     for robot in self.env.robots:
    #         robot_actions = actions_dict["robots"][robot.name]["dof_pos_target"]
    #         self.env.set_dof_targets(robot.name, list(robot_actions.values()))

    #     self.env.simulate()

    #     return self.env.get_states(), None

    # def __post_physics_step(self, env_states: Obs) -> tuple[Obs, Obs, Reward, Success, TimeOut, Info | None]:
    #     """Post-physics step."""
    #     for callback in self.post_physics_step_callback:
    #         callback(env_states)

    #     return (
    #         self._observation(env_states),
    #         self._reward(env_states),
    #         self._terminated(env_states),
    #         self._time_out(env_states),
    #         {"privileged_observation": self._privileged_observation(env_states)},
    #     )

    def step(self, actions: Action) -> tuple[Obs, Obs, Reward, Success, TimeOut, Info | None]:
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

        for robot in self.env.robots:
            robot_actions = actions_dict["robots"][robot.name]["dof_pos_target"]
            self.env.set_dof_targets(robot.name, list(robot_actions.values()))

        self.env.simulate()

        env_states = self.env.get_states()

        for callback in self.post_physics_step_callback:
            callback(env_states)

        return (
            self._observation(env_states),
            self._reward(env_states),
            self._terminated(env_states),
            self._time_out(env_states),
            {"privileged_observation": self._privileged_observation(env_states)},
        )

    def reset(self, env_ids: list[int] | None = None) -> tuple[Obs, Info | None]:
        """Reset the environment. This base implementation does not do anything.

        Args:
            env_ids: The environment ids to reset

        Returns:
            obs: The observation
            priv_obs: The privileged observation
            info: The info
        """
        if env_ids is None:
            env_ids = list(range(self.env.num_envs))

        for callback in self.reset_callback:
            callback(env_ids)

        env_states = self.env.get_states(env_ids=env_ids)
        info = {
            "privileged_observation": self._privileged_observation(env_states),
        }

        return self._observation(env_states), info

    def close(self) -> None:
        """Close the environment."""
        for callback in self.close_callback:
            callback()

        self.env.close()

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
