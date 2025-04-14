"""Gym-like environment wrapper."""

from __future__ import annotations

from typing import Generic, TypeVar

import gymnasium as gym
import torch
from loguru import logger as log

from metasim.sim import BaseSimHandler
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, TimeOut

THandler = TypeVar("THandler", bound=BaseSimHandler)


class EnvWrapper(Generic[THandler]):
    """Gym-like environment wrapper."""

    handler: THandler

    def __init__(self, *args, **kwargs) -> None: ...
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]: ...
    def render(self) -> None: ...
    def close(self) -> None: ...

    @property
    def episode_length_buf(self) -> list[int]: ...


def IdentityEnvWrapper(cls: type[BaseSimHandler]) -> type[EnvWrapper[BaseSimHandler]]:
    """Gym-like environment wrapper for IsaacLab."""

    class IdentityEnv(EnvWrapper[BaseSimHandler]):
        def __init__(self, *args, **kwargs):
            self.handler = cls(*args, **kwargs)
            self.handler.launch()

        def reset(self, states: list[EnvState] | None = None, env_ids: list[int] | None = None) -> tuple[Obs, Extra]:
            if env_ids is None:
                env_ids = list(range(self.handler.num_envs))

            if states is not None:
                self.handler.set_states(states, env_ids=env_ids)
            return self.handler.reset(env_ids=env_ids)

        def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            return self.handler.step(action)

        def render(self) -> None:
            log.warning("render() is not implemented yet")

        def close(self) -> None:
            self.handler.close()

        @property
        def episode_length_buf(self) -> list[int]:
            return self.handler.episode_length_buf

        @property
        def observation_space(self) -> gym.Space:
            return self.handler.scenario.task.observation_space

        @property
        def action_space(self) -> gym.Space:
            action_low = torch.tensor(
                [limit[0] for limit in self.handler.scenario.robot.joint_limits.values()], dtype=torch.float32
            )
            action_high = torch.tensor(
                [limit[1] for limit in self.handler.scenario.robot.joint_limits.values()], dtype=torch.float32
            )
            return gym.spaces.Box(low=action_low, high=action_high, shape=(len(action_low),), dtype=torch.float32)

    return IdentityEnv


def GymEnvWrapper(cls: type[THandler]) -> type[EnvWrapper[THandler]]:
    """Gym-like environment wrapper for IsaacGym, MuJoCo, Pybullet, SAPIEN, Genesis, etc."""

    class GymEnv:
        def __init__(self, *args, **kwargs):
            self.handler = cls(*args, **kwargs)
            self.handler.launch()
            self._episode_length_buf = torch.zeros(self.handler.num_envs, dtype=torch.int32)

        def reset(self, states: list[EnvState] | None = None, env_ids: list[int] | None = None) -> tuple[Obs, Extra]:
            if env_ids is None:
                env_ids = list(range(self.handler.num_envs))

            self._episode_length_buf[env_ids] = 0
            if states is not None:
                self.handler.set_states(states, env_ids=env_ids)
            self.handler.checker.reset(self.handler, env_ids=env_ids)
            self.handler.refresh_render()
            states = self.handler.get_states()
            return states, None

        def step(self, actions: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            self._episode_length_buf += 1
            self.handler.set_dof_targets(self.handler.robot.name, actions)
            self.handler.simulate()
            reward = None
            success = self.handler.checker.check(self.handler)
            states = self.handler.get_states()
            time_out = self._episode_length_buf >= self.handler.scenario.episode_length
            return states, reward, success, time_out, None

        def render(self) -> None:
            log.warning("render() is not implemented yet")
            pass

        def close(self) -> None:
            self.handler.close()

        def _get_reward(self) -> Reward:
            if hasattr(self.handler.task, "reward_fn"):
                # XXX: compatible with old states format
                states = [{**state["robots"], **state["objects"]} for state in self.handler.get_states()]
                return self.handler.task.reward_fn(states)
            else:
                return None

        @property
        def episode_length_buf(self) -> list[int]:
            return self._episode_length_buf.tolist()

        @property
        def action_space(self) -> gym.Space:
            action_low = torch.tensor(
                [limit[0] for limit in self.handler.scenario.robot.joint_limits.values()], dtype=torch.float32
            )
            action_high = torch.tensor(
                [limit[1] for limit in self.handler.scenario.robot.joint_limits.values()], dtype=torch.float32
            )
            return gym.spaces.Box(low=action_low, high=action_high, shape=(len(action_low),), dtype=torch.float32)

        @property
        def observation_space(self) -> gym.Space:
            observation_space = {}
            for obj in self.handler.scenario.task.observation_space.keys():
                if obj == "robot":
                    for joint in self.handler.scenario.robot.joint_names:
                        observation_space[joint] = gym.spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float32
                        )
                else:
                    for key, value in self.handler.scenario.task.observation_space[obj].items():
                        observation_space[obj][key] = gym.spaces.Box(
                            low=value["low"], high=value["high"], shape=value["shape"], dtype=value["dtype"]
                        )
            return gym.spaces.Dict(observation_space)

    return GymEnv
