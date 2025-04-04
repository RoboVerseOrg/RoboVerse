"""Gym-like environment wrapper."""

from __future__ import annotations

from typing import Generic, TypeVar

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
            self.handler.simulate()
            obs = self.handler.get_observation()
            return obs, None

        def step(self, actions: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
            self._episode_length_buf += 1
            self.handler.set_dof_targets(self.handler.robot.name, actions)
            self.handler.simulate()
            obs = self.handler.get_observation()
            reward = self._get_reward()
            success = self.handler.checker.check(self.handler)
            time_out = self._episode_length_buf >= self.handler.scenario.episode_length
            return obs, reward, success, time_out, None

        def render(self) -> None:
            log.warning("render() is not implemented yet")
            pass

        def close(self) -> None:
            self.handler.close()

        def _get_reward(self) -> Reward:
            from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg

            if hasattr(self.handler.task, "reward_fn"):
                # XXX: compatible with old states format
                states = [{**state["robots"], **state["objects"]} for state in self.handler.get_states()]
                return self.handler.task.reward_fn(states)

            if isinstance(self.handler.task, BaseRLTaskCfg):
                final_reward = torch.zeros(self.handler.num_envs)
                for reward_func, reward_weight in zip(
                    self.handler.task.reward_functions, self.handler.task.reward_weights
                ):
                    # XXX: compatible with old states format
                    states = [{**state["robots"], **state["objects"]} for state in self.handler.get_states()]
                    final_reward += reward_func(self.handler.robot.name)(states) * reward_weight
                return final_reward
            else:
                return None

        @property
        def episode_length_buf(self) -> list[int]:
            return self._episode_length_buf.tolist()

    return GymEnv
