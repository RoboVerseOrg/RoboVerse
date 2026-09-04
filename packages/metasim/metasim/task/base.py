"""A base task env for roboverse."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch

from metasim.constants import SimType
from metasim.queries.base import BaseQueryType
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import CompatActionInput, Info, Obs, Reward, Success, Termination, TimeOut
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
    supported_simulators: tuple[str, ...] | None = None
    """Backends this task is known to run on (e.g. ``("mujoco", "newton")``).

    ``None`` means *undeclared* (no check). When declared, constructing the task with a
    ``scenario.simulator`` outside the tuple raises ``ValueError`` at construction instead of failing
    later inside a backend — a task whose assets, actuator patching or success checker only exist for
    some backends must say so here. The registry and the docs task-by-simulator matrix read this field.
    """

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
        if scenario is None:
            scenario = type(self).scenario  # the docstring contract for class-level defaults
            if scenario is None:
                raise ValueError(
                    f"{type(self).__name__} requires a scenario — pass one to __init__ or set the class attribute."
                )
        self.scenario = scenario
        self.num_envs = self.scenario.num_envs
        self._check_supported_simulator(scenario)

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

    @classmethod
    def _check_supported_simulator(cls, scenario) -> None:
        """Raise if ``scenario.simulator`` is outside the task's declared ``supported_simulators``."""
        if cls.supported_simulators is None or isinstance(scenario, BaseSimHandler):
            return
        sim = getattr(scenario, "simulator", None)
        sim_name = getattr(sim, "value", sim)
        if sim_name is not None and str(sim_name) not in cls.supported_simulators:
            raise ValueError(
                f"{cls.__name__} does not support simulator {sim_name!r}; declared supported_simulators = "
                f"{cls.supported_simulators}. Pass a scenario with one of those backends, or extend the task "
                "and add the backend to supported_simulators once it is verified."
            )

    def _get_initial_states(self) -> list[dict]:
        """Return per-env initial states (override in subclasses)."""
        return None

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """Instantiate the environment.

        Args:
            scenario: The scenario configuration

        ``extra_spec`` is read *before* the handler exists, so a subclass
        ``_extra_spec`` may reference ``self.handler`` (it will be ``None``,
        not undefined) but must not require a live handler — queries are
        bound later, inside ``handler.launch()``.
        """
        handler_class = get_sim_handler_class(SimType(scenario.simulator))
        self.handler: BaseSimHandler | None = None
        queries = self.extra_spec
        self.handler = handler_class(scenario, queries)
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

    def _process_action(self, actions: CompatActionInput) -> CompatActionInput:
        """Transform actions before they are applied to the handler (default: identity).

        This is the sanctioned hook for action transforms (delta control,
        unnormalisation, end-effector mapping, …). Override it instead of
        overriding ``step`` — ``step`` calls it first, so the rest of the step
        pipeline (callbacks, clamping, simulate) stays shared and uniform.
        """
        return actions

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

    def _as_step_tensor(self, value, *, dtype: torch.dtype, hook: str) -> torch.Tensor:
        """``value`` as a ``(num_envs,)`` tensor of ``dtype`` on the env device: the type every ``step()`` field has.

        ``_reward`` / ``_terminated`` / ``_time_out`` may return a tensor of the dtype ``get_states``
        produced, a hand-built tensor without ``device=``, or a Python / numpy sequence; consumers get
        one shape, dtype and device regardless of backend and base class. A wrong shape (a scalar, an
        ``(N, 1)`` column, a tuple) is rejected here by name: downstream it would silently disable
        ``RLTaskEnv``'s auto-reset or index ``reset`` with nested env ids.
        """
        try:
            tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError(
                f"{type(self).__name__}.{hook} returned {type(value).__name__}, which is not a ({self.num_envs},) "
                f"{dtype} tensor: {err}"
            ) from err
        if tuple(tensor.shape) != (self.num_envs,):
            raise ValueError(
                f"{type(self).__name__}.{hook} returned shape {tuple(tensor.shape)}; step() needs one value per "
                f"env, shape ({self.num_envs},)."
            )
        return tensor

    def _run_reset_callbacks(self, env_ids: list[int]) -> None:
        """Run the task's reset callbacks.

        A list of ``fn(env_ids)`` (the base contract) or the humanoid packs' ``{name: (fn, params)}``
        mapping; anything else is a configuration error, not a no-op.
        """
        callbacks = self.reset_callback
        if isinstance(callbacks, dict):
            for fn, params in callbacks.values():
                fn(self, env_ids, **(params or {}))
            return
        if not isinstance(callbacks, (list, tuple)):
            raise TypeError(
                f"{type(self).__name__}.reset_callback must be a list of callables or a name -> (fn, params) dict"
            )
        for callback in callbacks:
            callback(env_ids)

    def _write_reset_states(self, states, env_ids: list[int], *, refresh: bool = True) -> None:
        """Write the reset states and refresh the renderer unless the backend's write already did.

        ``set_states_refreshes`` says whether the write refreshed (on Isaac Sim the unconditional
        refresh was two extra RTX passes); ``refresh=False`` skips it where no frame is consumed (the
        RL auto-reset without cameras).
        """
        self.handler.set_states(states=states, env_ids=env_ids)
        if refresh and not self.handler.set_states_refreshes:
            self.handler.refresh_render()

    def _time_out(self, env_states) -> torch.Tensor:
        """Timeout flags."""
        return self._episode_steps >= self.max_episode_steps

    def step(self, actions: CompatActionInput) -> tuple[Obs, Reward, Success, TimeOut, Info | None]:
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
        actions = self._process_action(actions)
        for callback in self.pre_physics_step_callback:
            callback(actions)

        self.handler.set_dof_targets(actions)

        self.handler.simulate()

        env_states = self.handler.get_states(mode="tensor")

        for callback in self.post_physics_step_callback:
            callback(env_states)

        # compute reward/termination, normalised to the step() contract (see ``_as_step_tensor``)
        rewards: Reward = self._as_step_tensor(self._reward(env_states), dtype=torch.float32, hook="_reward")
        terminated: Termination = self._as_step_tensor(
            self._terminated(env_states), dtype=torch.bool, hook="_terminated"
        )

        # increment step counter and compute a single unified timeout
        self._episode_steps = self._episode_steps + 1
        timeout: TimeOut = self._as_step_tensor(self._time_out(env_states), dtype=torch.bool, hook="_time_out")

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
        seed: int | None = None,
    ) -> tuple[Obs, Info | None]:
        """Reset the environment.

        Args:
            env_ids: The environment ids to reset
            states: Optional external states to set for the selected envs. If None, use initial states.
            seed: Optional reproducibility seed. Propagated to ``handler.set_seed`` when
                the backend implements it. If the backend doesn't yet implement
                ``set_seed``, a one-shot warning is logged so the caller knows the
                ``env.reset(seed=N)`` contract isn't fully honoured on that backend
                rather than silently believing the rollout is reproducible.

        Returns:
            obs: The observation
            priv_obs: The privileged observation
            info: The info
        """
        if seed is not None:
            set_seed = getattr(self.handler, "set_seed", None)
            if callable(set_seed):
                set_seed(seed)
            else:
                # Warn once per task instance, not per reset — RL training resets
                # thousands of times and a per-call warning would drown the log.
                if not getattr(self, "_seed_unsupported_warned", False):
                    from loguru import logger as _log

                    _log.warning(
                        f"{type(self).__name__}: handler "
                        f"{type(self.handler).__name__} does not implement set_seed; "
                        f"env.reset(seed={seed}) only seeds the gym base RNG, not the "
                        f"simulator. Rollouts will not be bit-reproducible across runs "
                        f"until the backend adds set_seed."
                    )
                    self._seed_unsupported_warned = True
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        self._run_reset_callbacks(env_ids)
        states_to_set = self._initial_states if states is None else states
        self._write_reset_states(states_to_set, env_ids)
        env_states = self.handler.get_states(env_ids=env_ids, mode="tensor")
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
