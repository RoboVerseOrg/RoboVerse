from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Action, Reward, TensorState
from metasim.utils.state import list_state_to_tensor
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg, CallbacksCfg


class AgentTask(RLTaskEnv):
    """Base RLTaskEnv wrapper shared across Unitree locomotion embodiments."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: Any | BaseEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = config
        _callbacks_cfg = asdict(getattr(self.cfg, "callbacks", CallbacksCfg()))
        self._query: dict = _callbacks_cfg.pop("query", {})
        self.robot = scenario.robots[0]
        BaseTaskEnv.__init__(self, scenario=scenario, device=device)
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)
        # buffers will be allocated lazily once handler is available
        self.obs_buf_queue: deque[torch.Tensor] | None = None
        self.priv_obs_buf_queue: deque[torch.Tensor] | None = None
        self.actions: torch.Tensor | None = None
        # self.torques: torch.Tensor | None = None
        self.rew_buf: torch.Tensor | None = None
        self.reset_buf: torch.Tensor | None = None
        # self.time_out_buf: torch.Tensor | None = None
        self.extras: dict[str, Any] = {}
        self._default_env_states = deepcopy(self._initial_states)
        self.setup_initial_env_states = deepcopy(self._initial_states)
        self.extras_buffer: dict[str, any] = {}

        # Callbacks
        self._bind_callbacks(callbacks=_callbacks_cfg)

    def _bind_callbacks(self, callbacks: dict | None = None):
        for _callbacks in callbacks.values():
            for _key, _val in _callbacks.items():
                if not isinstance(_val, tuple):
                    assert callable(_val) or isinstance(_val, object)
                    _callbacks[_key] = (_val, {})
                if hasattr(_callbacks[_key][0], "bind_handler"):
                    _callbacks[_key][0].bind_handler(self.handler)

        _setup_callbacks = callbacks.pop("setup", {})
        for _setup_fn, _params in _setup_callbacks.values():
            _setup_fn(**_params)  ## call itself
        self.reset_callback = callbacks.pop("reset", {})
        assert isinstance(self.reset_callback, dict)
        self.pre_physics_step_callback = callbacks.pop("pre_step", {})
        assert isinstance(self.pre_physics_step_callback, dict)
        self.post_physics_step_callback = callbacks.pop("post_step", {})
        assert isinstance(self.post_physics_step_callback, dict)
        self.terminate_callback = callbacks.pop("terminate", {})
        assert isinstance(self.terminate_callback, dict)

    # ------------------------------------------------------------------ #
    # RLTaskEnv hooks
    # ------------------------------------------------------------------ #
    def _get_initial_states(self):
        raise NotImplementedError

    def _extra_spec(self) -> dict:
        """Expose optional sensor queries to the simulator handler."""
        return self._query

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_states(self) -> TensorState:
        """Get the current simulator state."""
        return self.handler.get_states()

    def set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set simulator state for selected env indices."""
        self.handler.set_states(states=states, env_ids=env_ids)

    def _physics_step(self, actions: Action) -> TensorState:
        """Issue low-level actions and simulate one physics step."""
        self.handler.set_dof_targets(actions)
        self.handler.simulate()  # decimation control in task_env level
        return self.handler.get_states()

    def _reward(self, env_states: TensorState) -> Reward:
        raise NotImplementedError

    def _terminated(self, env_states: TensorState) -> torch.BoolTensor:
        raise NotImplementedError

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        raise NotImplementedError

    def _observation(self, env_states):
        # return super()._observation(env_states) --- IGNORE ---
        pass

    def _privileged_observation(self, env_states):
        # return super()._privileged_observation(env_states) --- IGNORE ---
        pass

    # ------------------------------------------------------------------ #
    # Observation utilities
    # ------------------------------------------------------------------ #
    @property
    def obs_buf(self) -> torch.Tensor:
        """Stacked observation buffer with history along features."""
        if self.obs_buf_queue is None or len(self.obs_buf_queue) == 0:
            raise RuntimeError("Observation buffer not initialized.")
        return torch.cat(list(self.obs_buf_queue), dim=1)

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Stacked privileged observation buffer with history along features."""
        if self.priv_obs_buf_queue is None or len(self.priv_obs_buf_queue) == 0:
            raise RuntimeError("Privileged observation buffer not initialized.")
        return torch.cat(list(self.priv_obs_buf_queue), dim=1)

    @property
    def default_env_states(self) -> TensorState:
        """Initial environment states used for resets."""
        return self._default_env_states
