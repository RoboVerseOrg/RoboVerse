"""A FastTD3 checkpoint as a ``metasim.bench`` policy.

``python -m metasim bench run --task <task> --sim <sim> \\
    --policy roboverse_learn.rl.fast_td3.bench_policy:FastTD3Policy --checkpoint runs/<task>/final.pt``

scores the actor the way ``evaluate.py`` drives it (``actor(obs_normalizer(obs))``), under the
benchmark's protocol and result schema instead of the trainer's own loop.
"""

from __future__ import annotations

from typing import Any

import torch

from metasim.bench import PolicyContext, TaskInstance
from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization


class FastTD3Policy:
    """The deterministic FastTD3 actor of a checkpoint, built on the first observation it sees.

    The checkpoint (``evaluate.py`` / ``train.py`` format) carries the actor weights, the observation
    normalizer and the training ``config`` (hidden size, init scale). The observation width is read from
    the first observation and the action width from the context, so the same checkpoint evaluates with
    any ``num_envs``.
    """

    name = "fast_td3"

    def __init__(self, checkpoint: str, device: str | torch.device | None = None) -> None:
        self.checkpoint = str(checkpoint)
        self._payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if "actor_state_dict" not in self._payload:
            raise ValueError(f"{self.checkpoint} is not a FastTD3 checkpoint: it has no 'actor_state_dict'")
        self._config: dict[str, Any] = dict(self._payload.get("config", {}) or {})
        self._device = torch.device(device) if device is not None else None
        self._actor: Actor | None = None
        self._normalizer: EmpiricalNormalization | None = None

    @property
    def config(self) -> dict[str, Any]:
        """The training configuration stored in the checkpoint (task, sim, robots, network sizes)."""
        return dict(self._config)

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """A feed-forward actor keeps no episode state."""

    def _build(self, n_obs: int, ctx: PolicyContext) -> None:
        device = self._device or ctx.device
        actor = Actor(
            n_obs=n_obs,
            n_act=ctx.action_dim,
            num_envs=ctx.num_envs,
            device=device,
            init_scale=float(self._config.get("init_scale", 0.1)),
            hidden_dim=int(self._config.get("actor_hidden_dim", 256)),
        )
        # ``noise_scales`` is the per-env exploration noise of training; it is sized by the trainer's num_envs
        # and never read by the deterministic forward pass.
        state = {k: v for k, v in self._payload["actor_state_dict"].items() if not k.startswith("noise_scales")}
        shape = f"observation width {n_obs}, action width {ctx.action_dim}, hidden {self._config.get('actor_hidden_dim', 256)}"
        try:
            missing, unexpected = actor.load_state_dict(state, strict=False)
        except RuntimeError as err:  # a size mismatch: the checkpoint was trained for another task
            raise ValueError(f"{self.checkpoint} does not fit this task ({shape}): {err}") from err
        missing = [k for k in missing if not k.startswith("noise_scales")]
        if missing or unexpected:
            raise ValueError(
                f"{self.checkpoint} does not fit this task ({shape}): missing {missing}, unexpected {list(unexpected)}."
            )
        normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        obs_state = self._payload.get("obs_normalizer_state")
        if obs_state:
            normalizer.load_state_dict(obs_state)
        actor.eval()
        normalizer.eval()
        self._actor, self._normalizer = actor.to(device), normalizer.to(device)

    @torch.no_grad()
    def act(self, obs: Any, ctx: PolicyContext) -> torch.Tensor:
        """``actor(normalizer(obs))`` for the flat observation tensor an ``RLTaskEnv`` returns."""
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"FastTD3Policy expects a flat observation tensor (an RLTaskEnv), got {type(obs).__name__}")
        if self._actor is None:
            self._build(int(obs.shape[-1]), ctx)
        device = next(self._actor.parameters()).device
        actions = self._actor(self._normalizer(obs.to(device)))
        return actions.float().to(ctx.device)
