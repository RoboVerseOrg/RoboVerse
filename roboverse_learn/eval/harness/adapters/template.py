"""Copy-me reference: the ~30-line shape of a real policy adapter.

A policy author implements this one class — the harness handles embodiment inference,
typed obs/action derivation, vectorized rollout, chunk scheduling, and multi-sim parity.
There is no per-policy eval-loop boilerplate to copy, no per-arm key convention (keys are
canonical and typed), and no manual batching (``ObsBatch`` is already batched).

Fill in ``__init__`` (load your checkpoint), ``bind`` (record the derived specs / build any
key mapping), and ``act`` (obs -> action). If your model outputs an action chunk, set
``chunk_len`` in :meth:`describe` and return ``(B, chunk_len, ...)`` tensors — the runner
schedules/ensembles them.
"""

from __future__ import annotations

import torch

from ..obs import ActionBatch, ObsBatch
from ..policy import PolicyCard
from ..spec import ActionSpec, ObsSpec


class TemplatePolicy:
    """Minimal skeleton — replace the body with your model's inference."""

    def __init__(self, checkpoint: str | None = None, *, chunk_len: int = 1) -> None:
        # load model / processors / checkpoint here
        self.model = None
        self._chunk_len = chunk_len
        self._aspec: ActionSpec | None = None

    def describe(self) -> PolicyCard:
        # advertise chunk_len (and, for real negotiation, the obs fields you consume)
        return PolicyCard("template", ObsSpec(()), ActionSpec((), chunk_len=self._chunk_len))

    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        # the harness tells you the concrete typed specs it derived for this task/robot;
        # build any obs-key -> model-input mapping here (once), not per step.
        self._aspec = action_spec

    def reset(self, env_ids: torch.Tensor) -> None:
        # clear per-env recurrent/history state for the given envs (partial reset)
        pass

    def act(self, obs: ObsBatch) -> ActionBatch:
        # obs.tensors is a dict of canonical-key -> (B, *shape) batched tensors on the sim
        # device; obs.env_ids are the active env indices. Run your model and return an
        # ActionBatch keyed by self._aspec's fields, shaped (B, *shape) or (B, chunk_len, *shape).
        assert self._aspec is not None, "bind() must be called before act()"
        tensors = {f.key: torch.zeros(obs.batch_size, *f.shape) for f in self._aspec.fields}
        return ActionBatch(self._aspec, obs.env_ids, tensors)
