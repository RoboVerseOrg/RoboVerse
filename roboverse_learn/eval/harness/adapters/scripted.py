"""Dependency-free baseline policies — smoke tests, sanity baselines, negotiation examples.

All are embodiment-agnostic: they adapt to whatever action spec the harness derived
(via :meth:`bind`), so the same policy runs on a single-arm Franka, a bimanual ALOHA, or
a humanoid with no per-robot code. Obs and action share one canonical key scheme, so a
hold-pose policy is a one-line echo.
"""

from __future__ import annotations

import torch

from ..obs import ActionBatch, ObsBatch
from ..policy import PolicyCard
from ..spec import ActionSpec, ObsSpec, Space


class _BoundPolicy:
    """Base: learns the derived ObsSpec/ActionSpec via bind()."""

    name = "base"

    def __init__(self) -> None:
        self._aspec: ActionSpec | None = None

    def describe(self) -> PolicyCard:
        return PolicyCard(self.name, ObsSpec(()), ActionSpec((), chunk_len=1))

    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        self._aspec = action_spec

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def _empty(self, obs: ObsBatch):
        assert self._aspec is not None, "bind() must be called before act()"
        return {f: torch.zeros(obs.batch_size, *self._aspec.field(f).shape) for f in self._aspec.keys()}


class ZeroActionPolicy(_BoundPolicy):
    """Commands all-zero targets (arm to zero joint pos, gripper closed)."""

    name = "zero"

    def act(self, obs: ObsBatch) -> ActionBatch:
        return ActionBatch(self._aspec, obs.env_ids, self._empty(obs))


class HoldPosePolicy(_BoundPolicy):
    """Commands each chain to hold its current joint pose (echoes the joint-space obs)."""

    name = "hold_pose"

    def act(self, obs: ObsBatch) -> ActionBatch:
        assert self._aspec is not None, "bind() must be called before act()"
        tensors = {}
        for f in self._aspec.fields:
            tensors[f.key] = obs.tensors[f.key] if f.key in obs.tensors else torch.zeros(obs.batch_size, *f.shape)
        return ActionBatch(self._aspec, obs.env_ids, tensors)


class RandomPolicy(_BoundPolicy):
    """Uniform-random targets in [low, high] per field (joints in [-1,1], ee_pose small)."""

    name = "random"

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def act(self, obs: ObsBatch) -> ActionBatch:
        assert self._aspec is not None, "bind() must be called before act()"
        tensors = {}
        for f in self._aspec.fields:
            hi = 0.05 if f.space == Space.EE_POSE else self.scale
            tensors[f.key] = (torch.rand(obs.batch_size, *f.shape) * 2 - 1) * hi
        return ActionBatch(self._aspec, obs.env_ids, tensors)
