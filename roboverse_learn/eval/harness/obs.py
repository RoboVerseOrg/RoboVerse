"""Typed, torch-native, num_envs-major carriers for observations and actions.

Obs and actions stay as batched device tensors rather than per-env numpy dicts. That
is what lets the in-process transport be zero-copy and the runner be vectorized: the
batch dimension is first-class, never reconstructed from a Python list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .spec import ActionSpec, ObsSpec


@dataclass
class ObsBatch:
    """Observation for a subset of envs. ``tensors[key]`` has shape ``(B, *field.shape)``."""

    spec: ObsSpec
    env_ids: torch.Tensor  # (B,) global env indices this batch covers
    tensors: dict[str, torch.Tensor]
    task: dict[str, Any] = field(default_factory=dict)  # non-tensor payload (language/goal)

    @property
    def batch_size(self) -> int:
        return int(self.env_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.env_ids.device

    def index(self, mask: torch.Tensor) -> ObsBatch:
        """Sub-select rows by a boolean or index mask over the batch dimension."""
        return ObsBatch(self.spec, self.env_ids[mask], {k: v[mask] for k, v in self.tensors.items()}, dict(self.task))

    def to(self, device: torch.device | str) -> ObsBatch:
        return ObsBatch(
            self.spec, self.env_ids.to(device), {k: v.to(device) for k, v in self.tensors.items()}, dict(self.task)
        )

    def validate(self) -> ObsBatch:
        """Assert every required field is present with the declared per-env shape."""
        b = self.batch_size
        for f in self.spec.fields:
            if f.key not in self.tensors:
                if f.required:
                    raise KeyError(f"ObsBatch missing required field {f.key!r}")
                continue
            got = tuple(self.tensors[f.key].shape)
            if got != (b, *f.shape):
                raise ValueError(f"field {f.key!r} shape {got} != expected {(b, *f.shape)}")
        return self


@dataclass
class ActionBatch:
    """Action for a subset of envs. ``tensors[key]`` is ``(B, *shape)`` or ``(B, H, *shape)``."""

    spec: ActionSpec
    env_ids: torch.Tensor
    tensors: dict[str, torch.Tensor]

    @property
    def batch_size(self) -> int:
        return int(self.env_ids.shape[0])

    @property
    def is_chunked(self) -> bool:
        """True if tensors carry a horizon dim ``(B, H, ...)`` matching ``spec.chunk_len``."""
        if self.spec.chunk_len <= 1 or not self.tensors:
            return False
        any_key = self.spec.fields[0].key
        return self.tensors[any_key].dim() == len(self.spec.field(any_key).shape) + 2

    def validate(self) -> ActionBatch:
        b = self.batch_size
        for f in self.spec.fields:
            if f.key not in self.tensors:
                raise KeyError(f"ActionBatch missing field {f.key!r}")
            t = self.tensors[f.key]
            expected_flat = (b, *f.shape)
            expected_chunk = (b, self.spec.chunk_len, *f.shape)
            if tuple(t.shape) not in (expected_flat, expected_chunk):
                raise ValueError(
                    f"action field {f.key!r} shape {tuple(t.shape)} != {expected_flat} or {expected_chunk}"
                )
        return self
