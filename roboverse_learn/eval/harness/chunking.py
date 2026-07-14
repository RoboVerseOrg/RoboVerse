# Copyright (c) 2023 Tony Z. Zhao
# SPDX-License-Identifier: MIT
#
# Adapted from ACT (https://github.com/tonyzhaozh/act).
# Changes: ACT's per-step temporal-aggregation loop (a single env, numpy, `all_time_actions`
#   allocated over the whole episode) is reimplemented against RoboVerse's typed carriers —
#   vectorized over num_envs and over the typed ActionSpec fields, torch instead of numpy, and
#   backed by a ring buffer over the chunk horizon instead of an episode-length buffer. The
#   weighting deliberately differs from ACT: we weight exp(+k * rank) (newest prediction highest)
#   to match roboverse_learn.il's get_temporal_agg_action, whereas ACT uses exp(-k * i) (oldest
#   highest) — see the note below. Populated predictions are tracked with an explicit mask rather
#   than an `actions != 0` test.
# Full license: roboverse_learn/il/policies/act/LICENSE
"""Action chunking + temporal ensembling — first-class in the contract.

Chunking is expressed via ``ActionSpec.chunk_len`` and handled by ONE vectorized
implementation, so every policy — local or remote — gets identical timing instead of
each adapter re-deciding how to unroll or blend its chunks.

:class:`TemporalEnsembler` is the exp-weighted average of all predictions targeting the
current step, generalized over ``num_envs`` and the typed action fields. It is
numerically matched (to ~5e-07) against
``roboverse_learn.il.runners.base_eval_runner.get_temporal_agg_action``, which is the
behaviour it is meant to preserve.

.. note::
   Both this and the ``il`` implementation weight by ``exp(+k * rank)``, i.e. the
   *newest* prediction for a step gets the largest weight. The original ALOHA/ACT
   temporal aggregation uses ``exp(-k * i)`` and weights the *oldest* prediction
   highest. We match ``il`` (deliberately, for parity with existing checkpoints), not
   upstream ACT — do not describe this as the ACT scheme.

Memory is bounded by the chunk horizon, not the episode: only a prediction made in the
last ``chunk_len`` steps can target the current step, so the buffer is a per-env ring of
``chunk_len`` chunks (``O(num_envs * chunk_len^2 * dim)``), not ``O(num_envs *
episode_len^2 * dim)``.

:class:`ChunkScheduler` is the non-ensemble path: cache a chunk, emit one action per
step, re-query at the horizon boundary — per-env, so an env that resets mid-chunk
re-queries independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .obs import ActionBatch
from .spec import ActionSpec


def _dim(shape: tuple[int, ...]) -> int:
    d = 1
    for s in shape:
        d *= s
    return d


@dataclass
class ActionChunk:
    """A horizon of actions per env: ``tensors[key]`` is ``(B, H, *field.shape)``."""

    tensors: dict[str, torch.Tensor]
    horizon: int

    @classmethod
    def from_batch(cls, action: ActionBatch) -> ActionChunk:
        h = action.spec.chunk_len
        if action.is_chunked:
            return cls(dict(action.tensors), h)
        # a flat (B, *shape) action is a horizon-1 chunk
        return cls({k: v.unsqueeze(1) for k, v in action.tensors.items()}, 1)


class TemporalEnsembler:
    """Vectorized temporal ensembling over ``num_envs`` and typed fields (ring-buffered).

    ``_buf[key][env, slot, offset]`` holds the action a chunk pushed at step ``t`` (occupying
    ``slot = t % H``) predicted for step ``t + offset``. A query at step ``s`` gathers, for each
    ``offset j``, the chunk pushed at ``t = s - j`` — the only predictions that can target ``s``.
    """

    def __init__(
        self,
        *,
        action_spec: ActionSpec,
        num_envs: int,
        k: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        self.spec = action_spec
        self.H = max(1, action_spec.chunk_len)
        self.num_envs = num_envs
        self.k = k
        self.device = torch.device(device)
        self._dims = {f.key: _dim(f.shape) for f in action_spec.fields}
        h = self.H
        self._buf = {key: torch.zeros(num_envs, h, h, d, device=self.device) for key, d in self._dims.items()}
        # step at which the chunk in each ring slot was pushed (-1 = empty), and its valid length
        self._pushed_at = torch.full((num_envs, h), -1, dtype=torch.long, device=self.device)
        self._len = torch.zeros(num_envs, h, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._pushed_at.fill_(-1)
            self._len.zero_()
        else:
            ids = env_ids.to(self._pushed_at.device)
            self._pushed_at[ids] = -1
            self._len[ids] = 0

    def push(self, step: int, chunk: ActionChunk, env_ids: torch.Tensor) -> None:
        """Store a chunk predicted at ``step`` for envs ``env_ids``."""
        ids = env_ids.to(self.device)
        slot = step % self.H
        h = min(chunk.horizon, self.H)
        for key, dim in self._dims.items():
            a = chunk.tensors[key].reshape(len(ids), chunk.horizon, dim).to(self.device)
            self._buf[key][ids, slot, :h] = a[:, :h]
        self._pushed_at[ids, slot] = step
        self._len[ids, slot] = h

    def action_for(self, step: int, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exp-weighted average of all predictions targeting ``step`` for ``env_ids``.

        Uses the explicit ``_pushed_at``/``_len`` bookkeeping to select populated predictions,
        which is an **intended improvement** over il's ``all(actions != 0)`` populated-hack: a
        genuinely-zero action stays populated, and envs are decoupled (il's ``all(..., dim=envs)``
        let one env's zeros mask every env). Numerically identical to il on nonzero,
        per-env-aligned inputs (regression-tested); it only diverges on exactly the cases il gets
        wrong.
        """
        ids = env_ids.to(self.device)
        b, h = len(ids), self.H
        # offsets j = H-1 .. 0  =>  prediction steps t = step - j ascending (oldest -> newest),
        # which is the order il ranks predictions in.
        offs = torch.arange(h - 1, -1, -1, device=self.device)  # (H,)
        t_pred = step - offs  # (H,)
        slots = t_pred % h
        valid = (t_pred >= 0) & (self._pushed_at[ids][:, slots] == t_pred) & (self._len[ids][:, slots] > offs)
        mask = valid.float()  # (B, H)
        rank = mask.cumsum(dim=1) - 1.0  # rank among populated, prediction-time order
        w = torch.exp(self.k * rank) * mask
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        out: dict[str, torch.Tensor] = {}
        for key in self._dims:
            preds = self._buf[key][ids[:, None], slots[None, :], offs[None, :]]  # (B, H, dim)
            out[key] = (preds * w.unsqueeze(-1)).sum(dim=1).reshape(b, *self.spec.field(key).shape)
        return out


class ChunkScheduler:
    """Non-ensemble chunk cache: emit one action per step, re-query at the horizon."""

    def __init__(self, *, action_spec: ActionSpec, num_envs: int, device: torch.device | str = "cpu") -> None:
        self.spec = action_spec
        self.H = max(1, action_spec.chunk_len)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._chunk: dict[str, torch.Tensor] = {}
        self._ptr = torch.full((num_envs,), self.H, dtype=torch.long, device=self.device)
        # valid length of each env's cached chunk (0 => never filled / needs query). A chunk
        # shorter than H only fills ``_len`` steps, so re-query fires at the real tail — not
        # at H, which would emit stale data from a previous (longer) chunk.
        self._len = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def needs_query(self) -> torch.Tensor:
        """Env indices whose cached chunk is exhausted and need a fresh prediction."""
        return torch.nonzero(self._ptr >= self._len, as_tuple=False).flatten()

    def push(self, chunk: ActionChunk, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(self.device)
        h = min(chunk.horizon, self.H)
        for key in self._dims_keys():
            a = chunk.tensors[key].to(self.device)
            if key not in self._chunk:
                shape = (self.num_envs, self.H, *self.spec.field(key).shape)
                self._chunk[key] = torch.zeros(shape, device=self.device)
            self._chunk[key][ids, :h] = a[:, :h]
        self._ptr[ids] = 0
        self._len[ids] = h

    def action_for(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self._chunk:
            raise RuntimeError("ChunkScheduler.action_for called before any push (service needs_query first)")
        ids = env_ids.to(self.device)
        ptr = self._ptr[ids].clamp_max(self.H - 1)
        out = {key: self._chunk[key][ids, ptr] for key in self._dims_keys()}
        self._ptr[ids] = self._ptr[ids] + 1
        return out

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._ptr.fill_(self.H)
            self._len.zero_()
        else:
            ids = env_ids.to(self.device)
            self._ptr[ids] = self.H
            self._len[ids] = 0

    def _dims_keys(self) -> tuple[str, ...]:
        return self.spec.keys()
