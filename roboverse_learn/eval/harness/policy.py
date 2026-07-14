"""The Policy contract — the one thing a policy author implements.

A policy is a single pure ``act(ObsBatch) -> ActionBatch``. Because obs and actions
are typed, batched carriers, one ``act`` call serves all ``B`` active envs at once and
needs no per-arm key convention; a single-arm, single-env, single-step policy is just
the ``B=1, chunk_len=1`` special case, not a separate code path.

Chunking/temporal-ensembling is NOT the policy's concern: a policy declares
``ActionSpec.chunk_len`` and returns the chunk; the runner schedules/ensembles it
(:mod:`.chunking`). Policies with a stateful ``update_obs``/``get_action`` surface are
supported by writing a thin adapter, rather than by widening this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .obs import ActionBatch, ObsBatch
from .spec import ActionSpec, ObsSpec


@dataclass(frozen=True)
class PolicyCard:
    """A policy's static self-description, used in connect-time negotiation."""

    name: str
    needs_obs: ObsSpec  # the fields this policy consumes
    produces_action: ActionSpec  # what it emits (incl. chunk_len, control)
    device: str = "cpu"


@runtime_checkable
class Policy(Protocol):
    """A policy adapter. Batched and chunk-aware by construction.

    Four methods are the whole contract: ``describe``, ``bind``, ``reset``, ``act``. An
    optional ``close()`` is called once by :func:`~._evaluate.evaluate` when the rollout ends
    (a remote policy uses it to drop its socket).
    """

    def describe(self) -> PolicyCard:
        """Static self-description (obs it needs, action it produces)."""

    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        """Receive the concrete specs the harness derived for this task/robot.

        Called once before the rollout, after negotiation. A policy that advertises empty
        specs in :meth:`describe` learns the real ones here; build any key mapping now, not
        per step.
        """

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear per-env state for ``env_ids`` (partial reset on termination)."""

    def act(self, obs: ObsBatch) -> ActionBatch:
        """Map an ObsBatch (B active envs) to an ActionBatch.

        If ``describe().produces_action.chunk_len > 1``, the returned tensors carry a
        ``(B, chunk_len, ...)`` chunk; the runner handles scheduling/ensembling.
        """
