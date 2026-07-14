"""Transport abstraction — how an ObsBatch->ActionBatch call crosses a process boundary.

The runner only ever sees a :class:`PolicyHandle` (which satisfies the ``Policy`` protocol),
so it is oblivious to whether the real policy runs in this process (``InProcessTransport``,
zero-copy), in a local subprocess, or on a remote machine (``ws``). This is the mechanism
that lets a policy with conflicting dependencies run in its own environment while the
simulator runs here, with the typed spec carried across the boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ..obs import ActionBatch, ObsBatch
from ..policy import Policy, PolicyCard
from ..spec import ActionSpec, ObsSpec


@runtime_checkable
class Transport(Protocol):
    def connect(self) -> None: ...
    def describe(self) -> PolicyCard: ...
    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None: ...
    def reset(self, env_ids: torch.Tensor) -> None: ...
    def infer(self, obs: ObsBatch) -> ActionBatch: ...
    def close(self) -> None: ...


class InProcessTransport:
    """The policy object lives here; calls go straight through (zero serialization)."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def connect(self) -> None:
        pass

    def describe(self) -> PolicyCard:
        return self.policy.describe()

    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        self.policy.bind(obs_spec, action_spec)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.policy.reset(env_ids)

    def infer(self, obs: ObsBatch) -> ActionBatch:
        return self.policy.act(obs)

    def close(self) -> None:
        closer = getattr(self.policy, "close", None)
        if callable(closer):
            closer()


class PolicyHandle:
    """Presents the ``Policy`` protocol to the runner, delegating to any Transport."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        transport.connect()

    def describe(self) -> PolicyCard:
        return self.transport.describe()

    def bind(self, obs_spec: ObsSpec, action_spec: ActionSpec) -> None:
        self.transport.bind(obs_spec, action_spec)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.transport.reset(env_ids)

    def act(self, obs: ObsBatch) -> ActionBatch:
        return self.transport.infer(obs)

    def close(self) -> None:
        self.transport.close()
