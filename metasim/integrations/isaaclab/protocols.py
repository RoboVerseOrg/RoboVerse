from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from metasim.scenario.scenario import ScenarioCfg

from .runtime import IsaacLabRuntimeContext


@runtime_checkable
class IsaacLabTaskProtocol(Protocol):
    """Minimal protocol for IsaacLab-style tasks used in MetaSim.

    This protocol is intentionally small to keep required changes to external IsaacLab
    task snippets minimal. Tasks may optionally accept a `runtime` argument to attach
    to a shared Isaac Sim app instance.
    """

    # common MetaSim entrypoints pass these three values
    def __init__(
        self,
        scenario: ScenarioCfg,
        args: Any,
        device: Any | None = None,
        *,
        runtime: IsaacLabRuntimeContext | None = None,
        **kwargs: Any,
    ) -> None: ...
