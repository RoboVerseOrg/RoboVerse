"""PyRep handler — currently unmaintained.

The previous implementation predates the current ``BaseSimHandler``
contract: ``launch`` referenced attributes the base doesn't define,
``_set_states`` / ``_get_states`` dropped the ``env_ids`` parameter,
and ``_simulate`` was missing entirely. Rather than carry a handler
that fails opaquely on first use, this placeholder raises a clear
``NotImplementedError`` at construction.

``SimType.PYREP`` still dispatches here so the registry stays complete;
the abstract methods keep the ``BaseSimHandler`` signatures so a future
port has a working skeleton to fill in.
"""

from __future__ import annotations

from metasim.sim import BaseSimHandler
from metasim.types import CompatActionInput, DictStateBatch, TensorState

_UNMAINTAINED_MSG = (
    "PyrepHandler is currently unmaintained — it diverged from the "
    "BaseSimHandler contract and was never ported. Use a maintained "
    "backend (mujoco, isaacsim, sapien3, genesis, mjx, pybullet, blender, "
    "newton), or open an issue if you need PyRep support reinstated."
)


class PyrepHandler(BaseSimHandler):
    """Placeholder PyRep handler — raises ``NotImplementedError`` on construction."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(_UNMAINTAINED_MSG)

    # Abstract methods kept with their BaseSimHandler signatures so the
    # class stays concrete and a future port has a skeleton to fill in.

    def _set_states(self, states: TensorState | DictStateBatch, env_ids: list[int] | None = None) -> None:
        raise NotImplementedError(_UNMAINTAINED_MSG)

    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        raise NotImplementedError(_UNMAINTAINED_MSG)

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        raise NotImplementedError(_UNMAINTAINED_MSG)

    def _simulate(self) -> None:
        raise NotImplementedError(_UNMAINTAINED_MSG)

    def launch(self) -> None:
        raise NotImplementedError(_UNMAINTAINED_MSG)

    def close(self) -> None:
        return None
