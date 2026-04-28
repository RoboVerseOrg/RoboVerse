from __future__ import annotations

from typing import TYPE_CHECKING

from metasim.protocol_sim.core.interfaces import SimulatorAdapter
from metasim.types import CompatActionInput, TensorState

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class MetaSimProtocolAdapter(SimulatorAdapter):
    """Adapter from the protocol server surface to a MetaSim handler."""

    def __init__(self, handler: BaseSimHandler):
        self._handler = handler

    @property
    def handler(self) -> BaseSimHandler:
        """Return the wrapped MetaSim handler."""
        return self._handler

    def read_state(self) -> TensorState:
        """Read tensor state from the wrapped handler."""
        state = self._handler.get_states(mode="tensor")
        if state is None:
            raise RuntimeError("MetaSim handler returned no state.")
        if not isinstance(state, TensorState):
            raise TypeError(f"Expected TensorState from handler, got {type(state)!r}.")
        return state

    def apply_action(self, action: CompatActionInput) -> None:
        """Forward an action to the wrapped handler."""
        self._handler.set_dof_targets(action)

    def step(self) -> None:
        """Advance the wrapped handler by one simulation step."""
        self._handler.simulate()

    def close(self) -> None:
        """Close the wrapped handler."""
        self._handler.close()
