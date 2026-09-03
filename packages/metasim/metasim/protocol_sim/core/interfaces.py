from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from metasim.types import CompatActionInput, TensorState


class Transport(ABC):
    """Asynchronous message I/O boundary for an external controller."""

    @abstractmethod
    def start(self) -> None:  # pragma: no cover
        """Start the transport layer."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:  # pragma: no cover
        """Close the transport layer."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_command(self) -> Any | None:  # pragma: no cover
        """Return the latest inbound command message, or None before any message arrives."""
        raise NotImplementedError

    def get_latest_command_with_token(self) -> tuple[Any | None, Any | None]:
        """Return the latest command plus a token for new-message detection.

        Transports that reuse one mutable message object should override this and
        return a monotonic callback counter or sequence number.
        """
        msg = self.get_latest_command()
        return msg, (id(msg) if msg is not None else None)

    @abstractmethod
    def publish(self, channel: str, msg: Any) -> None:  # pragma: no cover
        """Publish an outbound message on a named channel."""
        raise NotImplementedError


class ProtocolCodec(ABC):
    """Convert between transport messages and MetaSim-facing protocol data."""

    @abstractmethod
    def decode_command(self, msg: Any) -> Any:  # pragma: no cover
        """Decode one inbound transport message."""
        raise NotImplementedError

    @abstractmethod
    def encode_messages(self, state: TensorState, *, sim_time_s: float) -> dict[str, Any]:  # pragma: no cover
        """Encode outbound transport messages from the current MetaSim state."""
        raise NotImplementedError


class CommandAdapter(ABC):
    """Map decoded protocol commands to MetaSim actions."""

    @abstractmethod
    def to_action(self, command: Any, state: TensorState) -> CompatActionInput | None:  # pragma: no cover
        """Return a MetaSim action for the decoded command, or None to step without action changes."""
        raise NotImplementedError


class SimulatorAdapter(ABC):
    """Small simulator surface needed by the protocol server."""

    @abstractmethod
    def read_state(self) -> TensorState:  # pragma: no cover
        """Read current simulator state."""
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: CompatActionInput) -> None:  # pragma: no cover
        """Apply a MetaSim action to the simulator."""
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:  # pragma: no cover
        """Advance the simulator by one handler step."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:  # pragma: no cover
        """Close simulator resources."""
        raise NotImplementedError
