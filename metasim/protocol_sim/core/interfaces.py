from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from metasim.protocol_sim.core.types import CanonicalRobotCommand, SimRobotObservation


class Transport(ABC):
    """Message transport layer (DDS/ROS2/LCM/UDP/etc.).

    A transport is responsible only for moving typed messages in/out of the process.
    Encoding/decoding semantic meaning is handled by a ProtocolCodec.
    """

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
        """Return the most recent command message, or None if nothing received yet."""
        raise NotImplementedError

    def get_latest_command_with_token(self) -> tuple[Any | None, int | None]:
        """Return latest command plus a monotonic/unique token for new-message detection.

        Transports that may reuse the same Python message object across callbacks should
        override this and provide a callback-driven token (e.g., sequence counter).
        """
        msg = self.get_latest_command()
        return msg, (id(msg) if msg is not None else None)

    @abstractmethod
    def publish(self, channel: str, msg: Any) -> None:  # pragma: no cover
        """Publish a message on a named channel/topic."""
        raise NotImplementedError


class ProtocolCodec(ABC):
    """Protocol semantics: map between vendor messages and canonical command/state."""

    @abstractmethod
    def decode_command(self, msg: Any) -> CanonicalRobotCommand:  # pragma: no cover
        """Decode a vendor-specific message into a canonical command."""
        raise NotImplementedError

    @abstractmethod
    def encode_messages(self, obs: SimRobotObservation, *, sim_time_s: float) -> dict[str, Any]:  # pragma: no cover
        """Encode outbound messages for the given observation."""
        raise NotImplementedError


class ActuationModel(ABC):
    """Convert a canonical command into simulator actuation (typically joint torques)."""

    @abstractmethod
    def compute_effort(self, cmd: CanonicalRobotCommand, obs: SimRobotObservation) -> np.ndarray:  # pragma: no cover
        """Return effort vector in *sorted simulator joint order*."""
        raise NotImplementedError


class StandbyController(ABC):
    """Compute a safe effort when no external controller is active."""

    @abstractmethod
    def compute_effort(self, obs: SimRobotObservation) -> np.ndarray:  # pragma: no cover
        """Return effort vector in *sorted simulator joint order*."""
        raise NotImplementedError


class ExternalAssist(ABC):
    """Apply extra forces/constraints that help keep the robot safe during bringup."""

    @abstractmethod
    def apply(self, obs: SimRobotObservation, *, dt: float) -> None:  # pragma: no cover
        """Apply the assist force/constraint."""
        raise NotImplementedError

    def start_release(self) -> None:
        """Begin ramping the assist off (called when protocol control is stable)."""
        return
