from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from roboverse_pack.robot_protocols.core.types import CanonicalRobotCommand, SimRobotObservation


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
