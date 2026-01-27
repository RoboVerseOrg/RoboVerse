from __future__ import annotations

import threading
from typing import Any

from roboverse_pack.robot_protocols.core.interfaces import Transport
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.profile import UnitreeRobotProfile


class UnitreeSdk2DdsTransport(Transport):
    """DDS transport using unitree_sdk2py."""

    def __init__(self, *, domain_id: int, iface: str, profile: UnitreeRobotProfile):
        self._domain_id = int(domain_id)
        self._iface = str(iface)
        self._profile = profile

        self._lock = threading.Lock()
        self._latest_cmd = None

        self._publishers: dict[str, Any] = {}

    def start(self) -> None:
        """Initialize and start the DDS transport."""
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "unitree_sdk2py is required for UnitreeSdk2DdsTransport. "
                "Install unitree_sdk2_python or activate the appropriate environment."
            ) from exc

        ChannelFactoryInitialize(self._domain_id, self._iface)

        # LowCmd subscriber (type depends on msg_type).
        if self._profile.msg_type == "hg":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdT
        elif self._profile.msg_type == "go":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdT
        else:  # pragma: no cover
            raise ValueError(f"Unknown msg_type: {self._profile.msg_type}")

        def _on_lowcmd(msg: LowCmdT):
            with self._lock:
                self._latest_cmd = msg

        sub = ChannelSubscriber(self._profile.lowcmd_topic, LowCmdT)
        sub.Init(_on_lowcmd, 10)
        self._sub = sub

        # LowState publisher (type depends on msg_type).
        if self._profile.msg_type == "hg":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateT
        else:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateT

        pub_low = ChannelPublisher(self._profile.lowstate_topic, LowStateT)
        pub_low.Init()
        self._publishers[self._profile.lowstate_topic] = pub_low

        pub_sport = ChannelPublisher(self._profile.sportstate_topic, SportModeState_)
        pub_sport.Init()
        self._publishers[self._profile.sportstate_topic] = pub_sport

        # Message factories (used by codec): store on transport for convenience.
        self._sportstate_factory = unitree_go_msg_dds__SportModeState_

        if self._profile.msg_type == "hg":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
        else:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowStateDefault
        self._lowstate_factory = LowStateDefault

    @property
    def lowstate_factory(self):
        """Get the factory for creating LowState messages."""
        return getattr(self, "_lowstate_factory", None)

    @property
    def sportstate_factory(self):
        """Get the factory for creating SportModeState messages."""
        return getattr(self, "_sportstate_factory", None)

    def close(self) -> None:
        """Close the DDS transport."""
        # unitree_sdk2py does not expose an explicit shutdown for subscribers/publishers.
        return

    def get_latest_command(self) -> Any | None:
        """Get the most recent LowCmd message received."""
        with self._lock:
            return self._latest_cmd

    def publish(self, channel: str, msg: Any) -> None:
        """Publish a message to a specific DDS channel."""
        pub = self._publishers.get(channel)
        if pub is None:  # pragma: no cover
            raise KeyError(f"No publisher registered for channel '{channel}'")
        pub.Write(msg)
