from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from roboverse_pack.robot_protocols.core.interfaces import ActuationModel, ProtocolCodec, Transport
from roboverse_pack.robot_protocols.core.sim_adapter import MetaSimAdapter


@dataclass
class ServerConfig:
    """Configuration for the robot protocol server."""

    dt: float
    realtime: bool = True


class RobotProtocolServer:
    """Generic, single-robot protocol emulation server."""

    def __init__(
        self,
        *,
        adapter: MetaSimAdapter,
        transport: Transport,
        codec: ProtocolCodec,
        actuation: ActuationModel,
        config: ServerConfig,
    ) -> None:
        self._adapter = adapter
        self._transport = transport
        self._codec = codec
        self._actuation = actuation
        self._config = config

        self._sim_time_s = 0.0
        self._last_cmd = None

    @property
    def sim_time_s(self) -> float:
        """Get the current simulation time in seconds."""
        return self._sim_time_s

    def start(self) -> None:
        """Start the server."""
        self._transport.start()

    def close(self) -> None:
        """Close the server and cleanup resources."""
        try:
            self._transport.close()
        finally:
            self._adapter.close()

    def run_forever(self) -> None:
        """Run the server loop indefinitely."""
        # Keep wall-time pacing in this loop to match unitree_mujoco behavior.
        next_wall = time.perf_counter()
        dt = float(self._config.dt)

        while True:
            obs = self._adapter.read_observation()

            msg = self._transport.get_latest_command()
            if msg is not None:
                self._last_cmd = msg

            if self._last_cmd is None:
                # No command yet: hold zero effort.
                effort = np.zeros((len(obs.joint_names_sorted),), dtype=np.float32)
            else:
                cmd = self._codec.decode_command(self._last_cmd)
                effort = self._actuation.compute_effort(cmd, obs)

            self._adapter.apply_effort(effort)
            self._adapter.step()

            self._sim_time_s += dt
            obs_after = self._adapter.read_observation()
            out = self._codec.encode_messages(obs_after, sim_time_s=self._sim_time_s)
            for channel, out_msg in out.items():
                self._transport.publish(channel, out_msg)

            if self._config.realtime:
                next_wall += dt
                sleep_s = max(0.0, next_wall - time.perf_counter())
                if sleep_s:
                    time.sleep(sleep_s)
