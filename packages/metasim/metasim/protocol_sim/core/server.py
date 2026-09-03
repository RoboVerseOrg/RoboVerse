from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from metasim.protocol_sim.core.interfaces import CommandAdapter, ProtocolCodec, SimulatorAdapter, Transport


@dataclass(frozen=True)
class ProtocolServerConfig:
    """Configuration for `ProtocolServer`."""

    dt: float
    realtime: bool = True

    def __post_init__(self) -> None:
        if float(self.dt) <= 0.0:
            raise ValueError("ProtocolServerConfig.dt must be positive.")


class ProtocolServer:
    """Run a MetaSim simulator against an asynchronous external command stream."""

    def __init__(
        self,
        *,
        sim: SimulatorAdapter,
        transport: Transport,
        codec: ProtocolCodec,
        command_adapter: CommandAdapter,
        config: ProtocolServerConfig,
    ) -> None:
        self._sim = sim
        self._transport = transport
        self._codec = codec
        self._command_adapter = command_adapter
        self._config = config

        self._sim_time_s = 0.0
        self._last_command_token: Any | None = None
        self._last_decoded_command: Any | None = None
        self._started = False

    @property
    def sim_time_s(self) -> float:
        """Current simulation time tracked by the protocol server."""
        return self._sim_time_s

    def start(self) -> None:
        """Start the external transport."""
        self._transport.start()
        self._started = True

    def close(self) -> None:
        """Close transport and simulator resources."""
        try:
            self._transport.close()
        finally:
            self._sim.close()
            self._started = False

    def _refresh_command(self) -> None:
        raw_msg, token = self._transport.get_latest_command_with_token()
        if raw_msg is None:
            return
        if token is None:
            token = id(raw_msg)
        if token == self._last_command_token:
            return
        self._last_command_token = token
        self._last_decoded_command = self._codec.decode_command(raw_msg)

    def step_once(self) -> None:
        """Run one protocol/simulator tick."""
        state_before = self._sim.read_state()
        self._refresh_command()

        if self._last_decoded_command is not None:
            action = self._command_adapter.to_action(self._last_decoded_command, state_before)
            if action is not None:
                self._sim.apply_action(action)

        self._sim.step()
        self._sim_time_s += float(self._config.dt)

        state_after = self._sim.read_state()
        for channel, msg in self._codec.encode_messages(state_after, sim_time_s=self._sim_time_s).items():
            self._transport.publish(channel, msg)

    def run_forever(self, *, max_steps: int | None = None) -> None:
        """Run the protocol loop until interrupted or `max_steps` is reached."""
        next_wall = time.perf_counter()
        steps = 0
        while max_steps is None or steps < max_steps:
            self.step_once()
            steps += 1

            if self._config.realtime:
                next_wall += float(self._config.dt)
                sleep_s = max(0.0, next_wall - time.perf_counter())
                if sleep_s:
                    time.sleep(sleep_s)
