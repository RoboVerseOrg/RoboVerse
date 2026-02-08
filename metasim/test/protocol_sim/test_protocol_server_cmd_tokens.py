from __future__ import annotations

import numpy as np
import pytest

from roboverse_pack.protocol_sim.core.interfaces import (
    ActuationModel,
    ProtocolCodec,
    StandbyController,
    Transport,
)
from roboverse_pack.protocol_sim.core.server import RobotProtocolServer, ServerConfig, StandbyConfig
from roboverse_pack.protocol_sim.core.types import CanonicalRobotCommand, SimRobotObservation


class _StopLoop(RuntimeError):
    pass


class _DummyMsg:
    def __init__(self):
        self.kp = np.zeros((1,), dtype=np.float32)
        self.kd = np.zeros((1,), dtype=np.float32)
        self.tau_ff = np.zeros((1,), dtype=np.float32)


class _DummyTransport(Transport):
    """Reuses one message object but advances a token each callback."""

    def __init__(self):
        self._msg = _DummyMsg()
        self._token = 0

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def get_latest_command(self):
        return self._msg

    def get_latest_command_with_token(self):
        # Mimic callback updates reusing the same Python object.
        self._msg.kp[0] = 10.0
        self._token += 1
        return self._msg, self._token

    def publish(self, channel: str, msg) -> None:
        return


class _DummyCodec(ProtocolCodec):
    def decode_command(self, msg) -> CanonicalRobotCommand:
        return CanonicalRobotCommand(
            joint_names=["j0"],
            q_des=np.zeros((1,), dtype=np.float32),
            qd_des=np.zeros((1,), dtype=np.float32),
            kp=np.array([float(msg.kp[0])], dtype=np.float32),
            kd=np.zeros((1,), dtype=np.float32),
            tau_ff=np.zeros((1,), dtype=np.float32),
            mode=np.ones((1,), dtype=np.int32),
        )

    def encode_messages(self, obs: SimRobotObservation, *, sim_time_s: float) -> dict[str, object]:
        return {}


class _DummyActuation(ActuationModel):
    def compute_effort(self, cmd: CanonicalRobotCommand, obs: SimRobotObservation) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)


class _DummyStandby(StandbyController):
    def compute_effort(self, obs: SimRobotObservation) -> np.ndarray:
        return np.array([-1.0], dtype=np.float32)


class _DummyAssist:
    def __init__(self):
        self.start_release_calls = 0

    def start_release(self) -> None:
        self.start_release_calls += 1

    def apply(self, obs: SimRobotObservation, *, dt: float) -> None:
        return


class _DummyAdapter:
    def __init__(self):
        self.applied_efforts: list[np.ndarray] = []
        self._obs = SimRobotObservation(
            joint_names_sorted=["j0"],
            q_sorted=np.zeros((1,), dtype=np.float32),
            dq_sorted=np.zeros((1,), dtype=np.float32),
            tau_sorted=None,
            root_state=np.zeros((13,), dtype=np.float32),
            body_names_sorted=None,
            body_state=None,
        )

    def read_observation(self) -> SimRobotObservation:
        return self._obs

    def apply_effort(self, effort_sorted: np.ndarray) -> None:
        self.applied_efforts.append(np.asarray(effort_sorted, dtype=np.float32).copy())
        if len(self.applied_efforts) >= 3:
            raise _StopLoop("stop after validating standby->protocol switch")

    def step(self) -> None:
        return

    def close(self) -> None:
        return


@pytest.mark.general
def test_server_detects_new_messages_via_transport_token():
    adapter = _DummyAdapter()
    assist = _DummyAssist()
    server = RobotProtocolServer(
        adapter=adapter,
        transport=_DummyTransport(),
        codec=_DummyCodec(),
        actuation=_DummyActuation(),
        config=ServerConfig(dt=0.01, realtime=False),
        standby_controller=_DummyStandby(),
        standby_config=StandbyConfig(enabled=True, required_active_cmds=2),
        assist=assist,
    )
    server.start()
    try:
        with pytest.raises(_StopLoop):
            server.run_forever()
    finally:
        server.close()

    # First step: standby effort while streak builds.
    assert np.allclose(adapter.applied_efforts[0], np.array([-1.0], dtype=np.float32))
    # Second step onward: protocol control takes over.
    assert np.allclose(adapter.applied_efforts[1], np.array([1.0], dtype=np.float32))
    assert np.allclose(adapter.applied_efforts[2], np.array([1.0], dtype=np.float32))
    # Elastic-band release is now manual (no automatic release on protocol takeover).
    assert assist.start_release_calls == 0
