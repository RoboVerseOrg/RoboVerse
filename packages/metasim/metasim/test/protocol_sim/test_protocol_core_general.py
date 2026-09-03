from __future__ import annotations

from pathlib import Path

import pytest
import torch

from metasim.protocol_sim.core.interfaces import CommandAdapter, ProtocolCodec, Transport
from metasim.protocol_sim.core.server import ProtocolServer, ProtocolServerConfig
from metasim.protocol_sim.core.sim_adapter import MetaSimProtocolAdapter
from metasim.types import CompatActionInput, RobotState, TensorState


def _state(marker: float) -> TensorState:
    return TensorState(
        objects={},
        robots={
            "robot": RobotState(
                root_state=torch.zeros((1, 13), dtype=torch.float32),
                body_names=[],
                body_state=torch.zeros((1, 0, 13), dtype=torch.float32),
                joint_pos=torch.tensor([[marker]], dtype=torch.float32),
                joint_vel=torch.zeros((1, 1), dtype=torch.float32),
                joint_pos_target=None,
                joint_vel_target=None,
                joint_effort_target=None,
            )
        },
        cameras={},
    )


class _SequenceTransport(Transport):
    def __init__(self, entries: list[tuple[object | None, int | None]]):
        self._entries = list(entries)
        self._index = 0
        self.started = False
        self.closed = False
        self.published: list[tuple[str, object]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def get_latest_command(self) -> object | None:
        msg, _ = self.get_latest_command_with_token()
        return msg

    def get_latest_command_with_token(self) -> tuple[object | None, int | None]:
        entry = self._entries[min(self._index, len(self._entries) - 1)]
        self._index += 1
        return entry

    def publish(self, channel: str, msg: object) -> None:
        self.published.append((channel, msg))


class _ReusableObjectTransport(_SequenceTransport):
    def __init__(self):
        self.msg = {"value": 0}
        super().__init__(entries=[])

    def get_latest_command_with_token(self) -> tuple[object | None, int | None]:
        self._index += 1
        self.msg["value"] = self._index
        return self.msg, self._index


class _RecordingCodec(ProtocolCodec):
    def __init__(self):
        self.decode_calls = 0
        self.encoded_markers: list[float] = []

    def decode_command(self, msg: object) -> object:
        self.decode_calls += 1
        return dict(msg) if isinstance(msg, dict) else msg

    def encode_messages(self, state: TensorState, *, sim_time_s: float) -> dict[str, object]:
        marker = float(state.robots["robot"].joint_pos[0, 0].item())
        self.encoded_markers.append(marker)
        return {"state": {"marker": marker, "sim_time_s": sim_time_s}}


class _RecordingCommandAdapter(CommandAdapter):
    def __init__(self):
        self.calls: list[tuple[object, float]] = []

    def to_action(self, command: object, state: TensorState) -> CompatActionInput | None:
        marker = float(state.robots["robot"].joint_pos[0, 0].item())
        self.calls.append((command, marker))
        value = float(command["value"] if isinstance(command, dict) else command)
        return [{"robot": {"dof_pos_target": {"j0": value}}}]


class _DummyHandler:
    def __init__(self):
        self.marker = 0.0
        self.actions: list[CompatActionInput] = []
        self.get_modes: list[str] = []
        self.simulate_calls = 0
        self.closed = False

    def get_states(self, env_ids=None, mode="dict"):
        assert env_ids is None
        self.get_modes.append(mode)
        return _state(self.marker)

    def set_dof_targets(self, action: CompatActionInput) -> None:
        self.actions.append(action)

    def simulate(self) -> None:
        self.simulate_calls += 1
        self.marker += 1.0

    def close(self) -> None:
        self.closed = True


@pytest.mark.general
def test_protocol_server_reuses_latest_decoded_command_between_transport_updates():
    handler = _DummyHandler()
    transport = _SequenceTransport(entries=[({"value": 4}, 1), ({"value": 4}, 1)])
    codec = _RecordingCodec()
    command_adapter = _RecordingCommandAdapter()
    server = ProtocolServer(
        sim=MetaSimProtocolAdapter(handler),
        transport=transport,
        codec=codec,
        command_adapter=command_adapter,
        config=ProtocolServerConfig(dt=0.01, realtime=False),
    )

    server.start()
    server.step_once()
    server.step_once()
    server.close()

    assert transport.started is True
    assert transport.closed is True
    assert handler.closed is True
    assert codec.decode_calls == 1
    assert len(command_adapter.calls) == 2
    assert len(handler.actions) == 2
    assert handler.simulate_calls == 2
    assert transport.published == [
        ("state", {"marker": 1.0, "sim_time_s": 0.01}),
        ("state", {"marker": 2.0, "sim_time_s": 0.02}),
    ]


@pytest.mark.general
def test_protocol_server_uses_transport_token_for_reused_message_objects():
    handler = _DummyHandler()
    transport = _ReusableObjectTransport()
    codec = _RecordingCodec()
    command_adapter = _RecordingCommandAdapter()
    server = ProtocolServer(
        sim=MetaSimProtocolAdapter(handler),
        transport=transport,
        codec=codec,
        command_adapter=command_adapter,
        config=ProtocolServerConfig(dt=0.01, realtime=False),
    )

    server.step_once()
    server.step_once()

    assert codec.decode_calls == 2
    assert handler.actions == [
        [{"robot": {"dof_pos_target": {"j0": 1.0}}}],
        [{"robot": {"dof_pos_target": {"j0": 2.0}}}],
    ]


@pytest.mark.general
def test_metasim_protocol_adapter_delegates_to_handler_contract():
    handler = _DummyHandler()
    adapter = MetaSimProtocolAdapter(handler)
    action = [{"robot": {"dof_pos_target": {"j0": 3.0}}}]

    state = adapter.read_state()
    adapter.apply_action(action)
    adapter.step()
    adapter.close()

    assert float(state.robots["robot"].joint_pos[0, 0].item()) == 0.0
    assert handler.get_modes == ["tensor"]
    assert handler.actions == [action]
    assert handler.simulate_calls == 1
    assert handler.closed is True


@pytest.mark.general
def test_protocol_sim_core_contains_no_task_robot_or_unitree_specific_modules():
    import metasim.protocol_sim as protocol_sim

    package_dir = Path(protocol_sim.__file__).resolve().parent
    forbidden_files = {
        "elastic_band.py",
        "task_alignment.py",
        "standby_policy.py",
        "unitree_sdk2_sim_server.py",
    }
    existing_files = {path.name for path in package_dir.rglob("*.py")}
    assert forbidden_files.isdisjoint(existing_files)

    forbidden_tokens = [
        "unitree",
        "humanoid",
        "roboverse_pack",
        "elasticband",
        "elastic_band",
        "standby",
        "task_alignment",
    ]
    for path in package_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden_tokens), path
