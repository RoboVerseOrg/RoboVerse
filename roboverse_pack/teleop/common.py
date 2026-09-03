from __future__ import annotations

"""Packet and command validation for canonical teleoperation streams."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

KNOWN_TELEOP_COMMANDS = frozenset({
    "capture_zero",
    "save_calibration",
    "capture_reference",
    "clear_calibration",
    "calibrate_and_save",
})


@dataclass(frozen=True)
class TeleopCommand:
    """Validated teleoperation runtime command."""

    name: str


def _validate_vector(payload: Any, *, hand_key: str, field: str, size: int) -> None:
    if not isinstance(payload, (list, tuple)) or len(payload) != size:
        raise ValueError(f"{hand_key}.{field} must be length-{size}")


def _validate_fingers(payload: Any, *, hand_key: str) -> None:
    if payload is None:
        return
    if not isinstance(payload, (list, tuple)) or len(payload) != 21:
        raise ValueError(f"{hand_key}.fingers must be length-21 when present")
    for finger_id, point in enumerate(payload):
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError(f"{hand_key}.fingers[{finger_id}] must be length-3")


def _validate_hand(packet: Mapping[str, Any], *, hand_key: str) -> None:
    hand = packet.get(hand_key)
    if not isinstance(hand, Mapping):
        raise ValueError(f"missing hand payload: {hand_key}")

    _validate_vector(hand.get("pos"), hand_key=hand_key, field="pos", size=3)
    _validate_vector(hand.get("quat"), hand_key=hand_key, field="quat", size=4)
    if hand.get("grip") is None:
        raise ValueError(f"{hand_key}.grip is required")
    _validate_fingers(hand.get("fingers"), hand_key=hand_key)


def validate_packet(packet: Any) -> None:
    """Validate a two-hand teleoperation packet."""
    if not isinstance(packet, Mapping):
        raise ValueError("packet must be a dict")
    for hand_key in ("left", "right"):
        _validate_hand(packet, hand_key=hand_key)


def load_packet_from_json_bytes(payload: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Decode and validate a UTF-8 JSON teleoperation packet."""
    try:
        packet = json.loads(bytes(payload).decode("utf-8"))
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packet payload must be valid UTF-8 JSON bytes") from exc
    validate_packet(packet)
    return dict(packet)


def _normalize_command_name(value: Any) -> str:
    command = str(value).strip().lower()
    if not command:
        raise ValueError("command is required")
    if command not in KNOWN_TELEOP_COMMANDS:
        raise ValueError(f"unknown teleop command: {command}")
    return command


def validate_command_payload(payload: Any) -> None:
    """Validate a command payload sent alongside teleoperation packets."""
    if not isinstance(payload, Mapping):
        raise ValueError("command payload must be a dict")
    _normalize_command_name(payload.get("command"))
    extra_keys = sorted(str(key) for key in payload.keys() if key != "command")
    if extra_keys:
        extras = ", ".join(extra_keys)
        raise ValueError(f"unexpected command payload fields: {extras}")


def parse_command_payload(payload: Any) -> TeleopCommand:
    """Parse a command payload into a canonical command object."""
    validate_command_payload(payload)
    assert isinstance(payload, Mapping)
    return TeleopCommand(name=_normalize_command_name(payload.get("command")))
