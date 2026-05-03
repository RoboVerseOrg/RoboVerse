from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from roboverse_pack.teleop.common import (
    KNOWN_TELEOP_COMMANDS,
    TeleopCommand,
    load_packet_from_json_bytes,
    parse_command_payload,
    validate_command_payload,
    validate_packet,
)
from roboverse_pack.teleop.hand_retargeting import (
    DEFAULT_WUJI_HAND_RETARGETING_CONFIG,
    WujiHandRetargetingRuntime,
    coerce_finger_landmarks,
)
from roboverse_pack.teleop.runtime import CanonicalTeleopRuntime
from roboverse_pack.teleop.transforms import (
    apply_profile_to_pose,
    default_avp_transform_profile,
    get_profile,
    profile_choices,
)


class _FakeHandRetargeter:
    def retarget(
        self,
        hand_key: str,
        fingers,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if fingers is None:
            return None
        base = 0.1 if hand_key == "left" else 0.2
        return np.full(20, base, dtype=np.float32)


class _FakeWujiRetargeter:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def retarget_left(self, fingers: np.ndarray) -> np.ndarray:
        assert fingers.shape == (21, 3)
        return np.linspace(-1.0, 1.0, 20, dtype=np.float32)

    def retarget_right(self, fingers: np.ndarray) -> np.ndarray:
        assert fingers.shape == (21, 3)
        return np.linspace(1.0, 3.0, 20, dtype=np.float32)


def _valid_hand(*, include_fingers: bool = False) -> dict[str, object]:
    hand: dict[str, object] = {
        "pos": [0.1, 0.2, 0.3],
        "quat": [0.0, 0.0, 0.0, 1.0],
        "grip": 0.4,
    }
    if include_fingers:
        hand["fingers"] = _valid_landmarks()
    return hand


def _valid_packet(
    *,
    left_pos: tuple[float, float, float] = (0.1, 0.2, 0.3),
    right_pos: tuple[float, float, float] = (0.4, 0.5, 0.6),
    left_grip: float = 0.25,
    right_grip: float = 0.75,
    include_fingers: bool = True,
) -> dict[str, object]:
    return {
        "left": {
            "pos": list(left_pos),
            "quat": [0.0, 0.0, 0.0, 1.0],
            "grip": left_grip,
            "fingers": _valid_landmarks() if include_fingers else None,
        },
        "right": {
            "pos": list(right_pos),
            "quat": [0.0, 0.0, 0.0, 1.0],
            "grip": right_grip,
            "fingers": _valid_landmarks() if include_fingers else None,
        },
    }


def _valid_landmarks() -> list[list[float]]:
    return [[float(i), float(i + 1), float(i + 2)] for i in range(21)]


def _reference_pose_provider(hand_key: str) -> np.ndarray:
    if hand_key == "left":
        return np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if hand_key == "right":
        return np.array([40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    raise KeyError(hand_key)


def test_packet_and_command_validation_match_bidex_contract() -> None:
    packet = {"left": _valid_hand(include_fingers=True), "right": _valid_hand(include_fingers=True)}

    validate_packet(packet)
    assert load_packet_from_json_bytes(json.dumps(packet).encode("utf-8")) == packet

    for command in KNOWN_TELEOP_COMMANDS:
        payload = {"command": command}
        validate_command_payload(payload)
        assert parse_command_payload(payload) == TeleopCommand(name=command)

    bad_packet = {"left": _valid_hand(include_fingers=True), "right": _valid_hand(include_fingers=True)}
    bad_packet["left"]["pos"] = [0.1, 0.2]
    with pytest.raises(ValueError, match=r"left\.pos"):
        validate_packet(bad_packet)

    with pytest.raises(ValueError, match="unexpected command payload fields: extra"):
        parse_command_payload({"command": "capture_zero", "extra": True})


def test_transform_profiles_apply_openarm_wrist_mapping() -> None:
    assert "identity" in profile_choices()
    assert default_avp_transform_profile("mujoco") == "avp_wrist_openarm_mujoco"
    assert default_avp_transform_profile("newton") == "avp_wrist_openarm_newton"

    identity_pose = apply_profile_to_pose(get_profile("identity"), "left", [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    transformed_left = apply_profile_to_pose(
        get_profile("avp_wrist_openarm_newton"),
        "left",
        [1.0, 2.0, 3.0, 0.70710678, 0.0, 0.0, 0.70710678],
    )
    transformed_right = apply_profile_to_pose(
        get_profile("avp_wrist_openarm_newton"),
        "right",
        [1.0, 2.0, 3.0, 0.70710678, 0.0, 0.0, 0.70710678],
    )

    assert identity_pose == pytest.approx([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert transformed_left[:3] == pytest.approx([-1.0, 2.0, 3.0])
    assert transformed_right[:3] == pytest.approx([-1.0, 2.0, 3.0])
    assert np.linalg.norm(np.asarray(transformed_left[3:], dtype=np.float32)) == pytest.approx(1.0, rel=1e-5)
    assert np.linalg.norm(np.asarray(transformed_right[3:], dtype=np.float32)) == pytest.approx(1.0, rel=1e-5)
    assert transformed_left[3:] != pytest.approx(transformed_right[3:])


def test_runtime_decodes_canonical_targets_with_calibration_profile_and_retargeting(tmp_path: Path) -> None:
    runtime = CanonicalTeleopRuntime(
        reference_pose_provider=_reference_pose_provider,
        calibration_path=tmp_path / "teleop-calibration.json",
        transform_profile="avp_wrist_openarm_newton",
        clip_target_pos=lambda pos: pos.astype(np.float32, copy=False),
        hand_retargeting_runtime=_FakeHandRetargeter(),
    )
    zero_packet = _valid_packet(
        left_pos=(0.10, 0.00, 0.00), right_pos=(0.10, 0.00, 0.00), left_grip=0.0, right_grip=0.0
    )
    runtime.observe_packet(zero_packet)

    assert runtime.handle_command({"command": "capture_zero"}) is True
    assert runtime.handle_command({"command": "save_calibration"}) is True
    assert runtime.calibration_path is not None
    assert runtime.calibration_path.exists()

    decoded = runtime.decode_packet(
        _valid_packet(left_pos=(0.20, 0.00, 0.00), right_pos=(0.05, 0.00, 0.00), left_grip=1.0, right_grip=0.5)
    )

    payload = decoded.as_payload()
    assert payload["left_work_pose_cm_xyzw"][:3] == pytest.approx([0.0, 20.0, 30.0])
    assert payload["right_work_pose_cm_xyzw"][:3] == pytest.approx([45.0, 50.0, 60.0])
    assert payload["left_close_ratio"] == pytest.approx(0.95)
    assert payload["right_close_ratio"] == pytest.approx(0.55)
    assert payload["left_hand_target_q_rad"] == pytest.approx([0.1] * 20)
    assert payload["right_hand_target_q_rad"] == pytest.approx([0.2] * 20)
    assert payload["transform_profile"] == "avp_wrist_openarm_newton"


def test_wuji_retargeting_config_and_landmark_coercion() -> None:
    assert DEFAULT_WUJI_HAND_RETARGETING_CONFIG.is_file()
    assert "roboverse_pack" in DEFAULT_WUJI_HAND_RETARGETING_CONFIG.as_posix()

    fingers = _valid_landmarks()
    fingers[0][0] = float("nan")
    landmarks = coerce_finger_landmarks(fingers)

    assert landmarks is not None
    assert landmarks.dtype == np.float32
    assert landmarks.shape == (21, 3)
    assert float(landmarks[0, 0]) == 0.0
    assert coerce_finger_landmarks([[0.0, 1.0, 2.0] for _ in range(20)]) is None


def test_wuji_retargeting_runtime_retargets_and_clips_with_factory() -> None:
    runtime = WujiHandRetargetingRuntime(retargeter_factory=lambda _path: _FakeWujiRetargeter())
    lower = np.full(20, -0.25, dtype=np.float32)
    upper = np.full(20, 0.25, dtype=np.float32)

    runtime.reset()
    left = runtime.retarget("left", _valid_landmarks(), joint_lower=lower, joint_upper=upper)
    right = runtime.retarget("right", _valid_landmarks())

    assert left is not None
    assert right is not None
    assert left.dtype == np.float32
    assert right.dtype == np.float32
    assert left.shape == (20,)
    assert right.shape == (20,)
    assert float(left.min()) >= -0.25
    assert float(left.max()) <= 0.25
    assert right.tolist() == pytest.approx(np.linspace(1.0, 3.0, 20, dtype=np.float32).tolist())
