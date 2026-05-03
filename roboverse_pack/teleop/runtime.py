from __future__ import annotations

# ruff: noqa: D102

"""Canonical teleoperation runtime for calibrated two-hand targets."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from roboverse_pack.teleop.common import (
    TeleopCommand,
    load_packet_from_json_bytes,
    parse_command_payload,
    validate_packet,
)
from roboverse_pack.teleop.transforms import TeleopTransformProfile, apply_profile_to_pose, get_profile


class HandRetargetingRuntime(Protocol):
    """Protocol for optional hand landmark retargeting backends."""

    def retarget(
        self,
        hand_key: str,
        fingers,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> np.ndarray | None: ...


@dataclass(frozen=True)
class CanonicalTeleopTargets:
    """Decoded canonical teleoperation targets for left and right hands."""

    left_work_pose_cm_xyzw: tuple[float, ...]
    right_work_pose_cm_xyzw: tuple[float, ...]
    left_close_ratio: float
    right_close_ratio: float
    left_hand_target_q_rad: tuple[float, ...] | None = None
    right_hand_target_q_rad: tuple[float, ...] | None = None
    left_arm_target_q_rad: tuple[float, ...] | None = None
    right_arm_target_q_rad: tuple[float, ...] | None = None
    transform_profile: str = "identity"
    status: str = "teleop:decoded"

    def as_payload(self) -> dict[str, object]:
        return {
            "left_work_pose_cm_xyzw": [float(value) for value in self.left_work_pose_cm_xyzw],
            "right_work_pose_cm_xyzw": [float(value) for value in self.right_work_pose_cm_xyzw],
            "left_close_ratio": float(self.left_close_ratio),
            "right_close_ratio": float(self.right_close_ratio),
            "left_hand_target_q_rad": None
            if self.left_hand_target_q_rad is None
            else [float(value) for value in self.left_hand_target_q_rad],
            "right_hand_target_q_rad": None
            if self.right_hand_target_q_rad is None
            else [float(value) for value in self.right_hand_target_q_rad],
            "left_arm_target_q_rad": None
            if self.left_arm_target_q_rad is None
            else [float(value) for value in self.left_arm_target_q_rad],
            "right_arm_target_q_rad": None
            if self.right_arm_target_q_rad is None
            else [float(value) for value in self.right_arm_target_q_rad],
            "transform_profile": self.transform_profile,
            "status": self.status,
        }


@dataclass
class CanonicalTeleopRuntime:
    """Decode validated packets into calibrated canonical teleoperation targets."""

    reference_pose_provider: Callable[[str], Sequence[float]] | Mapping[str, Sequence[float]]
    calibration_path: Path | str | None = None
    transform_profile: TeleopTransformProfile | str = "identity"
    clip_target_pos: Callable[[np.ndarray], np.ndarray] | None = None
    grip_to_close_ratio: Callable[[float], float] | None = None
    hand_retargeting_runtime: HandRetargetingRuntime | None = None
    hand_joint_limits: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None
    ramp_duration_s: float = 0.0
    clock: Callable[[], float] = time.time
    teleop_status: str = field(default="teleop:idle", init=False)
    latest_packet: dict[str, Any] | None = field(default=None, init=False)
    teleop_calibration: dict[str, Any] = field(default_factory=dict, init=False)
    runtime_teleop_calibration: dict[str, Any] = field(default_factory=dict, init=False)
    teleop_ramp_reference: dict[str, Any] = field(default_factory=dict, init=False)
    teleop_ramp_start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.transform_profile = get_profile(self.transform_profile) if isinstance(self.transform_profile, str) else self.transform_profile
        self.calibration_path = None if self.calibration_path is None else Path(self.calibration_path)
        if self.clip_target_pos is None:
            self.clip_target_pos = lambda pos: np.asarray(pos, dtype=np.float32)
        if self.grip_to_close_ratio is None:
            self.grip_to_close_ratio = lambda grip: float(np.clip(0.15 + 0.8 * grip, 0.15, 0.95))
        self._hand_joint_limits = self._normalize_joint_limits(self.hand_joint_limits)
        self.load_calibration_file_if_present()

    def _normalize_joint_limits(
        self,
        limits: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if limits is None:
            return {}
        normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for hand_key, (lower, upper) in limits.items():
            normalized[str(hand_key)] = (
                np.asarray(lower, dtype=np.float32).reshape(-1),
                np.asarray(upper, dtype=np.float32).reshape(-1),
            )
        return normalized

    def _vec_to_list(self, vec: np.ndarray) -> list[float]:
        return [float(value) for value in np.asarray(vec, dtype=np.float32).tolist()]

    def _quat_to_list(self, quat: np.ndarray) -> list[float]:
        return [float(value) for value in np.asarray(quat, dtype=np.float32).tolist()]

    def observe_packet(self, packet: Mapping[str, Any] | bytes | bytearray | memoryview) -> dict[str, Any]:
        if isinstance(packet, (bytes, bytearray, memoryview)):
            restored = load_packet_from_json_bytes(packet)
        else:
            validate_packet(packet)
            restored = dict(packet)
        self.latest_packet = restored
        return restored

    def handle_command(self, payload: TeleopCommand | Mapping[str, Any]) -> bool | None:
        command = payload if isinstance(payload, TeleopCommand) else parse_command_payload(payload)
        if command.name == "capture_zero":
            return self.capture_calibration()
        if command.name == "save_calibration":
            return self.save_calibration_file()
        if command.name == "capture_reference":
            return self.capture_reference()
        if command.name == "clear_calibration":
            self.clear_calibration()
            return None
        if command.name == "calibrate_and_save":
            captured = self.capture_calibration()
            if not captured:
                return False
            return self.save_calibration_file()
        self.teleop_status = f"teleop:command:unknown:{command.name}"
        return False

    def handle_commands(self, payloads: Sequence[TeleopCommand | Mapping[str, Any]]) -> list[bool | None]:
        return [self.handle_command(payload) for payload in payloads]

    def load_calibration_file_if_present(self) -> bool:
        if self.calibration_path is None or not self.calibration_path.exists():
            return False
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.teleop_status = "teleop:calibration:load-error"
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("left"), dict) or not isinstance(payload.get("right"), dict):
            self.teleop_status = "teleop:calibration:invalid-file"
            return False
        self.teleop_calibration = payload
        self.teleop_status = f"teleop:calibration:loaded:{self.calibration_path.name}"
        return True

    def save_calibration_file(self) -> bool:
        if self.calibration_path is None:
            self.teleop_status = "teleop:calibration:path-missing"
            return False
        if "left" not in self.teleop_calibration or "right" not in self.teleop_calibration:
            self.teleop_status = "teleop:calibration:none"
            return False
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(json.dumps(self.teleop_calibration, indent=2), encoding="utf-8")
        self.teleop_status = f"teleop:calibration:saved:{self.calibration_path.name}"
        return True

    def clear_calibration(self) -> None:
        self.teleop_calibration = {}
        self.runtime_teleop_calibration = {}
        self.teleop_ramp_reference = {}
        self.teleop_ramp_start_time = 0.0
        self.teleop_status = "teleop:calibration:cleared"

    def capture_calibration(self, packet: Mapping[str, Any] | None = None) -> bool:
        source = self.latest_packet if packet is None else packet
        if not isinstance(source, Mapping):
            self.teleop_status = "teleop:calibration:waiting-packet"
            return False

        calibration: dict[str, Any] = {
            "version": 1,
            "updated_at": float(self.clock()),
        }
        for hand_key in ("left", "right"):
            extracted = self.extract_hand_pose(source, hand_key)
            if extracted is None:
                self.teleop_status = f"teleop:calibration:invalid-{hand_key}"
                return False
            device_pos, device_quat = extracted
            sim_target = self._current_reference_pose(hand_key)
            calibration[hand_key] = {
                "device_zero_pos_m": self._vec_to_list(device_pos),
                "device_zero_quat": self._quat_to_list(
                    self._normalize_quat(device_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                ),
                "sim_zero_pos_cm": self._vec_to_list(sim_target[:3]),
                "sim_zero_quat": self._quat_to_list(
                    self._normalize_quat(sim_target[3:7], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                ),
            }
        self.teleop_calibration = calibration
        self.teleop_status = "teleop:calibration:captured"
        return True

    def capture_runtime_reference(self, packet: Mapping[str, Any] | None = None) -> bool:
        source = self.latest_packet if packet is None else packet
        if not isinstance(source, Mapping):
            self.teleop_status = "teleop:reference:waiting-packet"
            return False
        calibration: dict[str, Any] = {}
        for hand_key in ("left", "right"):
            extracted = self.extract_hand_pose(source, hand_key)
            if extracted is None:
                self.teleop_status = f"teleop:reference:invalid-{hand_key}"
                return False
            device_pos, device_quat = extracted
            sim_target = self._current_reference_pose(hand_key)
            calibration[hand_key] = {
                "device_zero_pos_m": self._vec_to_list(device_pos),
                "device_zero_quat": self._quat_to_list(
                    self._normalize_quat(device_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                ),
                "sim_zero_pos_cm": self._vec_to_list(sim_target[:3]),
                "sim_zero_quat": self._quat_to_list(
                    self._normalize_quat(sim_target[3:7], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                ),
            }
        self.runtime_teleop_calibration = calibration
        return True

    def capture_ramp_reference(self) -> None:
        self.teleop_ramp_reference = {}
        for hand_key in ("left", "right"):
            sim_target = self._current_reference_pose(hand_key)
            self.teleop_ramp_reference[hand_key] = {
                "sim_ref_pos_cm": self._vec_to_list(sim_target[:3]),
                "sim_ref_quat": self._quat_to_list(
                    self._normalize_quat(sim_target[3:7], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                ),
            }
        self.teleop_ramp_start_time = float(self.clock())

    def capture_reference(self, packet: Mapping[str, Any] | None = None) -> bool:
        if not self.capture_runtime_reference(packet):
            if self.teleop_status == "teleop:idle":
                self.teleop_status = "teleop:reference:invalid-packet"
            return False
        self.capture_ramp_reference()
        self.teleop_status = "teleop:reference:captured"
        return True

    def get_ramp_alpha(self) -> float:
        if self.teleop_ramp_start_time <= 0.0 or self.ramp_duration_s <= 0.0:
            return 1.0
        elapsed = float(self.clock()) - self.teleop_ramp_start_time
        return float(np.clip(elapsed / self.ramp_duration_s, 0.0, 1.0))

    def apply_reference(self, hand_key: str, target_pos_cm: np.ndarray, target_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = self.teleop_ramp_reference.get(hand_key)
        if not isinstance(reference, Mapping):
            return self._clip_target_pos(target_pos_cm), self._normalize_quat(
                target_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
        ref_pos = np.asarray(reference.get("sim_ref_pos_cm", []), dtype=np.float32)
        ref_quat = np.asarray(reference.get("sim_ref_quat", []), dtype=np.float32)
        if ref_pos.shape != (3,) or ref_quat.shape != (4,):
            fallback = ref_quat if ref_quat.shape == (4,) else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return self._clip_target_pos(target_pos_cm), self._normalize_quat(target_quat, fallback)
        alpha = self.get_ramp_alpha()
        if alpha >= 1.0:
            return self._clip_target_pos(target_pos_cm), self._normalize_quat(target_quat, ref_quat)
        blended_pos = ref_pos + alpha * (target_pos_cm - ref_pos)
        blended_quat = self._quat_slerp_xyzw(ref_quat, target_quat, alpha)
        return self._clip_target_pos(blended_pos), blended_quat

    def _current_reference_pose(self, hand_key: str) -> np.ndarray:
        if isinstance(self.reference_pose_provider, Mapping):
            pose = self.reference_pose_provider[hand_key]
        else:
            pose = self.reference_pose_provider(hand_key)
        pose_array = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_array.shape != (7,):
            raise ValueError(f"reference pose for {hand_key} must be shape (7,), got {pose_array.shape}")
        pose_array[3:7] = self._normalize_quat(pose_array[3:7], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return pose_array

    def extract_hand_pose(self, packet: Mapping[str, Any], hand_key: str) -> tuple[np.ndarray, np.ndarray] | None:
        hand = packet.get(hand_key)
        if not isinstance(hand, Mapping):
            return None
        pos = np.asarray(hand.get("pos", []), dtype=np.float32).reshape(-1)
        quat = np.asarray(hand.get("quat", []), dtype=np.float32).reshape(-1)
        if pos.shape != (3,) or quat.shape != (4,):
            return None
        quat = self._normalize_quat(quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        transformed = np.asarray(
            apply_profile_to_pose(self.transform_profile, hand_key, np.concatenate([pos, quat]).tolist()),
            dtype=np.float32,
        )
        return transformed[:3], self._normalize_quat(transformed[3:7], quat)

    def ensure_runtime_calibration(self, hand_key: str, device_pos_m: np.ndarray, device_quat: np.ndarray) -> None:
        if isinstance(self.teleop_calibration.get(hand_key), dict):
            return
        if isinstance(self.runtime_teleop_calibration.get(hand_key), dict):
            return
        sim_target = self._current_reference_pose(hand_key)
        self.runtime_teleop_calibration[hand_key] = {
            "device_zero_pos_m": self._vec_to_list(device_pos_m),
            "device_zero_quat": self._quat_to_list(
                self._normalize_quat(device_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            ),
            "sim_zero_pos_cm": self._vec_to_list(sim_target[:3]),
            "sim_zero_quat": self._quat_to_list(
                self._normalize_quat(sim_target[3:7], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            ),
        }
        self.teleop_status = f"teleop:auto-zero:{hand_key}"

    def apply_calibration(self, hand_key: str, device_pos_m: np.ndarray, device_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.runtime_teleop_calibration.get(hand_key)
        if not isinstance(calibration, dict):
            calibration = self.teleop_calibration.get(hand_key)
        if not isinstance(calibration, dict):
            return self._clip_target_pos(device_pos_m * 100.0), self._normalize_quat(
                device_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )

        device_zero_pos = np.asarray(calibration.get("device_zero_pos_m", []), dtype=np.float32).reshape(-1)
        device_zero_quat = np.asarray(calibration.get("device_zero_quat", []), dtype=np.float32).reshape(-1)
        sim_zero_pos = np.asarray(calibration.get("sim_zero_pos_cm", []), dtype=np.float32).reshape(-1)
        sim_zero_quat = np.asarray(calibration.get("sim_zero_quat", []), dtype=np.float32).reshape(-1)
        if device_zero_pos.shape != (3,) or device_zero_quat.shape != (4,) or sim_zero_pos.shape != (3,) or sim_zero_quat.shape != (4,):
            return self._clip_target_pos(device_pos_m * 100.0), self._normalize_quat(
                device_quat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )

        sim_pos = sim_zero_pos + (device_pos_m - device_zero_pos) * 100.0
        sim_quat = self._quat_mul_xyzw(
            sim_zero_quat,
            self._quat_mul_xyzw(self._quat_inverse_xyzw(device_zero_quat), device_quat),
        )
        return self._clip_target_pos(sim_pos), self._normalize_quat(sim_quat, sim_zero_quat)

    def decode_packet(self, packet: Mapping[str, Any] | bytes | bytearray | memoryview) -> CanonicalTeleopTargets:
        restored = self.observe_packet(packet)
        left_pose, left_close_ratio, left_hand_target = self._decode_hand(restored, "left")
        right_pose, right_close_ratio, right_hand_target = self._decode_hand(restored, "right")
        self.teleop_status = "teleop:decoded"
        return CanonicalTeleopTargets(
            left_work_pose_cm_xyzw=tuple(float(value) for value in left_pose.tolist()),
            right_work_pose_cm_xyzw=tuple(float(value) for value in right_pose.tolist()),
            left_close_ratio=float(left_close_ratio),
            right_close_ratio=float(right_close_ratio),
            left_hand_target_q_rad=None if left_hand_target is None else tuple(float(value) for value in left_hand_target.tolist()),
            right_hand_target_q_rad=None if right_hand_target is None else tuple(float(value) for value in right_hand_target.tolist()),
            transform_profile=self.transform_profile.name,
            status=self.teleop_status,
        )

    def decode_latest_packet(self) -> CanonicalTeleopTargets | None:
        if self.latest_packet is None:
            return None
        return self.decode_packet(self.latest_packet)

    def _decode_hand(self, packet: Mapping[str, Any], hand_key: str) -> tuple[np.ndarray, float, np.ndarray | None]:
        extracted = self.extract_hand_pose(packet, hand_key)
        if extracted is None:
            raise ValueError(f"invalid packet hand payload: {hand_key}")
        device_pos_m, device_quat = extracted
        self.ensure_runtime_calibration(hand_key, device_pos_m, device_quat)
        target_pos_cm, target_quat = self.apply_calibration(hand_key, device_pos_m, device_quat)
        target_pos_cm, target_quat = self.apply_reference(hand_key, target_pos_cm, target_quat)

        hand_payload = packet[hand_key]
        assert isinstance(hand_payload, Mapping)
        grip = float(np.clip(float(hand_payload.get("grip", 0.0)), 0.0, 1.0))
        close_ratio = float(self.grip_to_close_ratio(grip))
        hand_target = self._retarget_hand(hand_key, hand_payload.get("fingers"))
        return np.concatenate([target_pos_cm, target_quat]).astype(np.float32), close_ratio, hand_target

    def _retarget_hand(self, hand_key: str, fingers) -> np.ndarray | None:
        if self.hand_retargeting_runtime is None:
            return None
        joint_lower, joint_upper = self._hand_joint_limits.get(hand_key, (None, None))
        return self.hand_retargeting_runtime.retarget(
            hand_key,
            fingers,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )

    def _clip_target_pos(self, pos: np.ndarray) -> np.ndarray:
        clipped = self.clip_target_pos(np.asarray(pos, dtype=np.float32))
        return np.asarray(clipped, dtype=np.float32).reshape(3)

    @staticmethod
    def _normalize_quat(quat: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm < 1.0e-6:
            return np.asarray(fallback, dtype=np.float32).copy()
        return (np.asarray(quat, dtype=np.float32) / norm).astype(np.float32)

    @staticmethod
    def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
        inv = np.asarray(quat, dtype=np.float32).copy()
        inv[:3] *= -1.0
        return inv

    @staticmethod
    def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float32)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float32)
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )

    def _quat_slerp_xyzw(self, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        q0_norm = self._normalize_quat(q0.astype(np.float32, copy=False), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        q1_norm = self._normalize_quat(q1.astype(np.float32, copy=False), q0_norm)
        dot = float(np.dot(q0_norm, q1_norm))
        if dot < 0.0:
            q1_norm = -q1_norm
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_quat((1.0 - alpha) * q0_norm + alpha * q1_norm, q0_norm)
        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        if sin_theta_0 < 1.0e-6:
            return q0_norm
        theta = theta_0 * alpha
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return self._normalize_quat((s0 * q0_norm + s1 * q1_norm).astype(np.float32), q0_norm)


__all__ = [
    "CanonicalTeleopRuntime",
    "CanonicalTeleopTargets",
    "HandRetargetingRuntime",
]
