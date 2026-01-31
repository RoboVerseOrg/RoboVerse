from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from roboverse_pack.protocol_sim.core.types import CanonicalRobotCommand, SimRobotObservation
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.math_utils import quat_rotate_inverse_wxyz
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.profile import UnitreeRobotProfile


@dataclass
class AutoRemoteConfig:
    """Configuration for automated remote control actions."""

    enabled: bool = False
    press_start_after_s: float = 0.5
    press_a_after_s: float = 1.0


class UnitreeSdk2Codec:
    """Unitree SDK2 LowCmd/LowState codec.

    This codec is intentionally minimal and targets controller validation, matching
    the behavior of third_party/unitree_mujoco.
    """

    def __init__(
        self,
        *,
        profile: UnitreeRobotProfile,
        protocol_to_sorted: list[int],
        lowstate_factory: Any,
        sportstate_factory: Any,
        auto_remote: AutoRemoteConfig | None = None,
        gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        self._profile = profile
        self._protocol_to_sorted = list(protocol_to_sorted)
        self._lowstate_factory = lowstate_factory
        self._sportstate_factory = sportstate_factory
        self._auto_remote = auto_remote or AutoRemoteConfig()
        self._gravity_world = np.array(gravity_world, dtype=np.float32)

        self._prev_lin_vel_world: np.ndarray | None = None
        self._prev_time_s: float | None = None

    def decode_command(self, msg: Any) -> CanonicalRobotCommand:
        """Decode a Unitree LowCmd message into a CanonicalRobotCommand."""
        n = len(self._profile.motor_names)
        q_des = np.zeros((n,), dtype=np.float32)
        qd_des = np.zeros((n,), dtype=np.float32)
        kp = np.zeros((n,), dtype=np.float32)
        kd = np.zeros((n,), dtype=np.float32)
        tau = np.zeros((n,), dtype=np.float32)
        mode = np.zeros((n,), dtype=np.int32)

        for i in range(n):
            mc = msg.motor_cmd[i]
            q_des[i] = float(mc.q)
            qd_des[i] = float(mc.dq)
            kp[i] = float(mc.kp)
            kd[i] = float(mc.kd)
            tau[i] = float(mc.tau)
            # Some IDLs expose `mode`; keep 0 if missing.
            mode[i] = int(getattr(mc, "mode", 0))

        return CanonicalRobotCommand(
            joint_names=list(self._profile.motor_names),
            q_des=q_des,
            qd_des=qd_des,
            kp=kp,
            kd=kd,
            tau_ff=tau,
            mode=mode,
        )

    def _build_wireless_remote(self, sim_time_s: float) -> bytes:
        # 40 bytes is sufficient for the fields used by scripts/unitree_deploy/common/remote_controller.py
        payload = bytearray(40)

        keys = 0
        if self._auto_remote.enabled:
            if sim_time_s >= self._auto_remote.press_start_after_s:
                keys |= 1 << 2  # KeyMap.start
            if sim_time_s >= self._auto_remote.press_a_after_s:
                keys |= 1 << 8  # KeyMap.A

        struct.pack_into("<H", payload, 2, keys)
        # Axes (floats) default to 0.0: lx@4, rx@8, ry@12, ly@20
        return bytes(payload)

    def _assign_wireless_remote(self, low_state: Any, payload: bytes) -> None:
        wr = getattr(low_state, "wireless_remote", None)
        if wr is None:
            return
        try:
            # Common case: bytearray-like with slice assignment.
            wr[: len(payload)] = payload
            return
        except Exception:
            pass

        # Fallback: element-wise into a list of ints.
        try:
            for i, b in enumerate(payload):
                wr[i] = b
        except Exception:
            # Give up silently; not all IDLs expose this field as mutable.
            return

    def encode_messages(self, obs: SimRobotObservation, *, sim_time_s: float) -> dict[str, Any]:
        """Encode the robot observation into Unitree LowState and SportModeState messages."""
        n = len(self._profile.motor_names)
        q_proto = obs.q_sorted[self._protocol_to_sorted]
        dq_proto = obs.dq_sorted[self._protocol_to_sorted]

        low_state = self._lowstate_factory()

        # HG-only: used by deploy scripts during init.
        if self._profile.mode_machine is not None and hasattr(low_state, "mode_machine"):
            try:
                low_state.mode_machine = int(self._profile.mode_machine)
            except Exception:
                pass

        # Tick: mimic unitree_mujoco (ms).
        tick = int(sim_time_s / 1e-3)
        if hasattr(low_state, "tick"):
            try:
                low_state.tick = tick
            except Exception:
                pass

        for i in range(n):
            ms = low_state.motor_state[i]
            ms.q = float(q_proto[i])
            ms.dq = float(dq_proto[i])
            ms.tau_est = 0.0

        # IMU: publish pelvis/root pose and body-frame angular velocity.
        root = obs.root_state
        quat_wxyz = root[3:7].astype(np.float32, copy=False)
        ang_vel_world = root[10:13].astype(np.float32, copy=False)
        gyro_body = quat_rotate_inverse_wxyz(quat_wxyz, ang_vel_world)

        # Approximate proper acceleration (not relied on by deploy_real today).
        lin_vel_world = root[7:10].astype(np.float32, copy=False)
        acc_body = np.zeros((3,), dtype=np.float32)
        if self._prev_lin_vel_world is not None and self._prev_time_s is not None:
            dt = sim_time_s - self._prev_time_s
            if dt > 1e-6:
                acc_world = (lin_vel_world - self._prev_lin_vel_world) / float(dt)
                acc_body = quat_rotate_inverse_wxyz(quat_wxyz, acc_world - self._gravity_world)
        self._prev_lin_vel_world = lin_vel_world.copy()
        self._prev_time_s = float(sim_time_s)

        if hasattr(low_state, "imu_state"):
            try:
                low_state.imu_state.quaternion[0] = float(quat_wxyz[0])
                low_state.imu_state.quaternion[1] = float(quat_wxyz[1])
                low_state.imu_state.quaternion[2] = float(quat_wxyz[2])
                low_state.imu_state.quaternion[3] = float(quat_wxyz[3])

                low_state.imu_state.gyroscope[0] = float(gyro_body[0])
                low_state.imu_state.gyroscope[1] = float(gyro_body[1])
                low_state.imu_state.gyroscope[2] = float(gyro_body[2])

                low_state.imu_state.accelerometer[0] = float(acc_body[0])
                low_state.imu_state.accelerometer[1] = float(acc_body[1])
                low_state.imu_state.accelerometer[2] = float(acc_body[2])
            except Exception:
                pass

        # Wireless remote payload (for deploy_real state machine).
        payload = self._build_wireless_remote(sim_time_s)
        self._assign_wireless_remote(low_state, payload)

        # High / sport mode state (go message type in unitree_mujoco).
        sport = self._sportstate_factory()
        pos = root[0:3]
        vel = root[7:10]
        try:
            sport.position[0] = float(pos[0])
            sport.position[1] = float(pos[1])
            sport.position[2] = float(pos[2])
            sport.velocity[0] = float(vel[0])
            sport.velocity[1] = float(vel[1])
            sport.velocity[2] = float(vel[2])
        except Exception:
            pass

        return {
            self._profile.lowstate_topic: low_state,
            self._profile.sportstate_topic: sport,
        }
