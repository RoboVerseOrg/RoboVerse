from __future__ import annotations

import struct

import numpy as np

from roboverse_pack.robot_protocols.core.types import CanonicalRobotCommand, SimRobotObservation
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.actuation import UnitreeLowCmdActuationModel
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.codec import AutoRemoteConfig, UnitreeSdk2Codec
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.profile import UnitreeRobotProfile


def test_unitree_actuation_scatter_and_clip():
    # Sorted sim joints: a,b,c
    obs = SimRobotObservation(
        joint_names_sorted=["a", "b", "c"],
        q_sorted=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        dq_sorted=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        tau_sorted=None,
        root_state=np.zeros((13,), dtype=np.float32),
        body_names_sorted=None,
        body_state=None,
    )

    # Protocol order maps motor0->c, motor1->a
    protocol_to_sorted = [2, 0]
    cmd = CanonicalRobotCommand(
        joint_names=["c", "a"],
        q_des=np.array([1.0, 1.0], dtype=np.float32),
        qd_des=np.array([0.0, 0.0], dtype=np.float32),
        kp=np.array([10.0, 10.0], dtype=np.float32),
        kd=np.array([0.0, 0.0], dtype=np.float32),
        tau_ff=np.array([0.0, 0.0], dtype=np.float32),
        mode=np.array([1, 1], dtype=np.int32),
    )

    model = UnitreeLowCmdActuationModel(
        protocol_to_sorted=protocol_to_sorted,
        torque_limits_protocol=np.array([5.0, 100.0], dtype=np.float32),
    )
    out = model.compute_effort(cmd, obs)
    # motor0 torque = 10*(1-0)=10 -> clipped to 5, scattered to c (idx 2)
    # motor1 torque = 10*(1-0)=10, scattered to a (idx 0)
    assert np.allclose(out, np.array([10.0, 0.0, 5.0], dtype=np.float32))


def test_unitree_codec_auto_remote_presses_buttons():
    profile = UnitreeRobotProfile(robot_name="g1_dof29", msg_type="hg", motor_names=["j0"])
    codec = UnitreeSdk2Codec(
        profile=profile,
        protocol_to_sorted=[0],
        lowstate_factory=lambda: object(),
        sportstate_factory=lambda: object(),
        auto_remote=AutoRemoteConfig(enabled=True, press_start_after_s=0.5, press_a_after_s=1.0),
    )

    # t=0: nothing pressed
    payload = codec._build_wireless_remote(0.0)
    keys = struct.unpack("<H", payload[2:4])[0]
    assert (keys & (1 << 2)) == 0
    assert (keys & (1 << 8)) == 0

    # t=0.6: start pressed
    payload = codec._build_wireless_remote(0.6)
    keys = struct.unpack("<H", payload[2:4])[0]
    assert (keys & (1 << 2)) != 0
    assert (keys & (1 << 8)) == 0

    # t=1.1: start + A pressed
    payload = codec._build_wireless_remote(1.1)
    keys = struct.unpack("<H", payload[2:4])[0]
    assert (keys & (1 << 2)) != 0
    assert (keys & (1 << 8)) != 0
