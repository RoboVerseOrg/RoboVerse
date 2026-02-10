from __future__ import annotations

from roboverse_pack.protocol_sim.protocols.unitree_sdk2.profile import UnitreeRobotProfile


def _g1_dof29_profile() -> UnitreeRobotProfile:
    # Order from third_party/unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md (29 DOF).
    motor_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # unitree_mujoco sets mode_machine 5 for the 29-DOF G1 scene.
    return UnitreeRobotProfile(robot_name="g1_dof29", msg_type="hg", motor_names=motor_names, mode_machine=5)


_REGISTRY: dict[str, UnitreeRobotProfile] = {
    "g1_dof29": _g1_dof29_profile(),
}


def get_unitree_profile(robot_name: str) -> UnitreeRobotProfile:
    """Get the Unitree robot profile for a given robot name."""
    try:
        return _REGISTRY[robot_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Unitree profile for robot '{robot_name}'. Available: {sorted(_REGISTRY.keys())}"
        ) from exc
