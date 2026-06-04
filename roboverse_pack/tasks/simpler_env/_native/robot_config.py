"""Vendored google_robot controller config — values verbatim from upstream, no upstream import.

Reproduces the deployed control-mode sub-configs from
``mani_skill2_real2sim/agents/configs/google_robot/defaults.py::GoogleRobotDefaultConfig``
for the static base, control mode
``arm_pd_ee_delta_pose_align_interpolate_by_planner
_gripper_pd_joint_target_delta_pos_interpolate_by_planner``. The values here are checked
bit-identical against the upstream config dict by spike_combined_controller_parity (M6).
"""

from __future__ import annotations

import numpy as np

from .control import PDEEPoseControllerConfig, PDJointPosMimicControllerConfig

ARM_JOINT_NAMES = [
    "joint_torso",
    "joint_shoulder",
    "joint_bicep",
    "joint_elbow",
    "joint_forearm",
    "joint_wrist",
    "joint_gripper",
    "joint_head_pan",
    "joint_head_tilt",
]
GRIPPER_JOINT_NAMES = ["joint_finger_right", "joint_finger_left"]
EE_LINK_NAME = "link_gripper_tcp"

ARM_STIFFNESS = [
    1700.0,
    1737.0471680861954,
    979.975871856535,
    930.0,
    1212.154500274304,
    432.96500923932535,
    468.0013365498738,
    2000,
    2000,
]
ARM_DAMPING = [
    1059.9791902443303,
    1010.4720585373592,
    767.2803161582076,
    680.0,
    674.9946964336588,
    274.613381336198,
    340.532560578637,
    900,
    900,
]
ARM_FORCE_LIMIT = [300, 300, 100, 100, 100, 100, 100, 100, 100]
ARM_FRICTION = 0.0
ARM_VEL_LIMIT, ARM_ACC_LIMIT, ARM_JERK_LIMIT = 1.5, 2.0, 50.0

GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_FORCE_LIMIT = 200, 8, 60
GRIPPER_VEL_LIMIT, GRIPPER_ACC_LIMIT, GRIPPER_JERK_LIMIT = 1.0, 7.0, 50.0

CONTROL_MODE = (
    "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
)


def google_robot_deployed_controller_configs():
    """Return {"arm": ..., "gripper": ...} for the deployed control mode (vendored)."""
    arm = PDEEPoseControllerConfig(
        ARM_JOINT_NAMES,
        -1.0,
        1.0,
        np.pi / 2,  # pos_lower, pos_upper, rot_bound
        ARM_STIFFNESS,
        ARM_DAMPING,
        ARM_FORCE_LIMIT,
        friction=ARM_FRICTION,
        ee_link=EE_LINK_NAME,
        frame="ee_align",
        interpolate=True,
        interpolate_by_planner=True,
        interpolate_planner_vlim=ARM_VEL_LIMIT,
        interpolate_planner_alim=ARM_ACC_LIMIT,
        interpolate_planner_jerklim=ARM_JERK_LIMIT,
        normalize_action=False,
        drive_mode="force",
    )
    gripper = PDJointPosMimicControllerConfig(
        GRIPPER_JOINT_NAMES,
        -1.3 - 0.01,
        1.3 + 0.01,  # lower, upper
        GRIPPER_STIFFNESS,
        GRIPPER_DAMPING,
        GRIPPER_FORCE_LIMIT,
        use_delta=True,
        use_target=True,
        clip_target=True,
        clip_target_thres=0.01,
        normalize_action=True,
        drive_mode="force",
        interpolate=True,
        interpolate_by_planner=True,
        interpolate_planner_exec_set_target_vel=True,
        delta_target_from_last_drive_target=True,
        small_action_repeat_last_target=True,
        interpolate_planner_vlim=GRIPPER_VEL_LIMIT,
        interpolate_planner_alim=GRIPPER_ACC_LIMIT,
        interpolate_planner_jerklim=GRIPPER_JERK_LIMIT,
    )
    return {"arm": arm, "gripper": gripper}
