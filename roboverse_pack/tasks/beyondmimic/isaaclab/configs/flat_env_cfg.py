# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: import paths rewired to RoboVerse; upstream's G1FlatWoStateEstimationEnvCfg/G1FlatLowFreqEnvCfg replaced by G1FlatEnvCfgDeploy, which uses the delayed-actuator G1 config; docstrings added.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from isaaclab.utils import configclass

from roboverse_pack.tasks.beyondmimic.isaaclab.configs.tracking_env_cfg import TrackingEnvCfg
from roboverse_pack.tasks.beyondmimic.isaaclab.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from roboverse_pack.tasks.beyondmimic.isaaclab.robots.g1_delayed import G1_DELAYED_CYLINDER_CFG


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """Configuration for the G1 flat environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatEnvCfgDeploy(G1FlatEnvCfg):
    """Configuration for the G1 flat environment for deployment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_DELAYED_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        delattr(self.observations.policy, "base_lin_vel")
        delattr(self.observations.policy, "motion_anchor_pos_b")
