# Copyright (c) ManiSkill2-real2sim contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ManiSkill2-real2sim (https://github.com/simpler-env/ManiSkill2_real2sim).
# Changes: from `mani_skill2_real2sim/agents/controllers/pd_base_vel.py`; import paths rewritten for RoboVerse;
#   controller math unchanged (AST-identical to upstream, kept byte-faithful for bitwise parity).
# Full license: roboverse_pack/tasks/simpler_env/_native/control/LICENSE
import numpy as np

from ._vendor_utils import rotate_2d_vec_by_angle
from .pd_joint_vel import PDJointVelController, PDJointVelControllerConfig


class PDBaseVelController(PDJointVelController):
    """PDJointVelController for ego-centric base movement."""

    def _initialize_action_space(self):
        # At least support xy-plane translation and z-axis rotation
        assert len(self.joints) >= 3, len(self.joints)
        super()._initialize_action_space()

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        ori = self.qpos[2]
        vel = rotate_2d_vec_by_angle(action[:2], ori)
        new_action = np.hstack([vel, action[2:]])

        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(new_action[i])


class PDBaseVelControllerConfig(PDJointVelControllerConfig):
    controller_cls = PDBaseVelController
