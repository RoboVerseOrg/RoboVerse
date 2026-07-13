# Copyright (c) ManiSkill2-real2sim contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ManiSkill2-real2sim (https://github.com/simpler-env/ManiSkill2_real2sim).
# Changes: from `mani_skill2_real2sim/agents/controllers/pd_joint_vel.py`; import paths rewritten for RoboVerse
#   and line wrapping reflowed by the repo formatter; controller math unchanged (AST-identical to upstream, kept
#   byte-faithful for bitwise parity).
# Full license: roboverse_pack/tasks/simpler_env/_native/control/LICENSE
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from gymnasium import spaces

from .base_controller import BaseController, ControllerConfig


class PDJointVelController(BaseController):
    config: "PDJointVelControllerConfig"

    def _initialize_action_space(self):
        n = len(self.joints)
        low = np.float32(np.broadcast_to(self.config.lower, n))
        high = np.float32(np.broadcast_to(self.config.upper, n))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(0, damping[i], force_limit=force_limit[i], mode=self.config.drive_mode)
            joint.set_friction(friction[i])

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(action[i])


@dataclass
class PDJointVelControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    drive_mode: str = "force"
    normalize_action: bool = True
    controller_cls = PDJointVelController
