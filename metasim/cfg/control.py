"""Configuration classes for control."""

from __future__ import annotations

from typing import Literal

from metasim.utils import configclass


@configclass
class ControlCfg:
    """high level control config parameters cfg.

    This class defines the parameters for the control configuration. Note that stiffness and damping of each joint's joints are configured at robot cfg.
    """

    control_type: Literal["pos", "effort"] = "pos"
    action_scale: float = 1.0
    torque_limit_scale: float = 1.0  # scale it down can ensure safety
    action_offset: bool = False  # set true if target position = action * action_scale + default position
