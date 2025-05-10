"""Configuration classes for control."""

from __future__ import annotations

from metasim.utils import configclass


@configclass
class ControlCfg:
    """high level control parameters cfg.

    This class defines the higher level for the control configuration.
    """

    action_scale: float = 1.0 # to scale the actions
    torque_limit_scale: float = 1.0  # scale it down can ensure safety
    action_offset: bool = False  # set true if target position = action * action_scale + default position
