"""Balance task for humanoid robots.

TODO: Not Implemented because of collision detection issues.
"""

from metasim.utils import configclass

from .base_cfg import HumanoidTaskCfg


@configclass
class BalanceCfg(HumanoidTaskCfg):
    """Balance task for humanoid robots."""

    episode_length = 1000
