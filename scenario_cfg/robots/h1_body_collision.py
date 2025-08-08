from __future__ import annotations

from metasim.utils import configclass

from .h1_cfg import H1Cfg


@configclass
class H1BodyCollisionCfg(H1Cfg):
    # This configuration enables the full-body collision model for the H1 humanoid robot.
    # 1. The corresponding MJCF model includes collision geoms for the entire body,
    #    with simplified shapes to reduce simulation cost. However, it is still
    #    significantly more computationally expensive than the default "feet-only" version,
    #    especially under the MJX backend.
    # 2. This configuration is necessary for certain tasks in HumanoidBench (e.g., "sit", "crawl")
    #    that involve meaningful contact between limbs or torso and the environment.
    #    Without enabling full-body collisions, such behaviors may become physically invalid
    #    or lead to unstable simulation artifacts.
    name: str = "h1"
    mjx_mjcf_path: str = "roboverse_data/robots/h1/mjcf/mjx_h1_body.xml"
