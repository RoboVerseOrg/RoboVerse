from __future__ import annotations

from metasim.utils import configclass

from .h1_cfg import H1Cfg


@configclass
class H1Cfg(H1Cfg):
    name: str = "h1_body_collision"
    mjx_mjcf_path: str = "roboverse_data/robots/h1/mjcf/mjx_h1_body.xml"

    