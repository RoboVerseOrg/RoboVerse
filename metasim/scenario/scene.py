from __future__ import annotations

from metasim.utils.configclass import configclass


@configclass
class SceneCfg:
    """Base config class for scenes."""

    name: str | None = None
    usd_path: str | None = None
    urdf_path: str | None = None
    mjcf_path: str | None = None

    positions: list[tuple[float, float, float]] | None = None
    default_position: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None

    scale: tuple[float, float, float] | None = None


@configclass
class GSSceneCfg:
    """Base config class for GS scenes."""

    with_gs_background: bool = False
    gs_background_path: str | None = None

    # format: (x, y, z, qx, qy, qz, qw)
    gs_background_pose_tum: tuple[float, float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0, 1)
