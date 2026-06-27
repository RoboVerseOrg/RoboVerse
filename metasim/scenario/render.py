"""Sub-module containing the render configuration."""

from __future__ import annotations

from typing import Literal

from metasim.utils.configclass import configclass


@configclass
class RenderCfg:
    """Render configuration."""

    mode: Literal["rasterization", "raytracing", "pathtracing", "realtime_pathtracing"] = "raytracing"
    """Rendering technique. Only the IsaacSim backend consumes this; it maps to the
    Omniverse ``/rtx/rendermode`` setting:

    - ``raytracing`` -> ``RaytracedLighting`` (RTX - Real-Time, legacy)
    - ``pathtracing`` -> ``PathTracing`` (RTX - Interactive, offline path tracing)
    - ``realtime_pathtracing`` -> ``RealTimePathTracing`` (RTX - Real-Time 2.0; requires a
      recent Isaac Sim build where that renderer is available)
    - ``rasterization`` is not supported by IsaacSim."""
    samples: int | None = None
    device: str = "AUTO"
    hdri_path: str | None = None
    """Optional HDRI environment map. Either a path to a single ``.hdr`` / ``.exr``
    file, or a directory — handlers that support image-based world lighting will
    pick one file randomly per launch. Setting this enables photoreal world
    lighting; if unset, handlers fall back to a flat / gradient sky."""
