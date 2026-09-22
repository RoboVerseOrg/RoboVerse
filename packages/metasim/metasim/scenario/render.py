"""Sub-module containing the render configuration."""

from __future__ import annotations

import math
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
    file, or a directory — handlers that support image-based world lighting pick
    one file per launch, deterministically from ``seed`` (vary ``seed`` for
    different picks). Setting this enables photoreal world lighting; if unset,
    handlers fall back to a flat / gradient sky."""

    seed: int = 0
    """Blender Cycles / HDRI seed; does not reseed the training process."""
    denoise: bool = True
    max_bounces: int | None = None
    """Path-tracing ray depth; ``None`` keeps each backend's own default (Cycles 10, Isaac Sim 4)."""
    exposure: float = -0.55
    """Blender display exposure in stops. Isaac Sim uses its own tone mapper."""
    view_transform: str = "Standard"
    """Blender color management; use AgX for photographic highlight rolloff."""
    settle_frames: int = 2
    """Isaac render-only propagation frames before sample accumulation."""
    asset_timeout_s: float = 30.0
    """Maximum wait for pending USD assets in Isaac captures; never steps physics."""

    def __post_init__(self):
        """Reject invalid render budgets before launching a renderer."""
        for name in ("samples", "max_bounces", "settle_frames"):
            value = getattr(self, name)
            if value is None and name in ("samples", "max_bounces"):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"RenderCfg.{name} must be a positive integer, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("RenderCfg.seed must be a nonnegative integer")
        if (
            isinstance(self.exposure, bool)
            or not isinstance(self.exposure, float | int)
            or not math.isfinite(self.exposure)
        ):
            raise ValueError("RenderCfg.exposure must be finite")
        if type(self.denoise) is not bool:
            raise ValueError("RenderCfg.denoise must be a boolean")
        if not isinstance(self.view_transform, str) or not self.view_transform:
            raise ValueError("RenderCfg.view_transform must be a nonempty string")
        if (
            isinstance(self.asset_timeout_s, bool)
            or not isinstance(self.asset_timeout_s, float | int)
            or not math.isfinite(self.asset_timeout_s)
            or self.asset_timeout_s <= 0
        ):
            raise ValueError("RenderCfg.asset_timeout_s must be finite and positive")
