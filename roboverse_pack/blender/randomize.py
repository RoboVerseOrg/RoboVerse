"""Lightweight pose / camera / DOF samplers used by the Blender demo.

Pure-Python utilities — no ``bpy`` dependency. They produce numeric tuples
that callers feed to whatever simulator / handler API they're using.

If these prove useful beyond the Blender demo, consider promoting them
into ``metasim.randomization`` as building blocks for higher-level
``Randomizer`` classes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class HemisphereCfg:
    """Hemispherical camera sampling around a target point."""

    target: tuple[float, float, float] = (0.0, 0.0, 0.4)
    radius_range: tuple[float, float] = (1.2, 2.0)
    elev_deg_range: tuple[float, float] = (15.0, 60.0)
    azim_deg_range: tuple[float, float] = (0.0, 360.0)
    look_at_jitter: float = 0.05


def sample_camera_hemisphere(
    cfg: HemisphereCfg, rng: random.Random | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (camera_pos, look_at) sampled on a hemisphere around cfg.target."""
    rng = rng or random
    r = rng.uniform(*cfg.radius_range)
    elev = math.radians(rng.uniform(*cfg.elev_deg_range))
    azim = math.radians(rng.uniform(*cfg.azim_deg_range))
    cx, cy, cz = cfg.target
    pos = (
        cx + r * math.cos(elev) * math.cos(azim),
        cy + r * math.cos(elev) * math.sin(azim),
        cz + r * math.sin(elev),
    )
    j = cfg.look_at_jitter
    look_at = (
        cx + rng.uniform(-j, j),
        cy + rng.uniform(-j, j),
        cz + rng.uniform(-j, j),
    )
    return pos, look_at


@dataclass
class PoseBoxCfg:
    """Random object pose sampled inside an axis-aligned box, free yaw."""

    xyz_min: tuple[float, float, float] = (-0.3, -0.3, 0.04)
    xyz_max: tuple[float, float, float] = (0.3, 0.3, 0.04)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)


def sample_pose_box(
    cfg: PoseBoxCfg, rng: random.Random | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return (pos, quat_wxyz) — rotation is yaw-only about world +Z."""
    rng = rng or random
    pos = tuple(rng.uniform(lo, hi) for lo, hi in zip(cfg.xyz_min, cfg.xyz_max))
    yaw = rng.uniform(*cfg.yaw_range)
    half = yaw * 0.5
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    return pos, quat


def sample_dof_pos(
    joint_limits: dict[str, tuple[float, float]],
    ratio_range: tuple[float, float] = (0.1, 0.9),
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Pick a random valid pose for every joint as a fraction of its limit range.

    ``ratio_range`` clamps the lerp factor so the robot stays away from joint
    limits; (0, 1) would let it ride the bounds.
    """
    rng = rng or random
    out: dict[str, float] = {}
    for jn, (lo, hi) in joint_limits.items():
        t = rng.uniform(*ratio_range)
        out[jn] = float(lo + t * (hi - lo))
    return out


def disjoint_xy_samples(
    n: int,
    xy_min: tuple[float, float],
    xy_max: tuple[float, float],
    min_dist: float = 0.1,
    max_tries: int = 200,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Sample ``n`` 2D points in a rectangle, rejecting any closer than min_dist.

    Useful for placing several objects on a table without overlap.
    """
    rng = rng or random
    pts: list[tuple[float, float]] = []
    for _ in range(n):
        x = y = 0.0
        for _try in range(max_tries):
            x = rng.uniform(xy_min[0], xy_max[0])
            y = rng.uniform(xy_min[1], xy_max[1])
            if all(math.hypot(x - px, y - py) >= min_dist for px, py in pts):
                pts.append((x, y))
                break
        else:
            pts.append((x, y))
    return pts


def quat_from_euler_xyz(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert intrinsic XYZ Euler angles (radians) to quaternion (w, x, y, z)."""
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def numpy_seed_from(seed: int | None) -> np.random.Generator:
    """Seeded numpy RNG for any caller that wants a Generator instead of random.Random."""
    return np.random.default_rng(seed)
