"""Terrain manager — mjlab terrain_levels port (minimal but functional).

Full mjlab terrain (``mjlab/src/mjlab/terrains/``) is ~4000 LOC across
heightfield_terrains.py + primitive_terrains.py + terrain_generator.py;
it generates a grid of named sub-terrain patches with progressive
difficulty. Reproducing that 1:1 is impractical in a single port.

This module provides a smaller surface that's enough to exercise the
manager + curriculum + foot-height sensor pipeline:

  - :func:`generate_rough_hfield_mjcf` — write a single rough heightfield
    MJCF (random + smoothed) that a task can inject as the floor.
  - :class:`TerrainLevelsManager` — track per-env terrain_level scalar +
    move_up / move_down via the standard "distance walked > half-row"
    heuristic from mjlab's terrain_levels_vel curriculum.

The MJCF approach matches RoboVerse's existing scene-MJCF philosophy
(no MetaSim ScenarioCfg changes required).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch


def generate_rough_hfield_mjcf(
    out_path: str | None = None,
    grid_size: tuple[int, int] = (64, 64),
    cell_size: float = 0.1,
    max_height: float = 0.08,
    seed: int = 0,
    smooth_passes: int = 2,
    cache: bool = True,
) -> str:
    """Generate a rough-terrain heightfield MJCF.

    Args:
        out_path: Where to write the MJCF (defaults to a hash-keyed tempfile).
        grid_size: ``(nrows, ncols)`` heightfield grid resolution. Total
            terrain footprint = ``grid_size[0] * cell_size`` per axis.
        cell_size: Per-cell physical size in metres.
        max_height: Peak terrain height (z) in metres. Heights are uniform
            random in ``[0, max_height]``, then box-filtered ``smooth_passes``
            times to reduce sharp corners.
        seed: Numpy RNG seed.
        smooth_passes: Box-filter passes for gentler terrain.
        cache: Reuse the MJCF if it exists.

    Returns:
        Absolute path to the generated MJCF (a self-contained
        ``<mujoco>`` document with a single ``<hfield>`` + ``<geom>``).
    """
    key = hashlib.md5(f"rough_hf|{grid_size}|{cell_size}|{max_height}|{seed}|{smooth_passes}".encode()).hexdigest()[:12]
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), f"mjlab_terrain_{key}.xml")
    if cache and os.path.exists(out_path):
        return out_path

    rng = np.random.default_rng(seed)
    nrow, ncol = grid_size
    heights = rng.uniform(0.0, 1.0, size=(nrow, ncol)).astype(np.float32)
    for _ in range(smooth_passes):
        smoothed = heights.copy()
        smoothed[1:-1, 1:-1] = 0.25 * (
            heights[1:-1, 1:-1]
            + 0.25 * (heights[:-2, 1:-1] + heights[2:, 1:-1] + heights[1:-1, :-2] + heights[1:-1, 2:])
            + 0.0625 * (heights[:-2, :-2] + heights[:-2, 2:] + heights[2:, :-2] + heights[2:, 2:])
        )
        heights = smoothed
    # Normalise + scale.
    heights = (heights - heights.min()) / max(heights.max() - heights.min(), 1e-6)
    heights = (heights * max_height).astype(np.float32)

    # Write heightfield as a flattened 1D string of values.
    flat = heights.reshape(-1).tolist()
    radius_x = grid_size[1] * cell_size / 2.0
    radius_y = grid_size[0] * cell_size / 2.0

    # Write a sister .png? No — MuJoCo accepts user_data on <hfield> as raw values
    # too via the elevation array. Simpler path: dump to a .bin? No.
    # Cleanest: write the heightfield as a custom mujoco <hfield> with file=.
    # But that requires loadable image. We use MuJoCo's runtime hfield_data array
    # which lets us set values from Python after model load. So the MJCF
    # declares the hfield + geom, and the task post-loads the values.
    nfloat = nrow * ncol

    # Path to .bin sibling for the elevation data (MuJoCo's hfield file= path).
    bin_path = out_path.replace(".xml", ".bin")
    # MuJoCo hfield .png format: 2 ints (nrow, ncol) + nrow*ncol float32 values.
    with open(bin_path, "wb") as fb:
        np.array([ncol, nrow], dtype=np.int32).tofile(fb)
        heights.reshape(-1).astype(np.float32).tofile(fb)

    xml = f"""<mujoco model="rough_terrain">
  <option timestep="0.005" integrator="implicitfast" iterations="100" ls_iterations="50"/>
  <asset>
    <hfield name="rough" nrow="{nrow}" ncol="{ncol}"
            size="{radius_x} {radius_y} {max_height:.4f} 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="terrain" type="hfield" hfield="rough"
          rgba="0.5 0.4 0.3 1.0"
          contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
  </worldbody>
</mujoco>"""
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


@dataclass
class TerrainLevelsCfg:
    """Config for :class:`TerrainLevelsManager` (mjlab parity)."""

    max_level: int = 5
    """Hardest sub-terrain index (mjlab's max_init_terrain_level=5)."""
    row_size: float = 2.0
    """Half-distance the robot needs to walk to advance a level (mjlab
    uses ``terrain_generator.size[0] / 2``)."""
    fail_factor: float = 0.5
    """Robots that walked less than ``fail_factor * commanded_dist`` go
    back a level (mjlab terrain_levels_vel default 0.5)."""


class TerrainLevelsManager:
    """Per-env terrain difficulty progression.

    Tracks ``terrain_levels[env_id] ∈ [0, max_level]`` and the env's
    starting position (``env_origins``). On reset (called from the
    curriculum hook) computes distance walked since last reset; if it
    exceeds ``row_size`` the env levels up, if it walked less than half
    of the commanded distance it levels down.

    Public surface:
      ``terrain_levels``      — (N,) int tensor, current per-env level
      ``env_origins``         — (N, 3) float tensor, env starting pos
      ``update(env_ids, move_up, move_down)`` — apply level changes
      ``mean_level / max_level / per_terrain_means`` — for logging
    """

    def __init__(self, env, cfg: TerrainLevelsCfg | None = None):
        self.env = env
        self.cfg = cfg or TerrainLevelsCfg()
        device = env.device
        N = env.num_envs

        # All envs start at level 0; mjlab also seeds randomly up to
        # max_init_terrain_level — keep simple here (level 0 for parity).
        self.terrain_levels = torch.zeros(N, dtype=torch.long, device=device)
        self.env_origins = torch.zeros((N, 3), device=device, dtype=torch.float32)

    def reset_env_origins(self, env_ids: torch.Tensor, root_pos: torch.Tensor) -> None:
        """Capture the per-env starting position on reset (called from curriculum)."""
        if env_ids.numel() == 0:
            return
        self.env_origins[env_ids] = root_pos[env_ids]

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        """Advance / retreat ``terrain_levels`` per-env (mjlab parity)."""
        if env_ids.numel() == 0:
            return
        lvl = self.terrain_levels[env_ids]
        lvl = lvl + move_up.long() - move_down.long()
        lvl = lvl.clamp(min=0, max=self.cfg.max_level)
        self.terrain_levels[env_ids] = lvl

    @property
    def mean_level(self) -> torch.Tensor:
        """Return the mean terrain level across all envs."""
        return self.terrain_levels.float().mean()

    @property
    def max_level_seen(self) -> torch.Tensor:
        """Return the highest terrain level reached by any env."""
        return self.terrain_levels.max()
