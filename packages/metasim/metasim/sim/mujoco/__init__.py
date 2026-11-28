# ruff: noqa: F401
"""MuJoCo backend. The headless GL decision (:mod:`metasim.sim.gl`) runs before ``mujoco`` is imported."""

from metasim.sim.gl import apply_headless_gl

apply_headless_gl()

from .mujoco import MujocoHandler
