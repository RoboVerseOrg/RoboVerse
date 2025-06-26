"""Sub-module containing the randomization configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils.configclass import configclass


@configclass
class FrictionRandomCfg:
    """Friction randomization configuration."""

    enabled: bool = False
    """Whether to randomize friction"""
    range: tuple[float, float] = MISSING  # TODO: Add dynamic and static randomization option
    """Friction sampling range"""
    num_buckets: int = MISSING
    """Distribution buckets"""
    dist_fn: callable = MISSING
    """Distribution function, usually uniform or normal"""


@configclass
class MassRandomCfg:
    """Mass randomization configuration."""

    enabled: bool = False
    """Whether to randomize mass"""
    range: tuple[float, float] = MISSING
    """Mass sampling range"""
    dist_fn: callable = MISSING
    """Distribution sampling function for all env, usually uniform or normal"""


@configclass
class RandomizationCfg:
    """Randomization configuration."""

    camera: bool = False
    """Randomize camera pose"""
    light: bool = False
    """Randomize light direction, temperature, intensity"""
    ground: bool = False
    """Randomize ground"""
    reflection: bool = False
    """Randomize reflection (roughness, metallic, reflectance), attach random diffuse color to surfaces that have no material"""
    table: bool = False
    """Randomize table albedo"""
    wall: bool = False
    """Add wall and roof, randomize wall"""
    scene: bool = False
    """Randomize scene"""
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level"""

    friction: FrictionRandomCfg = FrictionRandomCfg()
    """Friction randomization configuration for rigid bodies"""
    mass: MassRandomCfg = MassRandomCfg()
    """Mass randomization configuration for rigid bodies"""

    def __post_init__(self):
        """Post-initialization configuration."""
        assert self.level in [0, 1, 2, 3]
        if self.level >= 0:
            pass
        if self.level >= 1:
            self.table = True
            self.ground = True
            self.wall = True
        if self.level >= 2:
            self.camera = True
        if self.level >= 3:
            self.light = True
            self.reflection = True
