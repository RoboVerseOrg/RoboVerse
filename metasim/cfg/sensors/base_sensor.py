"""Sub-module containing the base sensor configuration."""

from dataclasses import MISSING

from metasim.utils.configclass import configclass


@configclass
class BaseSensorCfg:
    """Base sensor configuration."""

    name: str = MISSING
    """Sensor name"""
