"""Sub-module containing the contact force sensor configuration."""

from __future__ import annotations

from dataclasses import MISSING

from metasim.cfg.sensors import BaseSensorCfg
from metasim.utils.configclass import configclass


@configclass
class ContactForceSensorCfg(BaseSensorCfg):
    """Contact force sensor cfg."""

    base_link: str | tuple[str, str] = MISSING
    """Body link to feel the contact force.
        If a ``str``, the sensor will be attached to the root link of the object specified by the name.
        If a ``tuple[str, str]``, the sensor will be attached to the body link specified by the second str of the object specified by the first str.
    """
    source_link: str | tuple[str, str] | None = None
    """Body link to feel the contact force from.
        If ``None``, the sensor will feel the contact force from all the source links.
        If a ``str``, the sensor will only feel the contact force from the root link of the object specified by the name.
        If a ``tuple[str, str]``, the sensor will only feel the contact force from the body link specified by the second str of the object specified by the first str.
    """
