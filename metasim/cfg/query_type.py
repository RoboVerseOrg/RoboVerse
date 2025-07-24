from dataclasses import dataclass


@dataclass
class SitePos:
    """Return worldframe (x,y,z) position of a given site."""

    site: str


@dataclass
class ContactForce:
    """Return 6D contact force/torque measured by a named sensor."""

    sensor_name: str
