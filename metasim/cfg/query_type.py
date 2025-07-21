from dataclasses import dataclass

@dataclass
class SitePos:
    """Return world‑frame (x, y, z) position of a given site."""
    site: str


@dataclass
class ContactForce:
    """Return 6‑D contact force/torque measured by a named sensor."""
    sensor_name: str
