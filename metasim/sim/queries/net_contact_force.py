from __future__ import annotations
from metasim.queries.base import BaseQueryType


class NetContactForce(BaseQueryType):

    def __init__(self, site_name: str):
        super().__init__()

    def bind_handler(self, handler, *args, **kwargs):
        """Remember the site-id once the handler is known."""
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            robot_name = handler._robot.name
            full_name = f"{robot_name}/{self.site_name}" if "/" not in self.site_name else self.site_name

        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for SitePos query")

    def __call__(self):
        """Return (num_env, num_bodies) contact force whenever `get_extra()` is invoked.

        """
        mod = self.handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            return self.handler.contact_sensor
        else:
            raise ValueError(f"Unsupported handler type: {type(self.handler)} for NetContactForce query")
        