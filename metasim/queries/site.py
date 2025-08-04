# metasim/queries/mjx_queries.py
from __future__ import annotations

try:
    import jax
    import mujoco

    from metasim.sim.mjx import MJXHandler
    from metasim.sim.mujoco import MujocoHandler
except:
    pass
import torch

from metasim.queries.base import BaseQueryType

# ------------------------ util cache ------------------------
_site_cache = {}


def _get_site_id(model: mujoco.MjModel, name: str) -> int:
    key = id(model)
    site_dict = _site_cache.setdefault(key, {})
    if name not in site_dict:
        site_dict[name] = model.site(name).id
    return site_dict[name]


class SitePos(BaseQueryType):
    """World frame position of a MuJoCo site."""

    # supported_handlers = [MJXHandler, MujocoHandler]

    def __init__(self, site_name: str):
        super().__init__()
        self.site_name = site_name
        self._sid: int | None = None

    def bind_handler(self, handler: MJXHandler, *args, **kwargs):
        """Store site id once, called during handler.launch()."""
        super().bind_handler(handler)
        if isinstance(self.handler, MJXHandler):
            robot_name = handler._robot.name
            full_name = f"{robot_name}/{self.site_name}" if "/" not in self.site_name else self.site_name
            self._sid = _get_site_id(handler._mj_model, full_name)
        elif isinstance(self.handler, MujocoHandler):
            robot_name = handler.robot.name
            full_name = f"{robot_name}/{self.site_name}" if "/" not in self.site_name else self.site_name
            self._sid = _get_site_id(handler.physics.model, full_name)
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for SitePosQuery")

    # ------------------------------------------------------------------ call
    def __call__(self):
        """Return (N_env, 3) site position each time get_extra() is invoked."""
        if isinstance(self.handler, MJXHandler):
            # MJX: site_xpos shape (N_env, N_site, 3)
            val = self.handler._data.site_xpos[:, self._sid]
            return torch.from_dlpack(jax.dlpack.to_dlpack(val))
        elif isinstance(self.handler, MujocoHandler):
            # raw MuJoCo is single‑env; expand to (1,3) for consistency
            pos = self.handler.data.site_xpos[self._sid]
            return torch.as_tensor(pos).unsqueeze(0)  # (1, 3)
        else:
            raise ValueError(f"Unsupported handler type: {type(self.handler)} for SitePosQuery")
