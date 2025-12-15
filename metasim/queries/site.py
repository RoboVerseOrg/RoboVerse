# metasim/queries/mjx_queries.py
from __future__ import annotations

import importlib

from metasim.queries.base import BaseQueryType

_site_cache: dict[int, dict[str, int]] = {}


def _get_site_id(mj_model, name: str) -> int:
    key = id(mj_model)
    site_dict = _site_cache.setdefault(key, {})
    if name not in site_dict:
        site_dict[name] = mj_model.site(name).id
    return site_dict[name]


class SitePos(BaseQueryType):
    """World-frame position of a MuJoCo site (works for MJX & raw MuJoCo)."""

    def __init__(self, site_name: str):
        super().__init__()
        self.site_name = site_name
        self._sid: int | None = None  # site id resolved during bind

    def bind_handler(self, handler, *args, **kwargs):
        """Remember the site-id once the handler is known."""
        super().bind_handler(handler, *args, **kwargs)  # Remember to call the base method
        mod = handler.__class__.__module__
        if mod.startswith("metasim.sim.mjx"):
            # Use the MJCF identifier which matches site name prefixes in the model
            robot_prefix = handler._mujoco_robot_name
            full_name = f"{robot_prefix}{self.site_name}" if "/" not in self.site_name else self.site_name
            self._sid = _get_site_id(handler._mj_model, full_name)

        elif mod.startswith("metasim.sim.mujoco"):
            # Use the MJCF identifier which matches site name prefixes in the model
            robot_prefix = handler._mujoco_robot_names[0]
            full_name = f"{robot_prefix}{self.site_name}" if "/" not in self.site_name else self.site_name
            self._sid = _get_site_id(handler.physics.model, full_name)

        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for SitePos query")

    def __call__(self):
        """Return (N_env, 3) site position whenever `get_extra()` is invoked.

        * Heavy libraries are imported **inside** the relevant branch only.
        """
        mod = self.handler.__class__.__module__

        if mod.startswith("metasim.sim.mjx"):
            # ── MJX branch ────────────────────────────────────────────────
            torch = importlib.import_module("torch")

            # Check if batched MJX data is initialized
            if hasattr(self.handler, "_data") and self.handler._data is not None:
                from metasim.sim.mjx.mjx_helper import j2t

                val = self.handler._data.site_xpos[:, self._sid]
                return j2t(val)
            else:
                # Fall back to render_data (always available after launch)
                import mujoco

                mujoco.mj_forward(self.handler._mj_model, self.handler._render_data)
                pos = self.handler._render_data.site_xpos[self._sid]
                # Return as (1, 3) since we only have single-env data from render_data
                return torch.as_tensor(pos).unsqueeze(0)

        elif mod.startswith("metasim.sim.mujoco"):
            # ── raw MuJoCo branch ────────────────────────────────────────
            torch = importlib.import_module("torch")

            pos = self.handler.data.site_xpos[self._sid]
            return torch.as_tensor(pos).unsqueeze(0)

        raise ValueError(f"Unsupported handler type: {type(self.handler)} for SitePos query")
