# metasim/queries/contact_queries.py
from __future__ import annotations

import importlib

from metasim.queries.base import BaseQueryType


class ContactData(BaseQueryType):
    """MuJoCo contact data query (works for MJX & raw MuJoCo)."""

    def __init__(self, geom1_name: str, geom2_name: str):
        super().__init__()
        self.geom1_name = geom1_name
        self.geom2_name = geom2_name
        self._geom1_id: int | None = None
        self._geom2_id: int | None = None

    def bind_handler(self, handler, *args, **kwargs):
        """Remember the geom IDs once the handler is known."""
        super().bind_handler(handler, *args, **kwargs)
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.mjx"):
            # Use geom names as-is (no automatic robot name prefix)
            self._geom1_id = handler._mj_model.geom(self.geom1_name).id
            self._geom2_id = handler._mj_model.geom(self.geom2_name).id

        elif mod.startswith("metasim.sim.mujoco"):
            # Use geom names as-is (no automatic robot name prefix)
            self._geom1_id = handler.physics.model.geom(self.geom1_name).id
            self._geom2_id = handler.physics.model.geom(self.geom2_name).id

        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for ContactData query")

    def __call__(self):
        """Return contact info of the two geom."""
        mod = self.handler.__class__.__module__

        if mod.startswith("metasim.sim.mjx"):
            torch = importlib.import_module("torch")
            jax = importlib.import_module("jax")

            contacts = self.handler._data.contact  # batched
            # [num_env, nconmax]
            pair_mask = ((contacts.geom1 == self._geom1_id) & (contacts.geom2 == self._geom2_id)) | (
                (contacts.geom1 == self._geom2_id) & (contacts.geom2 == self._geom1_id)
            )
            dist_mask = contacts.dist < 1e-6
            mask = pair_mask & dist_mask  # [num_env, nconmax]
            has_contact = jax.numpy.any(mask, axis=1)  # [num_env]
            has_contact = torch.from_dlpack(has_contact.__dlpack__())
            return has_contact.to(dtype=torch.float32, device=self.handler.device)

        elif mod.startswith("metasim.sim.mujoco"):
            torch = importlib.import_module("torch")
            import numpy as np

            nenv = self.handler.num_envs
            has_contact = np.zeros(nenv, dtype=bool)
            data = self.handler.data
            for i in range(data.ncon):
                con = data.contact[i]
                if (con.geom1 == self._geom1_id and con.geom2 == self._geom2_id) or (
                    con.geom1 == self._geom2_id and con.geom2 == self._geom1_id
                ):
                    if con.dist < 1e-6:
                        has_contact = True
                        break

            return torch.tensor(has_contact, dtype=torch.float32, device=self.handler.device)

        raise ValueError(f"Unsupported handler type: {type(self.handler)} for ContactData query")
