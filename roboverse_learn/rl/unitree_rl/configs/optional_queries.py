from __future__ import annotations

from typing import Dict

import torch

from metasim.queries.base import BaseQueryType
import numpy as np

class NetContactForce(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """

    # Supported handlers by module path
    supported_handlers = [
        "metasim.sim.isaacgym.isaacgym",
        "metasim.sim.isaacsim.isaacsim",
    ]

    def __init__(self):
        super().__init__()

    def bind_handler(self, handler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)

    def _for_isaacgym(self) -> Dict[str, torch.Tensor]:
        # Ensure source tensor exists
        from isaacgym import gymtorch

        contact_forces = gymtorch.wrap_tensor(
            self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
        )
        # Refresh once to populate
        # self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        return contact_forces

    def _for_isaacsim(self) -> Dict[str, torch.Tensor]:
        # only for single robot envs for now
        robot_name = self.handler.contact_sensor.cfg.prim_path.split("/")[-2]
        contact_forces = self.handler.contact_sensor.data.net_forces_w
        return {robot_name: contact_forces}

    def _for_mujoco(self) -> Dict[str, torch.Tensor]:
        import mujoco
        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)
        robot_name = self.handler.robot.name
        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        # extend the num_envs dim
        rname = self.handler.robot.name
        model_name = self.handler.mj_objects[rname].model
        body_ids_origin = []
        for bi in range(self.handler.physics.model.nbody):
            bname = self.handler.physics.model.body(bi).name
            if bname.split("/")[0] == model_name and bname != f"{model_name}/":
                body_ids_origin.append(bi)

        # Reindex into alphabetical order of body names
        reindex = self.handler.get_body_reindex(rname)              # indices into body_ids_origin
        ids_sorted = [body_ids_origin[i] for i in reindex]          # global body IDs in sorted order

        # Slice and add batch dim: (1, n_robot_bodies, 3)
        contact_forces = contact_forces[ids_sorted].unsqueeze(0)
        return {robot_name: contact_forces}

    def __call__(self):
        mod = self.handler.__class__.__module__
        if mod.startswith("metasim.sim.isaacgym"):
            return self._for_isaacgym()
        elif mod.startswith("metasim.sim.isaacsim"):
            return self._for_isaacsim()
        elif mod.startswith("metasim.sim.mujoco"):
            return self._for_mujoco()
        else:  # pragma: no cover - other handlers not yet supported
            raise ValueError(f"Unsupported handler type: {type(self.handler)} for NetContactForce query")
