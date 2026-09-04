from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import isaacgym
except ImportError:
    pass

from collections import deque

import numpy as np
import torch

from metasim.queries.base import BaseQueryType

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler

try:
    import mujoco
except ImportError:
    pass


def mujoco_net_contact_forces_world(model, data) -> np.ndarray:
    """Net contact force on every MuJoCo body, ``(nbody, 3)`` float64 in the **world** frame.

    ``mj_contactForce`` reports each contact's wrench in the contact frame (normal, tangent, tangent)
    acting on ``geom2``, and ``data.contact.frame`` holds those three axes as rows in world
    coordinates; the world-frame force is the axes weighted by the components, added to ``geom2``'s
    body with ``geom1``'s body taking the reaction. Summing the contact-frame components directly gave
    a vector of plausible magnitude in the wrong direction, and the old signs gave each body the
    reaction instead of the force it feels; Isaac Sim / Isaac Gym report world-frame forces on the
    body, so this matches them. One vectorised pass, no per-contact torch dispatch.
    """
    nbody = int(model.nbody)
    ncon = int(data.ncon)
    net = np.zeros((nbody, 3), dtype=np.float64)
    if ncon == 0:
        return net
    wrench = np.zeros((ncon, 6), dtype=np.float64)
    for i in range(ncon):
        mujoco.mj_contactForce(getattr(model, "ptr", model), getattr(data, "ptr", data), i, wrench[i])
    frames = np.asarray(data.contact.frame[:ncon], dtype=np.float64).reshape(ncon, 3, 3)
    world = np.einsum("nji,nj->ni", frames, wrench[:, :3])  # rows are axes: f_world = frame^T f_contact
    body1 = np.asarray(model.geom_bodyid)[np.asarray(data.contact.geom1[:ncon])]
    body2 = np.asarray(model.geom_bodyid)[np.asarray(data.contact.geom2[:ncon])]
    np.add.at(net, body2, world)
    np.add.at(net, body1, -world)
    return net


class ContactForces(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """

    def __init__(self, history_length: int = 3):
        super().__init__()
        self.history_length = history_length
        self._current_contact_force = None
        self._contact_forces_queue = deque(maxlen=history_length)
        self._newton_env_ids = None
        self._newton_local_ids = None
        self._newton_valid_mask = None
        self._newton_body_count = None

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Bind the simulator handler and pre-compute per-robot indexing."""
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        if self.simulator in ["isaacgym", "mujoco"]:
            self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
        elif self.simulator == "isaacsim":
            self.handler.init_contact_sensor(self.robots[0].name)
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = torch.tensor(
                [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
                dtype=torch.int,
                device=self.handler.device,
            )
        elif self.simulator == "newton":
            self.handler.init_contact_sensor(self.robots[0].name)
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = list(range(len(sorted_body_names)))
            self._build_newton_reindex()
        elif self.simulator == "superdex":
            # handler.get_contact_forces() already returns bodies in get_body_names(sort=True) order
            self.body_ids_reindex = list(range(len(self.handler.get_body_names(self.robots[0].name, True))))
        else:
            raise NotImplementedError

        self.initialize()
        self.__call__()

    def initialize(self):
        """Warm-start the queue with `history_length` entries."""
        for _ in range(self.history_length):
            if self.simulator == "isaacgym":
                self._current_contact_force = isaacgym.gymtorch.wrap_tensor(
                    self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
                )
            elif self.simulator == "isaacsim":
                self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
            elif self.simulator == "mujoco":
                self._current_contact_force = self._get_contact_forces_mujoco()
            elif self.simulator in ("newton", "superdex"):
                self._current_contact_force = self.handler.get_contact_forces()
            else:
                raise NotImplementedError
            if self.simulator == "newton":
                self._contact_forces_queue.append(self._map_newton_contact_forces(self._current_contact_force))
            else:
                self._contact_forces_queue.append(
                    self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
                )

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """Compute net contact forces on each body.

        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        model, data = self.handler.physics.model, self.handler.physics.data
        forces = mujoco_net_contact_forces_world(model, data)
        return torch.from_numpy(forces).to(device=self.handler.device, dtype=torch.float32)

    def _get_contact_forces_newton(self) -> torch.Tensor:
        """Get contact forces from Newton simulator.

        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        return self.handler.get_contact_forces()

    def __call__(self):
        """Fetch the newest net contact forces and update the queue."""
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self._current_contact_force = self._get_contact_forces_mujoco()
        elif self.simulator in ("newton", "superdex"):
            self._current_contact_force = self.handler.get_contact_forces()
        else:
            raise NotImplementedError
        if self.simulator == "newton":
            self._contact_forces_queue.append(self._map_newton_contact_forces(self._current_contact_force))
        else:
            self._contact_forces_queue.append(
                self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
            )
        return {self.robots[0].name: self}

    @property
    def contact_forces_history(self) -> torch.Tensor:
        """Return stacked history as (num_envs, history_length, num_bodies, 3)."""
        return torch.stack(list(self._contact_forces_queue), dim=1)  # (num_envs, history_length, num_bodies, 3)

    @property
    def contact_forces(self) -> torch.Tensor:
        """Return the latest contact forces snapshot."""
        return self._contact_forces_queue[-1]

    def _build_newton_reindex(self) -> None:
        """Build mapping from Newton contact sensor rows to per-env sorted body indices."""
        if self.handler is None:
            return
        sensor = getattr(self.handler, "_contact_sensor", None)
        model = getattr(self.handler, "_model", None)
        if sensor is None or model is None:
            self._newton_body_count = 0
            self._newton_env_ids = None
            self._newton_local_ids = None
            self._newton_valid_mask = None
            return

        body_worlds = model.body_world.numpy()
        per_env_maps = []
        body_count = None
        for env_id in range(self.num_envs):
            body_ids = self.handler._get_body_indices(env_id, self.robots[0].name)
            if not body_ids:
                per_env_maps.append({})
                if body_count is None:
                    body_count = 0
                continue
            body_names = [model.body_key[idx] for idx in body_ids]
            sorted_pairs = sorted(zip(body_names, body_ids, strict=False), key=lambda pair: pair[0])
            sorted_body_ids = [idx for _, idx in sorted_pairs]
            if body_count is None:
                body_count = len(sorted_body_ids)
            per_env_maps.append({body_idx: local_idx for local_idx, body_idx in enumerate(sorted_body_ids)})

        if body_count is None:
            body_count = 0

        env_ids = []
        local_ids = []
        valid = []
        for row in sensor.sensing_objs:
            body_idx = row[0]
            if not isinstance(body_idx, (int, np.integer)):
                env_ids.append(-1)
                local_ids.append(-1)
                valid.append(False)
                continue
            body_idx = int(body_idx)
            if body_idx < 0 or body_idx >= len(body_worlds):
                env_ids.append(-1)
                local_ids.append(-1)
                valid.append(False)
                continue
            env_id = int(body_worlds[body_idx])
            if env_id < 0 or env_id >= self.num_envs:
                env_ids.append(-1)
                local_ids.append(-1)
                valid.append(False)
                continue
            local_idx = per_env_maps[env_id].get(body_idx)
            if local_idx is None or local_idx >= body_count:
                env_ids.append(env_id)
                local_ids.append(-1)
                valid.append(False)
                continue
            env_ids.append(env_id)
            local_ids.append(local_idx)
            valid.append(True)

        self._newton_env_ids = torch.tensor(env_ids, device=self.handler.device, dtype=torch.long)
        self._newton_local_ids = torch.tensor(local_ids, device=self.handler.device, dtype=torch.long)
        self._newton_valid_mask = torch.tensor(valid, device=self.handler.device, dtype=torch.bool)
        self._newton_body_count = body_count

    def _map_newton_contact_forces(self, forces: torch.Tensor) -> torch.Tensor:
        """Map Newton sensor forces to (num_envs, num_bodies, 3) in sorted body order."""
        if forces is None:
            return torch.zeros((self.num_envs, 0, 3), device=self.handler.device)
        if forces.ndim == 3 and forces.shape[0] == self.num_envs:
            return forces
        if self._newton_body_count is None:
            self._build_newton_reindex()
        body_count = self._newton_body_count or 0
        output = torch.zeros((self.num_envs, body_count, 3), device=forces.device, dtype=forces.dtype)
        if body_count == 0:
            return output
        if self._newton_env_ids is None or self._newton_local_ids is None or self._newton_valid_mask is None:
            return output
        if forces.shape[0] != self._newton_env_ids.shape[0]:
            return output
        mask = self._newton_valid_mask
        if mask.any():
            output[self._newton_env_ids[mask], self._newton_local_ids[mask]] = forces[mask]
        return output
