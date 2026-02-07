from __future__ import annotations

from dataclasses import dataclass

try:
    import isaacgym
except ImportError:
    pass

from collections import deque

import numpy as np
import torch

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler


@dataclass(frozen=True)
class ContactForcesData:
    """Serializable contact force payload returned via `handler.get_extra()`.

    This object is intentionally small and contains only tensors so it can be
    transferred across process boundaries in `ParallelSimWrapper`.
    """

    contact_forces_history: torch.Tensor
    """Stacked history shaped `(num_envs, history_length, num_bodies, 3)`."""

    contact_forces: torch.Tensor
    """Latest snapshot shaped `(num_envs, num_bodies, 3)`."""


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
        self._pybullet_obj_id: int | None = None
        self._pybullet_client_id: int | None = None
        self._pybullet_body_reindex: list[int] | None = None
        self._mjx_body_ids: list[int] | None = None
        self._mjx_device: str | None = None

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Bind the simulator handler and pre-compute per-robot indexing."""
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        if self.simulator in ["isaacgym", "mujoco"]:
            self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
        elif self.simulator == "isaacsim":
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = torch.tensor(
                [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
                dtype=torch.int,
                device=self.handler.device,
            )
        elif self.simulator == "pybullet":
            robot_name = self.robots[0].name
            self._pybullet_obj_id = int(getattr(self.handler, "object_ids", {}).get(robot_name, -1))
            self._pybullet_client_id = int(getattr(self.handler, "client", -1))
            self._pybullet_body_reindex = self.handler.get_body_reindex(robot_name)
            sorted_body_names = self.handler.get_body_names(robot_name, sort=True)
            self.body_ids_reindex = list(range(len(sorted_body_names)))
        elif self.simulator == "mjx":
            robot_name = self.robots[0].name
            self._mjx_device = str(self.handler.device)
            self._mjx_body_ids = self._resolve_mjx_body_ids(robot_name)
            self.body_ids_reindex = list(range(len(self._mjx_body_ids)))
        elif self.simulator == "newton":
            self.handler.init_contact_sensor(self.robots[0].name)
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = list(range(len(sorted_body_names)))
            self._build_newton_reindex()
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
            elif self.simulator == "pybullet":
                self._current_contact_force = self._get_contact_forces_pybullet()
            elif self.simulator == "mjx":
                self._current_contact_force = self._get_contact_forces_mjx()
            elif self.simulator == "newton":
                self._current_contact_force = self._get_contact_forces_newton()
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
        import mujoco

        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        return contact_forces

    def _resolve_mjx_body_ids(self, robot_name: str) -> list[int]:
        model = getattr(self.handler, "_mjx_model", None)
        if model is None:
            return []
        try:
            from metasim.sim.mjx.mjx_helper import sorted_body_ids
        except Exception:
            return []
        body_ids, _local_names = sorted_body_ids(model, f"{robot_name}/")
        return [int(i) for i in body_ids]

    def _get_contact_forces_mjx(self) -> torch.Tensor:
        """Best-effort contact force proxy for MJX using MuJoCo's `cfrc_ext`.

        Note: `cfrc_ext` contains net external forces on bodies (contact + other externals)
        in MuJoCo coordinates. This is sufficient for common contact-penalty terms.
        """
        robot_name = self.robots[0].name
        if self._mjx_body_ids is None:
            self._mjx_body_ids = self._resolve_mjx_body_ids(robot_name)
        body_ids = self._mjx_body_ids or []
        if not body_ids:
            return torch.zeros((self.num_envs, 0, 3), device=self.handler.device, dtype=torch.float32)

        data = getattr(self.handler, "_data", None)
        if data is None:
            return torch.zeros((self.num_envs, len(body_ids), 3), device=self.handler.device, dtype=torch.float32)

        try:
            forces = data.cfrc_ext[:, body_ids, 0:3]
        except Exception:
            return torch.zeros((self.num_envs, len(body_ids), 3), device=self.handler.device, dtype=torch.float32)

        from metasim.sim.mjx.mjx_helper import j2t

        device = self._mjx_device or str(self.handler.device)
        return j2t(forces, device=device).to(dtype=torch.float32)

    def _get_contact_forces_pybullet(self) -> torch.Tensor:
        """Compute net contact forces on each robot body/link in PyBullet.

        Returns:
            torch.Tensor: shape (num_bodies, 3) in sorted body-name order.
        """
        import pybullet as p

        robot_name = self.robots[0].name
        obj_id = self._pybullet_obj_id
        if obj_id is None or obj_id < 0:
            obj_id = int(getattr(self.handler, "object_ids", {}).get(robot_name, -1))
        if obj_id < 0:
            return torch.zeros((0, 3), device=self.handler.device, dtype=torch.float32)

        client_id = self._pybullet_client_id
        if client_id is None or client_id < 0:
            client_id = int(getattr(self.handler, "client", -1))
        kwargs = {}
        if client_id is not None and client_id >= 0:
            kwargs["physicsClientId"] = int(client_id)

        try:
            num_joints = int(p.getNumJoints(obj_id, **kwargs))
        except TypeError:
            num_joints = int(p.getNumJoints(obj_id))

        forces_origin = torch.zeros((num_joints + 1, 3), device=self.handler.device, dtype=torch.float32)

        def _accumulate(link_index: int, force_w: torch.Tensor) -> None:
            body_idx = 0 if int(link_index) == -1 else int(link_index) + 1
            if 0 <= body_idx < int(forces_origin.shape[0]):
                forces_origin[body_idx] += force_w

        def _force_on_b(contact) -> torch.Tensor:
            # contact[7] is `contactNormalOnB` in world frame, contact[9] is `normalForce`.
            normal_w = torch.tensor(contact[7], device=forces_origin.device, dtype=forces_origin.dtype)
            out = float(contact[9]) * normal_w

            # Friction components (best-effort; fields may not exist in all builds).
            if len(contact) >= 12:
                try:
                    fric1 = float(contact[10])
                    dir1 = torch.tensor(contact[11], device=forces_origin.device, dtype=forces_origin.dtype)
                    out = out + fric1 * dir1
                except Exception:
                    pass
            if len(contact) >= 14:
                try:
                    fric2 = float(contact[12])
                    dir2 = torch.tensor(contact[13], device=forces_origin.device, dtype=forces_origin.dtype)
                    out = out + fric2 * dir2
                except Exception:
                    pass
            return out

        try:
            contacts_a = p.getContactPoints(bodyA=obj_id, **kwargs)
        except TypeError:
            contacts_a = p.getContactPoints(bodyA=obj_id)
        for contact in contacts_a:
            body_b = int(contact[2])
            link_a = int(contact[3])
            link_b = int(contact[4])
            force_on_b = _force_on_b(contact)
            if body_b == obj_id:
                # Self-collision: apply equal and opposite forces to both links.
                _accumulate(link_a, -force_on_b)
                _accumulate(link_b, force_on_b)
            else:
                # Robot is bodyA: force on A is opposite to force on B.
                _accumulate(link_a, -force_on_b)

        try:
            contacts_b = p.getContactPoints(bodyB=obj_id, **kwargs)
        except TypeError:
            contacts_b = p.getContactPoints(bodyB=obj_id)
        for contact in contacts_b:
            body_a = int(contact[1])
            if body_a == obj_id:
                # Self-collision already handled above.
                continue
            link_b = int(contact[4])
            force_on_b = _force_on_b(contact)
            _accumulate(link_b, force_on_b)

        body_reindex = self._pybullet_body_reindex
        if not body_reindex:
            body_reindex = self.handler.get_body_reindex(robot_name)
            self._pybullet_body_reindex = body_reindex
        try:
            return forces_origin[body_reindex]
        except Exception:
            return forces_origin

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
        elif self.simulator == "pybullet":
            self._current_contact_force = self._get_contact_forces_pybullet()
        elif self.simulator == "mjx":
            self._current_contact_force = self._get_contact_forces_mjx()
        elif self.simulator == "newton":
            self._current_contact_force = self._get_contact_forces_newton()
        else:
            raise NotImplementedError
        if self.simulator == "newton":
            self._contact_forces_queue.append(self._map_newton_contact_forces(self._current_contact_force))
        else:
            self._contact_forces_queue.append(
                self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
            )
        # Return a serializable payload instead of the live query object. Returning
        # the query instance would capture unpicklable simulator handles (e.g.
        # IsaacGym/IsaacSim objects) and break multiprocessing.
        return {
            self.robots[0].name: ContactForcesData(
                contact_forces_history=self.contact_forces_history,
                contact_forces=self.contact_forces,
            )
        }

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
            sorted_pairs = sorted(zip(body_names, body_ids), key=lambda pair: pair[0])
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
