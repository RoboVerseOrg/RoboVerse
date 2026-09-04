# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: MetaSim query types (e.g. per-body contact forces) written for the port to replace Isaac Lab's sensor/manager access; no single upstream file corresponds to it.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from collections import deque

import torch

from metasim.sim.base import BaseQueryType, BaseSimHandler

try:
    import isaacgym
except ImportError:
    pass


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

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Bind handler to the query."""
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
        else:
            raise NotImplementedError
        self.initialize()
        self.__call__()

    def initialize(self):
        """Initialize the query."""
        for _ in range(self.history_length):
            if self.simulator == "isaacgym":
                self._current_contact_force = isaacgym.gymtorch.wrap_tensor(
                    self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
                )
            elif self.simulator == "isaacsim":
                self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
            elif self.simulator == "mujoco":
                self._current_contact_force = self._get_contact_forces_mujoco()
            else:
                raise NotImplementedError
            self._contact_forces_queue.append(
                self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
            )

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """World-frame net contact force per body; the frame / sign convention lives in one place."""
        from metasim.queries.contact_force import mujoco_net_contact_forces_world

        forces = mujoco_net_contact_forces_world(self.handler.physics.model, self.handler.physics.data)
        return torch.from_numpy(forces).to(device=self.handler.device, dtype=torch.float32)

    def __call__(self):
        """Call the query."""
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self._current_contact_force = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError
        self._contact_forces_queue.append(
            self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
        )
        return {self.robots[0].name: self}

    @property
    def contact_forces_history(self) -> torch.Tensor:
        """Get the contact forces history."""
        return torch.stack(list(self._contact_forces_queue), dim=1)  # (num_envs, history_length, num_bodies, 3)

    @property
    def contact_forces(self) -> torch.Tensor:
        """Get the current contact forces."""
        return self._contact_forces_queue[-1]
