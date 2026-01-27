from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from metasim.sim.base import BaseSimHandler
from roboverse_pack.robot_protocols.core.types import SimRobotObservation


@dataclass
class SimTiming:
    """Timing configuration for the simulator adapter."""

    dt: float
    realtime: bool = True


class MetaSimAdapter:
    """Single-env adapter around a MetaSim handler.

    This adapter assumes ``scenario.num_envs == 1`` and exposes observations/actions
    in sorted joint-name order to keep behavior consistent across simulators.
    """

    def __init__(self, handler: BaseSimHandler, robot_name: str, timing: SimTiming):
        if handler.num_envs != 1:
            raise ValueError("MetaSimAdapter currently supports only num_envs == 1.")
        self._handler = handler
        self._robot_name = robot_name
        self._timing = timing
        self._last_wall = time.perf_counter()

        self._joint_names_sorted = handler.get_joint_names(robot_name, sort=True)

    @property
    def joint_names_sorted(self) -> list[str]:
        """Get the list of joint names in sorted order."""
        return self._joint_names_sorted

    def read_observation(self) -> SimRobotObservation:
        """Read the current observation from the simulator."""
        ts = self._handler.get_states()
        rs = ts.robots[self._robot_name]

        q = rs.joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=False)
        dq = rs.joint_vel[0].detach().cpu().numpy().astype(np.float32, copy=False)
        tau = None
        if rs.joint_effort_target is not None:
            tau = rs.joint_effort_target[0].detach().cpu().numpy().astype(np.float32, copy=False)

        root_state = rs.root_state[0].detach().cpu().numpy().astype(np.float32, copy=False)

        body_names = rs.body_names if hasattr(rs, "body_names") else None
        body_state = None
        if rs.body_state is not None:
            body_state = rs.body_state[0].detach().cpu().numpy().astype(np.float32, copy=False)

        return SimRobotObservation(
            joint_names_sorted=list(self._joint_names_sorted),
            q_sorted=q,
            dq_sorted=dq,
            tau_sorted=tau,
            root_state=root_state,
            body_names_sorted=body_names,
            body_state=body_state,
        )

    def apply_effort(self, effort_sorted: np.ndarray) -> None:
        """Apply joint efforts to the simulator."""
        if effort_sorted.shape != (len(self._joint_names_sorted),):
            raise ValueError(
                f"Expected effort shape {(len(self._joint_names_sorted),)}, got {tuple(effort_sorted.shape)}"
            )

        # Most handlers accept torch.Tensor (N,J). For single-env, N=1.
        act = torch.as_tensor(effort_sorted, dtype=torch.float32).unsqueeze(0)
        self._handler.set_dof_targets(act)

    def step(self) -> None:
        """Step the simulator forward by one time step."""
        start = time.perf_counter()
        self._handler.simulate()
        if self._timing.realtime:
            elapsed = time.perf_counter() - start
            sleep_s = max(0.0, self._timing.dt - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    def close(self) -> None:
        """Close the simulator adapter."""
        self._handler.close()
