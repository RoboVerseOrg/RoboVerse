from __future__ import annotations

import numpy as np

from metasim.protocol_sim.core.interfaces import ActuationModel
from metasim.protocol_sim.core.types import CanonicalRobotCommand, SimRobotObservation


class UnitreeLowCmdActuationModel(ActuationModel):
    """Compute joint torques from Unitree LowCmd-style impedance fields."""

    def __init__(
        self,
        *,
        protocol_to_sorted: list[int],
        torque_limits_protocol: np.ndarray | None = None,
        kp_override_protocol: np.ndarray | None = None,
        kd_override_protocol: np.ndarray | None = None,
    ) -> None:
        self._protocol_to_sorted = np.asarray(protocol_to_sorted, dtype=np.int64)
        self._torque_limits_protocol = torque_limits_protocol
        self._kp_override_protocol = kp_override_protocol
        self._kd_override_protocol = kd_override_protocol

    def compute_effort(self, cmd: CanonicalRobotCommand, obs: SimRobotObservation) -> np.ndarray:
        """Compute joint efforts based on the command and observation."""
        # Map measured state into protocol motor order.
        q = obs.q_sorted[self._protocol_to_sorted]
        dq = obs.dq_sorted[self._protocol_to_sorted]

        n = len(self._protocol_to_sorted)
        tau = np.zeros((n,), dtype=np.float32)

        if cmd.tau_ff is not None:
            tau += cmd.tau_ff.astype(np.float32, copy=False)

        kp = self._kp_override_protocol if self._kp_override_protocol is not None else cmd.kp
        kd = self._kd_override_protocol if self._kd_override_protocol is not None else cmd.kd

        if cmd.q_des is not None and kp is not None:
            tau += kp.astype(np.float32, copy=False) * (cmd.q_des.astype(np.float32, copy=False) - q)

        if cmd.qd_des is not None and kd is not None:
            tau += kd.astype(np.float32, copy=False) * (cmd.qd_des.astype(np.float32, copy=False) - dq)

        # If the protocol exposes a per-motor mode, treat mode==0 as "disabled".
        if cmd.mode is not None:
            try:
                enabled = cmd.mode.astype(np.int32, copy=False) != 0
                tau = tau * enabled.astype(np.float32)
            except Exception:
                pass

        if self._torque_limits_protocol is not None:
            lim = self._torque_limits_protocol.astype(np.float32, copy=False)
            tau = np.clip(tau, -lim, lim)

        # Scatter torques back into sorted simulator joint order.
        out = np.zeros((len(obs.joint_names_sorted),), dtype=np.float32)
        out[self._protocol_to_sorted] = tau
        return out
