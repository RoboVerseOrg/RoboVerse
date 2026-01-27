from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from roboverse_pack.robot_protocols.core.interfaces import (
    ActuationModel,
    ExternalAssist,
    ProtocolCodec,
    StandbyController,
    Transport,
)
from roboverse_pack.robot_protocols.core.sim_adapter import MetaSimAdapter


@dataclass
class ServerConfig:
    """Configuration for the robot protocol server."""

    dt: float
    realtime: bool = True


@dataclass
class StandbyConfig:
    """Configuration for server-side standby control.

    Motivation: many real robot stacks (including Unitree deploy scripts) do not
    send stabilizing commands immediately at process start. In simulation, this
    can cause the robot to fall before the external controller is "ready".
    """

    enabled: bool = True
    # Consider a command "active" once any of these fields exceed the thresholds.
    active_kp_threshold: float = 1e-3
    active_kd_threshold: float = 1e-3
    active_tau_threshold: float = 1e-3
    # Require N consecutive *active* LowCmd messages before switching from standby
    # to protocol control. Useful to avoid switching on transient commands or
    # partially-initialized controllers.
    required_active_cmds: int = 1
    # If set, consider the command stream stale after this wall-time duration.
    cmd_timeout_s: float | None = None
    # If True, switch back to standby if the command stream becomes stale.
    revert_to_standby_on_timeout: bool = False


class RobotProtocolServer:
    """Generic, single-robot protocol emulation server."""

    def __init__(
        self,
        *,
        adapter: MetaSimAdapter,
        transport: Transport,
        codec: ProtocolCodec,
        actuation: ActuationModel,
        config: ServerConfig,
        standby_controller: StandbyController | None = None,
        standby_config: StandbyConfig | None = None,
        assist: ExternalAssist | None = None,
    ) -> None:
        self._adapter = adapter
        self._transport = transport
        self._codec = codec
        self._actuation = actuation
        self._config = config
        self._standby_controller = standby_controller
        self._standby_config = standby_config or StandbyConfig(enabled=False)
        self._assist = assist

        self._sim_time_s = 0.0
        self._last_cmd = None
        self._last_cmd_wall_s: float | None = None
        self._protocol_active = False
        self._active_cmd_streak = 0

    @property
    def sim_time_s(self) -> float:
        """Get the current simulation time in seconds."""
        return self._sim_time_s

    def start(self) -> None:
        """Start the server."""
        self._transport.start()

    def close(self) -> None:
        """Close the server and cleanup resources."""
        try:
            self._transport.close()
        finally:
            self._adapter.close()

    def _cmd_is_active(self, cmd) -> bool:
        """Heuristic to decide if an external controller has "taken over"."""
        try:
            if cmd.kp is not None and float(np.max(np.abs(cmd.kp))) > float(self._standby_config.active_kp_threshold):
                return True
        except Exception:
            pass
        try:
            if cmd.kd is not None and float(np.max(np.abs(cmd.kd))) > float(self._standby_config.active_kd_threshold):
                return True
        except Exception:
            pass
        try:
            if cmd.tau_ff is not None and float(np.max(np.abs(cmd.tau_ff))) > float(
                self._standby_config.active_tau_threshold
            ):
                return True
        except Exception:
            pass
        return False

    def _should_use_protocol(self, *, have_msg: bool, have_new_msg: bool, cmd_decoded) -> bool:
        if not self._standby_controller or not self._standby_config.enabled:
            # Legacy behavior: as soon as any message arrives, use it (else hold 0).
            return have_msg

        now = time.perf_counter()
        if self._standby_config.cmd_timeout_s is not None and self._standby_config.revert_to_standby_on_timeout:
            if self._last_cmd_wall_s is None or (now - self._last_cmd_wall_s) > float(
                self._standby_config.cmd_timeout_s
            ):
                self._protocol_active = False
                self._active_cmd_streak = 0

        if self._protocol_active:
            return have_msg

        if not have_msg:
            self._active_cmd_streak = 0
            return False

        # Only count new messages toward the "takeover" decision.
        if not have_new_msg:
            return False

        # Wait in standby until we see enough consecutive commands with non-zero gains/torques.
        required = int(getattr(self._standby_config, "required_active_cmds", 1) or 1)
        if required < 1:
            required = 1

        if cmd_decoded is None or not self._cmd_is_active(cmd_decoded):
            self._active_cmd_streak = 0
            return False

        self._active_cmd_streak += 1
        return self._active_cmd_streak >= required

    def run_forever(self) -> None:
        """Run the server loop indefinitely."""
        # Keep wall-time pacing in this loop to match unitree_mujoco behavior.
        next_wall = time.perf_counter()
        dt = float(self._config.dt)

        while True:
            obs = self._adapter.read_observation()

            msg = self._transport.get_latest_command()
            prev_cmd = self._last_cmd
            have_new_msg = msg is not None and msg is not prev_cmd
            cmd_decoded = None
            if have_new_msg:
                self._last_cmd = msg
                self._last_cmd_wall_s = time.perf_counter()
                # Decode once to support command gating (may still be unused).
                try:
                    cmd_decoded = self._codec.decode_command(msg)
                except Exception:
                    cmd_decoded = None

            use_protocol = self._should_use_protocol(
                have_msg=self._last_cmd is not None,
                have_new_msg=have_new_msg,
                cmd_decoded=cmd_decoded,
            )
            if use_protocol:
                if not self._protocol_active:
                    self._protocol_active = True
                    self._active_cmd_streak = 0
                    if self._assist is not None:
                        self._assist.start_release()
                # Ensure we have a decoded command for the last message.
                if cmd_decoded is None:
                    cmd_decoded = self._codec.decode_command(self._last_cmd)
                effort = self._actuation.compute_effort(cmd_decoded, obs)
            else:
                if self._standby_controller and self._standby_config.enabled:
                    effort = self._standby_controller.compute_effort(obs)
                else:
                    # No command yet: hold zero effort.
                    effort = np.zeros((len(obs.joint_names_sorted),), dtype=np.float32)

            self._adapter.apply_effort(effort)
            if self._assist is not None:
                self._assist.apply(obs, dt=dt)
            self._adapter.step()

            self._sim_time_s += dt
            obs_after = self._adapter.read_observation()
            out = self._codec.encode_messages(obs_after, sim_time_s=self._sim_time_s)
            for channel, out_msg in out.items():
                self._transport.publish(channel, out_msg)

            if self._config.realtime:
                next_wall += dt
                sleep_s = max(0.0, next_wall - time.perf_counter())
                if sleep_s:
                    time.sleep(sleep_s)
