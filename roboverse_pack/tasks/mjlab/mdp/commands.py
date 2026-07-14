# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: port of mjlab's `tasks/velocity/mdp/velocity_command.py` and `tasks/manipulation/mdp/commands.py` to
#   standalone torch command managers driven by RoboVerse's `ManagerBasedRVEnv`.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Command managers — mjlab Command* port.

Contents:
  - ``VelocityCommandManager`` — per-env ``(lin_vel_x, lin_vel_y, ang_vel_z)``
    sampler. Used by velocity-tracking rewards.
  - ``LiftingCommandManager`` — per-env 3-D target position for the
    object being lifted. Used by ``bring_object_reward`` and
    ``staged_position_reward`` in lift_cube tasks.

mjlab sources:
  src/mjlab/tasks/velocity/mdp/velocity_command.py
  src/mjlab/tasks/manipulation/mdp/commands.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class VelocityCommandRanges:
    """Sampling ranges (mjlab UniformVelocityCommandCfg.Ranges)."""

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] = (-math.pi, math.pi)


@dataclass
class VelocityCommandCfg:
    """Per-env velocity command sampler config (mjlab UniformVelocityCommandCfg)."""

    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rel_standing_envs: float = 0.1
    """Fraction of envs that get zero command (standing in place)."""
    rel_heading_envs: float = 0.3
    """Fraction with heading control instead of explicit ang_vel_z."""
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    ranges: VelocityCommandRanges = field(default_factory=VelocityCommandRanges)


class VelocityCommandManager:
    """Per-env (lin_vel_x, lin_vel_y, ang_vel_z) sampler.

    Stored on env as ``env._velocity_command``. Updated each env step via
    :meth:`update`. ``current()`` returns ``(num_envs, 3)`` tensor of current
    commands.

    Use via:
      env._velocity_command = VelocityCommandManager(env, VelocityCommandCfg())
      # in env._step: env._velocity_command.update()
      # in reward:    cmd = env._velocity_command.current()
    """

    def __init__(self, env, cfg: VelocityCommandCfg | None = None):
        self.env = env
        self.cfg = cfg or VelocityCommandCfg()
        device = env.device
        N = env.num_envs

        self._command = torch.zeros((N, 3), device=device, dtype=torch.float32)
        self._heading_target = torch.zeros((N,), device=device, dtype=torch.float32)
        self._is_standing = torch.zeros((N,), device=device, dtype=torch.bool)
        self._is_heading = torch.zeros((N,), device=device, dtype=torch.bool)
        self._next_resample_time = torch.zeros((N,), device=device, dtype=torch.float32)
        self._time = torch.zeros((N,), device=device, dtype=torch.float32)

        self.resample(torch.arange(N, device=device, dtype=torch.long))

    @property
    def dt(self) -> float:
        """Return the control step in seconds (sim dt times decimation)."""
        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            return 0.02  # fallback (mjlab 50Hz)
        sim_dt = self.env.scenario.sim_params.dt or 0.005
        decimation = getattr(cfg, "decimation", 4)
        return float(sim_dt * decimation)

    def resample(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh command for the given envs."""
        if env_ids.numel() == 0:
            return
        device = env_ids.device
        n = env_ids.numel()
        r = self.cfg.ranges

        def _u(lo: float, hi: float) -> torch.Tensor:
            return torch.empty(n, device=device).uniform_(lo, hi)

        self._command[env_ids, 0] = _u(*r.lin_vel_x)
        self._command[env_ids, 1] = _u(*r.lin_vel_y)
        self._command[env_ids, 2] = _u(*r.ang_vel_z)

        if self.cfg.heading_command:
            self._heading_target[env_ids] = _u(*r.ranges.heading) if False else _u(*r.heading)

        # standing / heading flags
        standing_mask = torch.rand(n, device=device) < self.cfg.rel_standing_envs
        heading_mask = (torch.rand(n, device=device) < self.cfg.rel_heading_envs) & ~standing_mask
        self._is_standing[env_ids] = standing_mask
        self._is_heading[env_ids] = heading_mask

        # Zero command for standing envs
        if standing_mask.any():
            stand_ids = env_ids[standing_mask]
            self._command[stand_ids] = 0.0

        # Next resample time
        rrt = self.cfg.resampling_time_range
        self._next_resample_time[env_ids] = self._time[env_ids] + _u(*rrt)

    def update(self) -> None:
        """Advance time; resample envs whose interval has elapsed."""
        self._time += self.dt
        expired = self._time >= self._next_resample_time
        if expired.any():
            self.resample(torch.where(expired)[0])

        # For heading-mode envs, convert heading error → ang_vel_z command
        if self.cfg.heading_command and self._is_heading.any():
            heading = self._current_heading()
            if heading is not None:
                err = (self._heading_target - heading + math.pi) % (2 * math.pi) - math.pi
                ids = torch.where(self._is_heading)[0]
                self._command[ids, 2] = (self.cfg.heading_control_stiffness * err[ids]).clamp(
                    *self.cfg.ranges.ang_vel_z
                )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset timer + resample for env_ids (called from env._reset_idx)."""
        if env_ids.numel() == 0:
            return
        self._time[env_ids] = 0.0
        self.resample(env_ids)

    def current(self) -> torch.Tensor:
        """Return ``(num_envs, 3)`` current command."""
        return self._command

    def _current_heading(self) -> torch.Tensor | None:
        """Yaw angle of all envs (from registered robot root quat)."""
        try:
            states = self.env.handler.get_states(mode="tensor")
            if not states.robots:
                return None
            robot_name = next(iter(states.robots))
            root = states.robots[robot_name].root_state
            # wxyz quat
            w, x, y, z = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(siny_cosp, cosy_cosp)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# lifting command — mjlab LiftingCommand port
# ---------------------------------------------------------------------------


@dataclass
class ObjectPoseRange:
    """Sampling ranges for the object's reset pose (x, y, z, yaw)."""

    x: tuple[float, float] = (0.2, 0.4)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)


@dataclass
class TargetPositionRange:
    """Sampling ranges for the target lift position (x, y, z)."""

    x: tuple[float, float] = (0.1, 0.5)
    y: tuple[float, float] = (-0.3, 0.3)
    z: tuple[float, float] = (0.2, 0.4)


@dataclass
class LiftingCommandCfg:
    """Per-env target-pose sampler config (mjlab LiftingCommandCfg)."""

    entity_name: str = "cube"
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    difficulty: str = "dynamic"  # ``"dynamic"`` or ``"fixed"``
    success_threshold: float = 0.05
    object_pose_range: ObjectPoseRange | None = field(default_factory=ObjectPoseRange)
    target_position_range: TargetPositionRange = field(default_factory=TargetPositionRange)
    fixed_target: tuple[float, float, float] = (0.4, 0.0, 0.3)


class LiftingCommandManager:
    """Per-env 3-D target position for the object being lifted.

    Stored on env as ``env.command_managers[name]``. ``current()`` returns
    ``(num_envs, 3)`` target world-frame position. ``update()`` resamples
    on the configured interval; ``reset(env_ids)`` resamples on env reset
    and (when in MuJoCo path) teleports the object inside
    ``object_pose_range``.

    mjlab parity caveats:
      - Object reset uses MuJoCo ``physics.data.qpos`` writes when the
        scene is loaded via scene-MJCF (the lift_cube_yam path); Newton
        path falls back to the manager-default behavior (no teleport).
      - ``compute_success`` matches mjlab — distance < ``success_threshold``.
      - Per-env env_origins not modeled here (single-env MuJoCo case).
    """

    def __init__(self, env, cfg: LiftingCommandCfg | None = None):
        self.env = env
        self.cfg = cfg or LiftingCommandCfg()
        device = env.device
        N = env.num_envs
        self._target_pos = torch.zeros((N, 3), device=device, dtype=torch.float32)
        self._episode_success = torch.zeros((N,), device=device, dtype=torch.float32)
        self._time = torch.zeros((N,), device=device, dtype=torch.float32)
        self._next_resample_time = torch.zeros((N,), device=device, dtype=torch.float32)
        self.resample(torch.arange(N, device=device, dtype=torch.long))

    @property
    def dt(self) -> float:
        """Return the control step in seconds (sim dt times decimation)."""
        sim_dt = self.env.scenario.sim_params.dt or 0.005
        decimation = getattr(self.env.cfg, "decimation", 4)
        return float(sim_dt * decimation)

    def resample(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh target position for the given envs."""
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        device = env_ids.device
        if self.cfg.difficulty == "fixed":
            t = torch.tensor(self.cfg.fixed_target, device=device, dtype=torch.float32)
            self._target_pos[env_ids] = t.expand(n, 3)
        else:
            r = self.cfg.target_position_range
            lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=device, dtype=torch.float32)
            hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=device, dtype=torch.float32)
            self._target_pos[env_ids] = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * torch.rand(n, 3, device=device)
        self._episode_success[env_ids] = 0.0
        rrt = self.cfg.resampling_time_range
        self._next_resample_time[env_ids] = self._time[env_ids] + (torch.empty(n, device=device).uniform_(*rrt))

    def update(self) -> None:
        """Advance time, resample expired envs, and latch goal success."""
        self._time += self.dt
        expired = self._time >= self._next_resample_time
        if expired.any():
            self.resample(torch.where(expired)[0])
        # Latch episode_success on successful contact with the goal.
        obj_pos = self._object_pos_w()
        if obj_pos is not None:
            err = torch.norm(self._target_pos - obj_pos, dim=-1)
            at_goal = (err < self.cfg.success_threshold).float()
            self._episode_success = torch.maximum(self._episode_success, at_goal)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset time, success, and target for the given envs; teleport object on MuJoCo."""
        if env_ids.numel() == 0:
            return
        self._time[env_ids] = 0.0
        self._episode_success[env_ids] = 0.0
        self.resample(env_ids)
        # MuJoCo scene-MJCF path: re-place the object inside object_pose_range.
        if self.cfg.object_pose_range is not None and hasattr(self.env.handler, "physics"):
            self._teleport_object_mujoco()

    def current(self) -> torch.Tensor:
        """Return the current (num_envs, 3) target world-frame position."""
        return self._target_pos

    @property
    def target_pos(self) -> torch.Tensor:
        """Return the current (num_envs, 3) target world-frame position."""
        return self._target_pos

    @property
    def episode_success(self) -> torch.Tensor:
        """Return the per-env latched goal-success flag (0 or 1)."""
        return self._episode_success

    def _object_pos_w(self) -> torch.Tensor | None:
        """World-frame position of the tracked object."""
        if not hasattr(self.env.handler, "physics"):
            # Newton path: read object root state per-env from get_states.
            st = self.env.handler.get_states(mode="tensor")
            objs = getattr(st, "objects", None) or {}
            obj = objs.get(self.cfg.entity_name)
            return obj.root_state[:, :3] if obj is not None else None
        import mujoco

        m = self.env.handler.physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, self.cfg.entity_name)
        if bid < 0:
            return None
        import numpy as np

        pos = np.asarray(self.env.handler.physics.data.xpos[bid], dtype=np.float32)
        return torch.tensor(pos, device=self.env.device).expand(self.env.num_envs, -1)

    def _teleport_object_mujoco(self) -> None:
        import mujoco
        import numpy as np

        physics = self.env.handler.physics
        m = physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, self.cfg.entity_name)
        if bid < 0:
            return
        # Find the freejoint qpos offset.
        jid = mp.body_jntadr[bid]
        if jid < 0:
            return
        qposadr = int(mp.jnt_qposadr[jid])
        r = self.cfg.object_pose_range
        rng = np.random.default_rng()
        pos = np.array(
            [rng.uniform(*r.x), rng.uniform(*r.y), rng.uniform(*r.z)],
            dtype=np.float64,
        )
        yaw = rng.uniform(*r.yaw)
        quat_wxyz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        with physics.reset_context():
            physics.data.qpos[qposadr : qposadr + 3] = pos
            physics.data.qpos[qposadr + 3 : qposadr + 7] = quat_wxyz
            # Zero the freejoint's qvel block.
            dofadr = int(mp.jnt_dofadr[jid])
            physics.data.qvel[dofadr : dofadr + 6] = 0.0


# ---------------------------------------------------------------------------
# motion command — mjlab MotionCommand port (skeleton)
# ---------------------------------------------------------------------------


@dataclass
class MotionCommandCfg:
    """Per-env motion-tracking command config (mjlab MotionCommandCfg).

    ``motion_file``: path to a numpy npz with keys ``joint_pos``,
    ``joint_vel``, ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``,
    ``body_ang_vel_w``. Setting this to ``None`` makes the command degrade
    to an "identity-tracking" target (target = current robot state),
    which keeps the tracking_*_error_exp rewards saturated at 1.0 — useful
    for smoke-testing the wiring before motion-data ingestion is set up.
    """

    entity_name: str = "robot"
    anchor_body_name: str = "pelvis"
    body_names: tuple[str, ...] = ()
    motion_file: str | None = None
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    # Adaptive sampling (mjlab parity).
    adaptive_sampling: bool = True
    """Weight resampling distribution by recent failure rate per bin."""
    adaptive_kernel_size: int = 5
    """Number of bins to smear failure-rate signal over (mjlab default 5)."""
    adaptive_lambda: float = 0.95
    """Geometric weighting of the smoothing kernel."""
    adaptive_error_threshold: float = 0.25
    """Anchor pos error above this counts as a "failed" frame."""


class _MotionLoader:
    """Minimal port of mjlab ``MotionLoader``.

    Loads (joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
    body_ang_vel_w) tensors. ``body_indexes`` selects the rows of the
    body-* tensors that correspond to the bodies named in
    ``MotionCommandCfg.body_names`` (resolution done by the caller).
    """

    def __init__(self, motion_file: str | None, n_bodies: int, device):
        import numpy as np

        self.device = device
        self.n_bodies = n_bodies

        if motion_file is None:
            # Identity-tracking degenerate mode: zero-length placeholder.
            T = 1
            self.joint_pos = torch.zeros((T, 0), device=device)
            self.joint_vel = torch.zeros((T, 0), device=device)
            self.body_pos_w = torch.zeros((T, n_bodies, 3), device=device)
            self.body_quat_w = torch.zeros((T, n_bodies, 4), device=device)
            self.body_quat_w[:, :, 0] = 1.0
            self.body_lin_vel_w = torch.zeros((T, n_bodies, 3), device=device)
            self.body_ang_vel_w = torch.zeros((T, n_bodies, 3), device=device)
            self.time_step_total = T
            self._is_identity = True
            return

        data = np.load(motion_file)
        # body_indexes-style slicing left for caller; we keep the full
        # body-axis here and slice via [..., body_indexes] downstream.
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]
        self._is_identity = False


class MotionCommandManager:
    """Per-env motion-tracking command (mjlab MotionCommand port).

    Exposes the attribute surface required by the
    ``motion_*_error_exp`` rewards in :mod:`mdp.rewards`:

      ``anchor_pos_w``           — target anchor world position (N, 3)
      ``robot_anchor_pos_w``     — robot anchor world position (N, 3)
      ``anchor_quat_w``          — target anchor quat wxyz       (N, 4)
      ``robot_anchor_quat_w``    — robot anchor quat wxyz        (N, 4)
      ``body_pos_relative_w``    — target body positions relative to
                                   anchor                          (N, B, 3)
      ``robot_body_pos_w``       — robot body world positions     (N, B, 3)
      ``body_quat_relative_w``   — target body quats              (N, B, 4)
      ``robot_body_quat_w``      — robot body quats               (N, B, 4)
      ``body_lin_vel_w``         — target body lin vels           (N, B, 3)
      ``robot_body_lin_vel_w``   — robot body lin vels            (N, B, 3)
      ``body_ang_vel_w``         — target body ang vels           (N, B, 3)
      ``robot_body_ang_vel_w``   — robot body ang vels            (N, B, 3)
      ``cfg.body_names``         — names of tracked bodies        (tuple)

    No motion file → degrades to identity-tracking (target = current
    robot state) so the reward fns can still be exercised before motion
    data is wired in.
    """

    def __init__(self, env, cfg: MotionCommandCfg):
        self.env = env
        self.cfg = cfg
        device = env.device
        N = env.num_envs

        # Resolve body indices in the robot's body name list.
        self._mujoco_body_ids: list[int] = []
        if cfg.body_names:
            if hasattr(env.handler, "physics"):
                import mujoco

                model = env.handler.physics.model
                mp = model.ptr if hasattr(model, "ptr") else model
                for name in cfg.body_names:
                    bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, name)
                    self._mujoco_body_ids.append(int(bid))
                # Anchor body index inside cfg.body_names.
                if cfg.anchor_body_name in cfg.body_names:
                    self._anchor_idx = cfg.body_names.index(cfg.anchor_body_name)
                else:
                    self._anchor_idx = 0
                # Resolve anchor mujoco body id separately (may not be in body_names).
                self._anchor_bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, cfg.anchor_body_name)
            else:
                self._anchor_idx = 0
                self._anchor_bid = -1

        self.motion = _MotionLoader(cfg.motion_file, len(cfg.body_names), device)
        self.time_steps = torch.zeros(N, dtype=torch.long, device=device)
        self._time_acc = torch.zeros(N, device=device)
        self._next_resample = torch.zeros(N, device=device)

        # Adaptive bin sampling state (mjlab parity).
        self._bin_count = max(1, int(self.motion.time_step_total // max(1, int(1.0 / env.step_dt))) + 1)
        self._bin_failed_count = torch.zeros(self._bin_count, device=device)
        self._current_bin_failed = torch.zeros(self._bin_count, device=device)
        # Smoothing kernel (geometric).
        k = max(1, cfg.adaptive_kernel_size)
        kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(k)],
            device=device,
        )
        self._kernel = kernel / kernel.sum()

        self.resample(torch.arange(N, device=device, dtype=torch.long))

    @property
    def dt(self) -> float:
        """Return the control step in seconds (sim dt times decimation)."""
        sim_dt = self.env.scenario.sim_params.dt or 0.005
        decimation = getattr(self.env.cfg, "decimation", 4)
        return float(sim_dt * decimation)

    def resample(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh motion start frame for the given envs."""
        if env_ids.numel() == 0:
            return
        if self.motion._is_identity:
            self.time_steps[env_ids] = 0
        else:
            new_steps = self._sample_starting_steps(env_ids.numel(), env_ids.device)
            self.time_steps[env_ids] = new_steps
        n = env_ids.numel()
        rrt = self.cfg.resampling_time_range
        self._next_resample[env_ids] = self._time_acc[env_ids] + (torch.empty(n, device=env_ids.device).uniform_(*rrt))

    def _sample_starting_steps(self, n: int, device) -> torch.Tensor:
        """Sample initial motion frame for ``n`` envs.

        Mjlab's adaptive sampler: bins are smoothed by a geometric kernel
        and weighted toward bins with higher recent failure rate. Without
        adaptive sampling (or when failed_count is zero), falls back to
        uniform.
        """
        T = self.motion.time_step_total
        if not self.cfg.adaptive_sampling or self._bin_count <= 1:
            return torch.randint(0, T, (n,), device=device, dtype=torch.long)

        weights = self._bin_failed_count.clone()
        if weights.sum() <= 0:
            return torch.randint(0, T, (n,), device=device, dtype=torch.long)
        # Convolve with kernel (1-D right-side smear, mjlab style).
        k = self._kernel
        if k.numel() > 1:
            smoothed = torch.zeros_like(weights)
            K = k.numel()
            for i in range(self._bin_count):
                acc = 0.0
                for j in range(K):
                    src = i - j
                    if 0 <= src < self._bin_count:
                        acc = acc + float(weights[src]) * float(k[j])
                smoothed[i] = acc
            weights = smoothed
        # Add a small uniform floor so unfailed bins still get some probability.
        weights = weights + weights.max().clamp(min=1e-6) * 0.05
        # Sample bins.
        idx = torch.multinomial(weights, n, replacement=True).to(device)
        # Map bin → frame: frames per bin = T / bin_count.
        per_bin = max(1, T // self._bin_count)
        starts = idx * per_bin
        # Jitter inside the bin.
        jitter = torch.randint(0, per_bin, (n,), device=device)
        return (starts + jitter).clamp(max=T - 1).long()

    def update(self) -> None:
        """Advance motion time, step the frame, and resample expired envs."""
        self._time_acc += self.dt
        if not self.motion._is_identity:
            self.time_steps = (self.time_steps + 1) % self.motion.time_step_total
            # Update failure stats per-bin (adaptive sampling feedback).
            if self.cfg.adaptive_sampling:
                self._update_failure_stats()
        expired = self._time_acc >= self._next_resample
        if expired.any():
            self.resample(torch.where(expired)[0])

    def _update_failure_stats(self) -> None:
        """Track failure rate per motion bin.

        A frame is "failed" when anchor position error > threshold.
        ``_bin_failed_count`` aggregates this so future resamples are
        biased toward bins where the policy struggles.
        """
        T = self.motion.time_step_total
        per_bin = max(1, T // self._bin_count)
        try:
            err = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        except Exception:
            return
        failed = (err > self.cfg.adaptive_error_threshold).float()
        # Increment bin counts for each env's current time_step.
        bin_idx = (self.time_steps // per_bin).clamp(max=self._bin_count - 1)
        for i in range(self.env.num_envs):
            self._bin_failed_count[bin_idx[i]] = 0.99 * self._bin_failed_count[bin_idx[i]] + 0.01 * failed[i]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset motion time and resample start frames for the given envs."""
        if env_ids.numel() == 0:
            return
        self._time_acc[env_ids] = 0.0
        self.resample(env_ids)

    def current(self) -> torch.Tensor:
        """Return ``(N, J*2)`` concatenated joint_pos+joint_vel reference at the current frame.

        Empty if identity-tracking mode.
        """
        return torch.cat(
            [self.motion.joint_pos[self.time_steps], self.motion.joint_vel[self.time_steps]],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # robot-side world-frame body state (read from MuJoCo physics.data).
    # ------------------------------------------------------------------

    def _robot_body_state_w(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-body (pos, quat, lin_vel, ang_vel) for all tracked robot bodies.

        MuJoCo path; falls back to identity for Newton (caller will see
        all-zeros and rewards collapse to 1.0).
        """
        device = self.env.device
        N = self.env.num_envs
        B = len(self.cfg.body_names)

        if not hasattr(self.env.handler, "physics") or B == 0:
            pos = torch.zeros((N, B, 3), device=device)
            quat = torch.zeros((N, B, 4), device=device)
            quat[:, :, 0] = 1.0
            lin = torch.zeros((N, B, 3), device=device)
            ang = torch.zeros((N, B, 3), device=device)
            return pos, quat, lin, ang

        import mujoco
        import numpy as np

        physics = self.env.handler.physics
        data = physics.data
        model = physics.model
        mp = model.ptr if hasattr(model, "ptr") else model

        bids = self._mujoco_body_ids
        pos = np.asarray(data.xpos[bids], dtype=np.float32)  # (B, 3)
        quat = np.asarray(data.xquat[bids], dtype=np.float32)  # (B, 4) wxyz
        # Linear/angular velocity in world frame via mj_objectVelocity.
        lin_arr = np.zeros((B, 6), dtype=np.float64)
        for i, bid in enumerate(bids):
            v = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(mp, data.ptr, mujoco.mjtObj.mjOBJ_BODY, int(bid), v, 0)
            lin_arr[i] = v
        # mj_objectVelocity returns [ang(3), lin(3)] when flg_local=0.
        ang = torch.tensor(lin_arr[:, :3], dtype=torch.float32, device=device)
        lin = torch.tensor(lin_arr[:, 3:], dtype=torch.float32, device=device)
        pos_t = torch.tensor(pos, device=device)
        quat_t = torch.tensor(quat, device=device)
        # Broadcast to (N, B, *).
        pos_t = pos_t.unsqueeze(0).expand(N, -1, -1)
        quat_t = quat_t.unsqueeze(0).expand(N, -1, -1)
        lin = lin.unsqueeze(0).expand(N, -1, -1)
        ang = ang.unsqueeze(0).expand(N, -1, -1)
        return pos_t, quat_t, lin, ang

    # ------------------------------------------------------------------
    # properties consumed by motion_*_error_exp rewards.
    # ------------------------------------------------------------------

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Return per-body world-frame positions on the robot side."""
        return self._robot_body_state_w()[0]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Return per-body world-frame orientations on the robot side."""
        return self._robot_body_state_w()[1]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Return per-body world-frame linear velocities on the robot side."""
        return self._robot_body_state_w()[2]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Return per-body world-frame angular velocities on the robot side."""
        return self._robot_body_state_w()[3]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Return the robot anchor body's world-frame position."""
        return self.robot_body_pos_w[:, self._anchor_idx]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Return the robot anchor body's world-frame orientation."""
        return self.robot_body_quat_w[:, self._anchor_idx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Target body positions at current motion frame, broadcast over envs."""
        if self.motion._is_identity:
            # Mirror robot state so anchor errors are zero.
            return self.robot_body_pos_w
        # (T, B, 3)[time_steps] → (N, B, 3)
        return self.motion.body_pos_w[self.time_steps]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return target body orientations at the current motion frame."""
        if self.motion._is_identity:
            return self.robot_body_quat_w
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Return target body linear velocities at the current motion frame."""
        if self.motion._is_identity:
            return self.robot_body_lin_vel_w
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Return target body angular velocities at the current motion frame."""
        if self.motion._is_identity:
            return self.robot_body_ang_vel_w
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Return the target anchor body's world-frame position."""
        return self.body_pos_w[:, self._anchor_idx]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Return the target anchor body's world-frame orientation."""
        return self.body_quat_w[:, self._anchor_idx]

    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        """Return target body positions aligned to the robot anchor.

        Mjlab parity: the target anchor is aligned to the robot anchor so
        the motion is replayed at the current robot location.
        """
        target = self.body_pos_w
        target_anchor = target[:, self._anchor_idx : self._anchor_idx + 1]
        robot_anchor = self.robot_anchor_pos_w.unsqueeze(1)
        return target - target_anchor + robot_anchor

    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        """Return target body orientations relative to the robot anchor."""
        # Approximation: hand back the target quaternions unchanged. mjlab
        # rotates each body's quat by the anchor delta; for parity we'd
        # need quat composition utilities. Defer this refinement until a
        # motion file exists to fit the data against.
        return self.body_quat_w
