from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from loguru import logger as log

from metasim.scenario.robot import RobotCfg
from metasim.types import RobotState, TensorState
from metasim.utils.math import quat_apply_inverse

from .utils import resolve_matching_names, resolve_pattern_values


@dataclass
class CompatArticulationData:
    """MetaSim-backed subset of IsaacLab's `ArticulationData` API."""

    root_state_w: torch.Tensor  # (num_envs, 13)
    body_state_w: torch.Tensor | None  # (num_envs, num_bodies, 13)
    joint_pos: torch.Tensor  # (num_envs, num_joints)
    joint_vel: torch.Tensor  # (num_envs, num_joints)
    joint_pos_target: torch.Tensor | None
    joint_vel_target: torch.Tensor | None
    joint_effort_target: torch.Tensor | None

    default_joint_pos: torch.Tensor  # (num_envs, num_joints)
    default_joint_vel: torch.Tensor  # (num_envs, num_joints)
    soft_joint_pos_limits: torch.Tensor  # (num_envs, num_joints, 2)

    GRAVITY_VEC_W: torch.Tensor  # (num_envs, 3)

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.root_state_w[:, 0:3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.root_state_w[:, 3:7]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.root_state_w[:, 10:13]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        # IsaacLab convention: base-linear velocity expressed in the body frame.
        return quat_apply_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        # IsaacLab convention: base-angular velocity expressed in the body frame.
        return quat_apply_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self.body_state_w is None:
            raise RuntimeError("body_state_w is not available for this articulation/backend.")
        return self.body_state_w[..., 0:3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self.body_state_w is None:
            raise RuntimeError("body_state_w is not available for this articulation/backend.")
        return self.body_state_w[..., 3:7]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self.body_state_w is None:
            raise RuntimeError("body_state_w is not available for this articulation/backend.")
        return self.body_state_w[..., 7:10]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self.body_state_w is None:
            raise RuntimeError("body_state_w is not available for this articulation/backend.")
        return self.body_state_w[..., 10:13]


def _normalize_env_ids(env_ids: Sequence[int] | torch.Tensor | None, *, num_envs: int) -> list[int]:
    if env_ids is None:
        return list(range(num_envs))
    if isinstance(env_ids, torch.Tensor):
        return env_ids.detach().cpu().to(dtype=torch.long).tolist()
    return list(env_ids)


class CompatArticulation:
    """Compatibility wrapper that presents a MetaSim robot as an IsaacLab `Articulation`."""

    def __init__(
        self,
        *,
        handler: Any,
        robot_cfg: RobotCfg,
        sim_name: str,
        env_origins: torch.Tensor,
    ) -> None:
        self._handler = handler
        self._robot_cfg = robot_cfg
        self._sim_name = sim_name
        self._env_origins = env_origins

        # Name lists (sorted for cross-backend determinism)
        self.joint_names: list[str] = handler.get_joint_names(sim_name, sort=True)
        self.body_names: list[str] = handler.get_body_names(sim_name, sort=True)

        self._default_joint_pos = self._build_default_joint_pos()
        self._default_joint_vel = self._build_default_joint_vel()
        self._soft_joint_pos_limits = self._build_soft_joint_pos_limits()

        self._data: CompatArticulationData | None = None
        self._root_state_local: torch.Tensor | None = None  # (num_envs, 13) in handler-local frame
        self._last_full_state: TensorState | None = None
        self.is_initialized: bool = False

    # ---------------------------------------------------------------------
    # IsaacLab-like convenience properties
    # ---------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        if self._data is not None:
            return self._data.joint_pos.device
        try:
            return self._handler.device
        except Exception:  # pragma: no cover
            return torch.device("cpu")

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    # ---------------------------------------------------------------------
    # Data refresh
    # ---------------------------------------------------------------------
    def update_from_states(self, *, full_state: TensorState, env_origins: torch.Tensor) -> None:
        """Refresh internal buffers from the latest handler `TensorState`."""
        if self._sim_name not in full_state.robots:
            raise KeyError(f"Robot '{self._sim_name}' not found in handler state.")

        robot_state: RobotState = full_state.robots[self._sim_name]
        root_state_local = robot_state.root_state
        root_state_w = root_state_local.clone()
        root_state_w[:, 0:3] = root_state_w[:, 0:3] + env_origins

        body_state_w = None
        if robot_state.body_state is not None:
            body_state_w = robot_state.body_state.clone()
            body_state_w[:, :, 0:3] = body_state_w[:, :, 0:3] + env_origins[:, None, :]

        num_envs = root_state_local.shape[0]
        gravity = torch.tensor([0.0, 0.0, -1.0], device=root_state_local.device, dtype=root_state_local.dtype).repeat(
            num_envs, 1
        )

        # Defaults/limits are part of IsaacLab's `ArticulationData` surface and may be
        # mutated by events (e.g., randomize default joint positions). Therefore, do
        # not overwrite them on every refresh once initialized.
        if self._data is None:
            default_joint_pos = self._default_joint_pos.to(device=root_state_local.device).repeat(num_envs, 1)
            default_joint_vel = self._default_joint_vel.to(device=root_state_local.device).repeat(num_envs, 1)
            soft_limits = self._soft_joint_pos_limits.to(device=root_state_local.device).repeat(num_envs, 1, 1)
        else:
            default_joint_pos = self._data.default_joint_pos
            default_joint_vel = self._data.default_joint_vel
            soft_limits = self._data.soft_joint_pos_limits

            # If device/shape changes (rare), rebuild from templates.
            if default_joint_pos.device != root_state_local.device or default_joint_pos.shape[0] != num_envs:
                default_joint_pos = self._default_joint_pos.to(device=root_state_local.device).repeat(num_envs, 1)
            if default_joint_vel.device != root_state_local.device or default_joint_vel.shape[0] != num_envs:
                default_joint_vel = self._default_joint_vel.to(device=root_state_local.device).repeat(num_envs, 1)
            if soft_limits.device != root_state_local.device or soft_limits.shape[0] != num_envs:
                soft_limits = self._soft_joint_pos_limits.to(device=root_state_local.device).repeat(num_envs, 1, 1)

        self._data = CompatArticulationData(
            root_state_w=root_state_w,
            body_state_w=body_state_w,
            joint_pos=robot_state.joint_pos,
            joint_vel=robot_state.joint_vel,
            joint_pos_target=robot_state.joint_pos_target,
            joint_vel_target=robot_state.joint_vel_target,
            joint_effort_target=robot_state.joint_effort_target,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            soft_joint_pos_limits=soft_limits,
            GRAVITY_VEC_W=gravity,
        )

        self._root_state_local = root_state_local
        self._last_full_state = full_state
        self._env_origins = env_origins
        self.is_initialized = True

    @property
    def data(self) -> CompatArticulationData:
        if self._data is None:
            raise RuntimeError("CompatArticulation is not initialized yet. Call update_from_states() first.")
        return self._data

    # ---------------------------------------------------------------------
    # Name resolution helpers (SceneEntityCfg selectors)
    # ---------------------------------------------------------------------
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, candidates=self.body_names, preserve_order=preserve_order)

    def find_joints(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, candidates=self.joint_names, preserve_order=preserve_order)

    # ---------------------------------------------------------------------
    # IsaacLab-compatible writer methods (used by some term classes)
    # ---------------------------------------------------------------------
    def write_joint_state_to_sim(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, *, env_ids: Sequence[int] | None = None
    ):
        env_ids_list = _normalize_env_ids(env_ids, num_envs=self._handler.num_envs)
        if not env_ids_list:
            return

        if self._last_full_state is not None and not hasattr(self._handler, "remotes"):
            try:
                base = self._last_full_state
                robot_state = base.robots[self._sim_name]
                env_ids_t = torch.tensor(env_ids_list, device=robot_state.joint_pos.device, dtype=torch.long)
                new_joint_pos = robot_state.joint_pos.clone()
                new_joint_vel = robot_state.joint_vel.clone()
                new_joint_pos[env_ids_t] = joint_pos.to(new_joint_pos.device)
                new_joint_vel[env_ids_t] = joint_vel.to(new_joint_vel.device)

                new_robot_state = RobotState(
                    root_state=robot_state.root_state,
                    body_names=robot_state.body_names,
                    body_state=robot_state.body_state,
                    joint_pos=new_joint_pos,
                    joint_vel=new_joint_vel,
                    joint_pos_target=robot_state.joint_pos_target,
                    joint_vel_target=robot_state.joint_vel_target,
                    joint_effort_target=robot_state.joint_effort_target,
                )
                robots = dict(base.robots)
                robots[self._sim_name] = new_robot_state
                patched = TensorState(objects=base.objects, robots=robots, cameras=base.cameras, extras=base.extras)
                self._handler.set_states(patched, env_ids=env_ids_list)
                return
            except Exception as exc:
                log.debug("TensorState fast-path joint write failed (%s). Falling back to dict states.", exc)

        # Fallback: dict-based state updates (works in parallel mode)
        if self._root_state_local is None:
            raise RuntimeError("Cannot write joint state before the articulation is initialized.")

        joint_pos_cpu = joint_pos.detach().cpu()
        joint_vel_cpu = joint_vel.detach().cpu()

        states = []
        for local_i, env_id in enumerate(env_ids_list):
            root = self._root_state_local[env_id]
            dof_pos = {jn: float(joint_pos_cpu[local_i, j].item()) for j, jn in enumerate(self.joint_names)}
            dof_vel = {jn: float(joint_vel_cpu[local_i, j].item()) for j, jn in enumerate(self.joint_names)}
            states.append({
                "objects": {},
                "robots": {
                    self._sim_name: {
                        "pos": root[0:3].detach().cpu(),
                        "rot": root[3:7].detach().cpu(),
                        "vel": root[7:10].detach().cpu(),
                        "ang_vel": root[10:13].detach().cpu(),
                        "dof_pos": dof_pos,
                        "dof_vel": dof_vel,
                    }
                },
                "cameras": {},
                "extras": {},
            })

        self._handler.set_states(states, env_ids=env_ids_list)

    def write_root_state_to_sim(self, root_state_w: torch.Tensor, *, env_ids: Sequence[int] | None = None):
        env_ids_list = _normalize_env_ids(env_ids, num_envs=self._handler.num_envs)
        if not env_ids_list:
            return

        # Convert to handler-local coordinates (MetaSim handlers for vectorized IsaacSim
        # subtract env origins in their public `get_states()` API).
        root_state_local = root_state_w.clone()
        env_ids_t = torch.tensor(env_ids_list, device=root_state_local.device, dtype=torch.long)
        root_state_local[:, 0:3] = root_state_local[:, 0:3] - self._env_origins[env_ids_t]

        if self._last_full_state is not None and not hasattr(self._handler, "remotes"):
            try:
                base = self._last_full_state
                robot_state = base.robots[self._sim_name]
                new_root_state = robot_state.root_state.clone()
                new_root_state[env_ids_t] = root_state_local.to(new_root_state.device)

                new_robot_state = RobotState(
                    root_state=new_root_state,
                    body_names=robot_state.body_names,
                    body_state=robot_state.body_state,
                    joint_pos=robot_state.joint_pos,
                    joint_vel=robot_state.joint_vel,
                    joint_pos_target=robot_state.joint_pos_target,
                    joint_vel_target=robot_state.joint_vel_target,
                    joint_effort_target=robot_state.joint_effort_target,
                )
                robots = dict(base.robots)
                robots[self._sim_name] = new_robot_state
                patched = TensorState(objects=base.objects, robots=robots, cameras=base.cameras, extras=base.extras)
                self._handler.set_states(patched, env_ids=env_ids_list)
                return
            except Exception as exc:
                log.debug("TensorState fast-path root write failed (%s). Falling back to dict states.", exc)

        if self._root_state_local is None:
            raise RuntimeError("Cannot write root state before the articulation is initialized.")

        root_state_local_cpu = root_state_local.detach().cpu()
        states = []
        for local_i, env_id in enumerate(env_ids_list):
            root = root_state_local_cpu[local_i]
            states.append({
                "objects": {},
                "robots": {
                    self._sim_name: {
                        "pos": root[0:3],
                        "rot": root[3:7],
                        "vel": root[7:10],
                        "ang_vel": root[10:13],
                    }
                },
                "cameras": {},
                "extras": {},
            })
        self._handler.set_states(states, env_ids=env_ids_list)

    def set_joint_position_target(self, position: torch.Tensor, *, env_ids: Sequence[int] | None = None):
        """IsaacLab-compatible API: cache joint position targets.

        MetaSim handlers apply targets through `handler.set_dof_targets(...)`.
        This method is provided for term compatibility only.
        """
        env_ids_list = _normalize_env_ids(env_ids, num_envs=self._handler.num_envs)
        if self._data is None:
            return
        if self._data.joint_pos_target is None:
            self._data.joint_pos_target = torch.zeros_like(self._data.joint_pos)
        env_ids_t = torch.tensor(env_ids_list, device=self._data.joint_pos_target.device, dtype=torch.long)
        self._data.joint_pos_target[env_ids_t] = position.to(self._data.joint_pos_target.device)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_default_joint_pos(self) -> torch.Tensor:
        values = resolve_pattern_values(
            getattr(self._robot_cfg, "default_joint_positions", None), self.joint_names, default=0.0
        )
        vec = torch.tensor([float(values[jn]) for jn in self.joint_names], dtype=torch.float32)
        return vec.unsqueeze(0)

    def _build_default_joint_vel(self) -> torch.Tensor:
        values = resolve_pattern_values(
            getattr(self._robot_cfg, "default_joint_velocities", None), self.joint_names, default=0.0
        )
        vec = torch.tensor([float(values[jn]) for jn in self.joint_names], dtype=torch.float32)
        return vec.unsqueeze(0)

    def _build_soft_joint_pos_limits(self) -> torch.Tensor:
        limits = resolve_pattern_values(
            getattr(self._robot_cfg, "joint_limits", None), self.joint_names, default=(-3.14, 3.14)
        )
        low = torch.tensor([float(limits[jn][0]) for jn in self.joint_names], dtype=torch.float32)
        high = torch.tensor([float(limits[jn][1]) for jn in self.joint_names], dtype=torch.float32)
        dof_pos_limits = torch.stack([low, high], dim=-1)  # (n_joints, 2)

        factor = float(getattr(self._robot_cfg, "soft_joint_pos_limit_factor", 0.9))
        mid = (dof_pos_limits[:, 0] + dof_pos_limits[:, 1]) / 2.0
        diff = dof_pos_limits[:, 1] - dof_pos_limits[:, 0]

        soft = torch.zeros_like(dof_pos_limits)
        soft[:, 0] = mid - 0.5 * diff * factor
        soft[:, 1] = mid + 0.5 * diff * factor
        return soft.unsqueeze(0)  # (1, n_joints, 2)


class CompatRigidObject:
    """Minimal rigid-object facade (root pose/vel only)."""

    def __init__(self, *, handler: Any, sim_name: str, env_origins: torch.Tensor) -> None:
        self._handler = handler
        self._sim_name = sim_name
        self._env_origins = env_origins
        self._last_full_state: TensorState | None = None
        self.is_initialized: bool = False

    def update_from_states(self, *, full_state: TensorState, env_origins: torch.Tensor) -> None:
        if self._sim_name in full_state.objects:
            root_state_local = full_state.objects[self._sim_name].root_state
        elif self._sim_name in full_state.robots:
            root_state_local = full_state.robots[self._sim_name].root_state
        else:
            raise KeyError(f"Object '{self._sim_name}' not found in handler state.")

        root_state_w = root_state_local.clone()
        root_state_w[:, 0:3] = root_state_w[:, 0:3] + env_origins
        self.root_state_w = root_state_w
        self._env_origins = env_origins
        self._last_full_state = full_state
        self.is_initialized = True
