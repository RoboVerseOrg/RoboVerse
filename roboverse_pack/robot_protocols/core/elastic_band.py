from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from roboverse_pack.robot_protocols.core.interfaces import ExternalAssist
from roboverse_pack.robot_protocols.core.types import SimRobotObservation


@dataclass
class ElasticBandConfig:
    """Configuration for the elastic band assist."""

    stiffness: float = 200.0
    damping: float = 100.0
    point: tuple[float, float, float] = (0.0, 0.0, 3.0)
    length: float = 0.0
    body_name: str = "torso_link"
    fallback_body_name: str = "base_link"
    release_time_s: float = 1.0


class ElasticBandAssist(ExternalAssist):
    """MuJoCo-style spring strap attached to the base/torso (optional safety harness)."""

    def __init__(self, *, handler: Any, robot_name: str, cfg: ElasticBandConfig):
        self._handler = handler
        self._robot_name = str(robot_name)
        self._cfg = cfg

        self._releasing = False
        self._release_elapsed = 0.0
        self._scale = 1.0

        self._apply_force: Callable[[np.ndarray], None] = self._make_applier(handler, robot_name, cfg)

    def start_release(self) -> None:
        """Release the elastic band by ramping the scale down to 0."""
        if self._cfg.release_time_s <= 0.0:
            self._scale = 0.0
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return
        self._releasing = True
        self._release_elapsed = 0.0

    def apply(self, obs: SimRobotObservation, *, dt: float) -> None:
        """Apply the elastic band force to the robot based on the current observation."""
        if self._scale <= 0.0:
            return

        if self._releasing:
            self._release_elapsed += float(dt)
            self._scale = float(max(0.0, 1.0 - self._release_elapsed / float(self._cfg.release_time_s)))
            if self._scale <= 0.0:
                self._apply_force(np.zeros((3,), dtype=np.float32))
                return

        x = obs.root_state[0:3].astype(np.float32, copy=False)
        dx = obs.root_state[7:10].astype(np.float32, copy=False)

        point = np.asarray(self._cfg.point, dtype=np.float32)
        delta = point - x
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return

        direction = delta / dist
        v = float(np.dot(dx, direction))
        f_mag = float(self._cfg.stiffness) * (dist - float(self._cfg.length)) - float(self._cfg.damping) * v
        force = (f_mag * direction).astype(np.float32, copy=False) * float(self._scale)
        self._apply_force(force)

    @staticmethod
    def _make_applier(handler: Any, robot_name: str, cfg: ElasticBandConfig) -> Callable[[np.ndarray], None]:
        # MuJoCo backend: write to xfrc_applied[body_id, :3].
        if (
            hasattr(handler, "physics")
            and hasattr(handler.physics, "data")
            and hasattr(handler.physics.data, "xfrc_applied")
        ):
            model_name = None
            try:
                model_name = handler.mj_objects[robot_name].model
            except Exception:
                model_name = None

            def _resolve(body: str) -> int:
                if model_name is not None:
                    try:
                        return handler.physics.model.body(f"{model_name}/{body}").id
                    except Exception:
                        pass
                suffix = f"/{body}"
                for bi in range(handler.physics.model.nbody):
                    name = handler.physics.model.body(bi).name
                    if name.endswith(suffix):
                        return handler.physics.model.body(bi).id
                raise KeyError(f"Body '{body}' not found for elastic band.")

            try:
                body_id = _resolve(cfg.body_name)
            except Exception:
                body_id = _resolve(cfg.fallback_body_name)

            def _apply(force: np.ndarray) -> None:
                handler.physics.data.xfrc_applied[body_id, 0:3] = force
                handler.physics.data.xfrc_applied[body_id, 3:6] = 0.0

            return _apply

        # IsaacGym backend: apply_rigid_body_force_tensors over global rigid body indices.
        if hasattr(handler, "gym") and hasattr(handler, "sim") and hasattr(handler, "_rigid_body_states"):
            try:
                import torch
                from isaacgym import gymapi, gymtorch
            except Exception as exc:  # pragma: no cover
                raise ImportError("ElasticBandAssist for IsaacGym requires isaacgym + torch.") from exc

            body_idx = None
            try:
                body_idx = handler._env_rigid_body_global_indices[0]["robot"][cfg.body_name]
            except Exception:
                try:
                    body_idx = handler._env_rigid_body_global_indices[0]["robot"][cfg.fallback_body_name]
                except Exception:
                    body_idx = None
            if body_idx is None:
                try:
                    body_idx = handler._body_info[robot_name]["global_indices"][cfg.body_name]
                except Exception:
                    body_idx = handler._body_info[robot_name]["global_indices"][cfg.fallback_body_name]

            device = handler._rigid_body_states.device
            n_bodies = int(handler._rigid_body_states.shape[0])
            forces = torch.zeros((n_bodies, 3), device=device, dtype=torch.float32)
            torques = torch.zeros((n_bodies, 3), device=device, dtype=torch.float32)

            def _apply(force: np.ndarray) -> None:
                forces.zero_()
                torques.zero_()
                forces[int(body_idx)] = torch.as_tensor(force, device=device, dtype=torch.float32)
                handler.gym.apply_rigid_body_force_tensors(
                    handler.sim,
                    gymtorch.unwrap_tensor(forces),
                    gymtorch.unwrap_tensor(torques),
                    gymapi.ENV_SPACE,
                )

            return _apply

        raise NotImplementedError("ElasticBandAssist is only implemented for MuJoCo and IsaacGym handlers.")
