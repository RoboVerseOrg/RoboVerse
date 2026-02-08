from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from roboverse_pack.protocol_sim.core.interfaces import ExternalAssist
from roboverse_pack.protocol_sim.core.types import SimRobotObservation


@dataclass
class ElasticBandConfig:
    """Configuration for the elastic band assist."""

    stiffness: float = 200.0
    damping: float = 100.0
    point: tuple[float, float, float] = (0.0, 0.0, 2.0)
    length: float = 0.0
    body_name: str = "torso_link"
    fallback_body_name: str = "base_link"
    release_time_s: float = 1.0

    def __post_init__(self) -> None:
        # Keep configuration numerically stable and intuitive.
        self.length = max(0.0, float(self.length))
        self.release_time_s = max(0.0, float(self.release_time_s))
        self.point = (float(self.point[0]), float(self.point[1]), float(self.point[2]))


class ElasticBandAssist(ExternalAssist):
    """Spring-damper safety harness attached to the base/torso."""

    def __init__(self, *, handler: Any, robot_name: str, cfg: ElasticBandConfig):
        self._handler = handler
        self._robot_name = str(robot_name)
        self._cfg = cfg
        self._cfg_lock = threading.Lock()

        self._releasing = False
        self._release_elapsed = 0.0
        self._scale = 1.0

        self._apply_force: Callable[[np.ndarray], None] = self._make_applier(handler, robot_name, cfg)

    def start_release(self) -> None:
        """Release the elastic band by ramping the scale down to 0."""
        with self._cfg_lock:
            release_time_s = float(self._cfg.release_time_s)
        if release_time_s <= 0.0:
            self._scale = 0.0
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return
        self._releasing = True
        self._release_elapsed = 0.0

    def set_length(self, length_m: float) -> float:
        """Set the spring rest length in meters (clamped to >= 0)."""
        with self._cfg_lock:
            self._cfg.length = max(0.0, float(length_m))
            return float(self._cfg.length)

    def set_anchor_height(self, height_m: float) -> None:
        """Set the world-frame z height of the anchor point."""
        with self._cfg_lock:
            x, y, _ = self._cfg.point
            self._cfg.point = (float(x), float(y), float(height_m))

    def get_length(self) -> float:
        """Get the current spring rest length in meters."""
        with self._cfg_lock:
            return float(self._cfg.length)

    def get_anchor_height(self) -> float:
        """Get the current world-frame anchor height (z) in meters."""
        with self._cfg_lock:
            return float(self._cfg.point[2])

    def apply(self, obs: SimRobotObservation, *, dt: float) -> None:
        """Apply the elastic band force to the robot based on the current observation."""
        if self._scale <= 0.0:
            return

        with self._cfg_lock:
            release_time_s = float(self._cfg.release_time_s)
            point = np.asarray(self._cfg.point, dtype=np.float32)
            stiffness = float(self._cfg.stiffness)
            damping = float(self._cfg.damping)
            length = float(self._cfg.length)

        if self._releasing:
            self._release_elapsed += float(dt)
            self._scale = float(max(0.0, 1.0 - self._release_elapsed / max(1e-12, release_time_s)))
            if self._scale <= 0.0:
                self._apply_force(np.zeros((3,), dtype=np.float32))
                return

        x = obs.root_state[0:3].astype(np.float32, copy=False)
        dx = obs.root_state[7:10].astype(np.float32, copy=False)

        delta = point - x
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return

        # Tension-only model: if the band is slack, apply no force.
        stretch = dist - length
        if stretch <= 0.0:
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return

        direction = delta / dist
        v = float(np.dot(dx, direction))
        # Damping only modulates pulling; never allow compression/pushing.
        f_mag = max(0.0, stiffness * stretch - damping * v)
        if f_mag <= 0.0:
            self._apply_force(np.zeros((3,), dtype=np.float32))
            return
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

        # IsaacSim / IsaacLab backend: set_external_force_and_torque on articulation bodies.
        if hasattr(handler, "scene") and hasattr(handler.scene, "articulations"):
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise ImportError("ElasticBandAssist for IsaacSim requires torch.") from exc

            robot_inst = handler.scene.articulations.get(robot_name)
            if robot_inst is not None and hasattr(robot_inst, "set_external_force_and_torque"):
                body_names = list(getattr(robot_inst, "body_names", []) or [])
                if not body_names:
                    raise ValueError(f"No articulation body names found for robot '{robot_name}'.")

                def _resolve(body: str) -> int:
                    if body in body_names:
                        return int(body_names.index(body))
                    suffix = f"/{body}"
                    for i, name in enumerate(body_names):
                        if str(name).endswith(suffix):
                            return int(i)
                    raise KeyError(f"Body '{body}' not found for elastic band.")

                try:
                    body_idx = _resolve(cfg.body_name)
                except Exception:
                    body_idx = _resolve(cfg.fallback_body_name)

                device = getattr(handler, "device", torch.device("cpu"))
                num_envs = int(getattr(handler, "num_envs", 1) or 1)
                n_bodies = len(body_names)
                forces = torch.zeros((num_envs, n_bodies, 3), device=device, dtype=torch.float32)
                torques = torch.zeros((num_envs, n_bodies, 3), device=device, dtype=torch.float32)

                def _apply(force: np.ndarray) -> None:
                    forces.zero_()
                    torques.zero_()
                    f = torch.as_tensor(force, device=device, dtype=torch.float32).reshape(3)
                    forces[:, int(body_idx), :] = f
                    # Signature may vary slightly across IsaacLab/IsaacSim versions.
                    try:
                        robot_inst.set_external_force_and_torque(
                            forces=forces,
                            torques=torques,
                            body_ids=None,
                            env_ids=None,
                            is_global=True,
                        )
                    except TypeError:
                        try:
                            robot_inst.set_external_force_and_torque(forces=forces, torques=torques, is_global=True)
                        except TypeError:
                            robot_inst.set_external_force_and_torque(forces, torques)

                return _apply

        # Newton backend: inject world-frame body force via NewtonHandler helper.
        if (
            hasattr(handler, "set_external_body_force")
            and hasattr(handler, "_get_body_indices")
            and hasattr(handler, "_model")
        ):
            body_ids = list(handler._get_body_indices(0, robot_name))
            if not body_ids:
                raise ValueError(f"No Newton body indices found for robot '{robot_name}'.")

            def _resolve(body: str) -> int:
                suffix = f"/{body}"
                for bid in body_ids:
                    name = str(handler._model.body_key[bid])
                    if name == body or name.endswith(suffix):
                        return int(bid)
                raise KeyError(f"Body '{body}' not found for elastic band.")

            try:
                body_id = _resolve(cfg.body_name)
            except Exception:
                body_id = _resolve(cfg.fallback_body_name)

            def _apply(force: np.ndarray) -> None:
                handler.set_external_body_force(body_id, np.asarray(force, dtype=np.float32))

            return _apply

        raise NotImplementedError(
            "ElasticBandAssist is implemented for MuJoCo, IsaacGym, IsaacSim, and Newton handlers."
        )
