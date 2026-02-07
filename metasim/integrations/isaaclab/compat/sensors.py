from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .utils import resolve_matching_names


@dataclass
class CompatContactSensorData:
    """Subset of IsaacLab ContactSensorData used by common MDP terms."""

    net_forces_w: torch.Tensor
    net_forces_w_history: torch.Tensor
    last_air_time: torch.Tensor | None = None
    current_air_time: torch.Tensor | None = None
    last_contact_time: torch.Tensor | None = None
    current_contact_time: torch.Tensor | None = None


class CompatContactSensor:
    """MetaSim-backed approximation of IsaacLab's `ContactSensor`."""

    def __init__(
        self,
        *,
        body_names: list[str],
        force_threshold: float = 0.0,
        track_air_time: bool = False,
        dt: float | None = None,
    ) -> None:
        self.body_names = list(body_names)
        self.force_threshold = float(force_threshold)
        self.track_air_time = bool(track_air_time)
        self._dt = float(dt) if dt is not None else None
        self.data: CompatContactSensorData | None = None
        self._in_contact: torch.Tensor | None = None
        self._last_air_time: torch.Tensor | None = None
        self._current_air_time: torch.Tensor | None = None
        self._last_contact_time: torch.Tensor | None = None
        self._current_contact_time: torch.Tensor | None = None

    def update_from_extra(self, extra: Any, *, device: torch.device | None = None, dt: float | None = None) -> None:
        """Update sensor buffers from a handler extra payload.

        Expected payload shapes:
        - `extra.contact_forces_history`: `(num_envs, history, num_bodies, 3)`
        """
        history = getattr(extra, "contact_forces_history", None)
        latest = getattr(extra, "contact_forces", None)
        if history is None:
            raise ValueError("Contact sensor extra payload missing `contact_forces_history`.")
        if latest is None:
            raise ValueError("Contact sensor extra payload missing `contact_forces`.")
        if device is not None:
            history = history.to(device)
            latest = latest.to(device)

        dt_s = float(dt) if dt is not None else self._dt
        if self.track_air_time and dt_s is not None:
            self._update_contact_times(net_forces_w=latest, dt=dt_s)

        self.data = CompatContactSensorData(
            net_forces_w=latest,
            net_forces_w_history=history,
            last_air_time=self._last_air_time if self.track_air_time else None,
            current_air_time=self._current_air_time if self.track_air_time else None,
            last_contact_time=self._last_contact_time if self.track_air_time else None,
            current_contact_time=self._current_contact_time if self.track_air_time else None,
        )

    def _ensure_time_buffers(self, *, num_envs: int, num_bodies: int, device: torch.device) -> None:
        if self._in_contact is None or self._in_contact.shape != (num_envs, num_bodies):
            self._in_contact = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.bool)
            self._last_air_time = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)
            self._current_air_time = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)
            self._last_contact_time = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)
            self._current_contact_time = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)

    def _update_contact_times(self, *, net_forces_w: torch.Tensor, dt: float) -> None:
        if not isinstance(net_forces_w, torch.Tensor):
            return
        num_envs = int(net_forces_w.shape[0])
        num_bodies = int(net_forces_w.shape[1])
        self._ensure_time_buffers(num_envs=num_envs, num_bodies=num_bodies, device=net_forces_w.device)
        assert self._in_contact is not None
        assert self._last_air_time is not None and self._current_air_time is not None
        assert self._last_contact_time is not None and self._current_contact_time is not None

        contact_now = net_forces_w.norm(dim=-1) > float(self.force_threshold)
        contact_prev = self._in_contact

        # liftoff (contact -> air): capture last contact time
        liftoff = contact_prev & (~contact_now)
        if liftoff.any():
            self._last_contact_time[liftoff] = self._current_contact_time[liftoff]
            self._current_contact_time[liftoff] = 0.0

        # touchdown (air -> contact): capture last air time
        touchdown = (~contact_prev) & contact_now
        if touchdown.any():
            self._last_air_time[touchdown] = self._current_air_time[touchdown]
            self._current_air_time[touchdown] = 0.0

        # accumulate current timers
        self._current_contact_time[contact_now] += float(dt)
        self._current_air_time[~contact_now] += float(dt)
        self._current_contact_time[~contact_now] = 0.0
        self._current_air_time[contact_now] = 0.0

        self._in_contact = contact_now

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, candidates=self.body_names, preserve_order=preserve_order)


@dataclass
class CompatCameraSensorData:
    """Subset of IsaacLab CameraData used by common MDP terms."""

    pos_w: torch.Tensor | None = None
    quat_w_world: torch.Tensor | None = None
    image_shape: tuple[int, int] | None = None
    intrinsic_matrices: torch.Tensor | None = None
    output: dict[str, torch.Tensor] | None = None
    info: list[dict[str, Any]] | None = None


class CompatCameraSensor:
    """MetaSim-backed approximation of IsaacLab's `Camera` sensor."""

    def __init__(self, *, data_types: list[str] | None = None, source_camera: str | None = None) -> None:
        self.data_types = list(data_types or ["rgb"])
        self.source_camera = source_camera
        self.data: CompatCameraSensorData | None = None

    def update_from_camera_state(self, camera_state: Any, *, device: torch.device | None = None) -> None:
        rgb = getattr(camera_state, "rgb", None)
        depth = getattr(camera_state, "depth", None)
        pos = getattr(camera_state, "pos", None)
        quat_world = getattr(camera_state, "quat_world", None)
        intrinsics = getattr(camera_state, "intrinsics", None)

        if device is not None:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.to(device)
            if isinstance(depth, torch.Tensor):
                depth = depth.to(device)
            if isinstance(pos, torch.Tensor):
                pos = pos.to(device)
            if isinstance(quat_world, torch.Tensor):
                quat_world = quat_world.to(device)
            if isinstance(intrinsics, torch.Tensor):
                intrinsics = intrinsics.to(device)

        output: dict[str, torch.Tensor] = {}
        if "rgb" in self.data_types and isinstance(rgb, torch.Tensor):
            output["rgb"] = rgb
        if "depth" in self.data_types and isinstance(depth, torch.Tensor):
            output["depth"] = depth

        image_shape: tuple[int, int] | None = None
        if isinstance(rgb, torch.Tensor) and rgb.ndim >= 3:
            image_shape = (int(rgb.shape[-3]), int(rgb.shape[-2]))  # (H, W)

        self.data = CompatCameraSensorData(
            pos_w=pos if isinstance(pos, torch.Tensor) else None,
            quat_w_world=quat_world if isinstance(quat_world, torch.Tensor) else None,
            image_shape=image_shape,
            intrinsic_matrices=intrinsics if isinstance(intrinsics, torch.Tensor) else None,
            output=output,
            info=[{} for _ in range(int(rgb.shape[0]))] if isinstance(rgb, torch.Tensor) and rgb.ndim >= 1 else [],
        )


@dataclass
class CompatRayCasterSensorData:
    """Minimal RayCaster-like data container (zeros by default)."""

    pos_w: torch.Tensor
    ray_hits_w: torch.Tensor


class CompatRayCasterSensor:
    """No-op RayCaster-compatible sensor (for import/runtime compatibility)."""

    def __init__(self, *, num_envs: int, num_rays: int = 0, device: torch.device | None = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        self.data = CompatRayCasterSensorData(
            pos_w=torch.zeros((int(num_envs), 3), device=dev, dtype=torch.float32),
            ray_hits_w=torch.zeros((int(num_envs), int(num_rays), 3), device=dev, dtype=torch.float32),
        )


@dataclass
class CompatImuSensorData:
    """Minimal IMU-like data container (zeros by default)."""

    quat_w: torch.Tensor
    projected_gravity_b: torch.Tensor


class CompatImuSensor:
    """No-op IMU-compatible sensor (for import/runtime compatibility)."""

    def __init__(self, *, num_envs: int, device: torch.device | None = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        self.data = CompatImuSensorData(
            quat_w=torch.zeros((int(num_envs), 4), device=dev, dtype=torch.float32),
            projected_gravity_b=torch.zeros((int(num_envs), 3), device=dev, dtype=torch.float32),
        )
