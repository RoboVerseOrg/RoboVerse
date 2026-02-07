from __future__ import annotations

from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState

from .assets import CompatArticulation, CompatRigidObject
from .sensors import CompatCameraSensor, CompatContactSensor, CompatImuSensor, CompatRayCasterSensor


class CompatScene:
    """MetaSim-backed facade that mimics the subset of IsaacLab's `InteractiveScene` used by MDP terms."""

    def __init__(
        self,
        *,
        handler: Any,
        scenario: ScenarioCfg,
        asset_name_map: dict[str, str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._handler = handler
        self._scenario = scenario
        self._asset_name_map = dict(asset_name_map or {})

        self.num_envs = int(getattr(scenario, "num_envs", 1))
        self.env_origins = self._infer_env_origins(device=device)

        self.articulations: dict[str, CompatArticulation] = {}
        self.rigid_objects: dict[str, CompatRigidObject] = {}
        self.sensors: dict[str, Any] = {}

        for robot in scenario.robots:
            if robot.name is None:
                continue
            self.articulations[robot.name] = CompatArticulation(
                handler=handler, robot_cfg=robot, sim_name=robot.name, env_origins=self.env_origins
            )

        for obj in scenario.objects:
            self.rigid_objects[obj.name] = CompatRigidObject(
                handler=handler, sim_name=obj.name, env_origins=self.env_origins
            )

        # Heuristic alias to reduce task-level changes: map "robot" to the first robot.
        if "robot" not in self._asset_name_map and len(scenario.robots) == 1 and scenario.robots[0].name is not None:
            self._asset_name_map["robot"] = scenario.robots[0].name

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------
    def resolve_asset_name(self, name: str) -> str:
        return self._asset_name_map.get(name, name)

    def __getitem__(self, name: str):
        resolved = self.resolve_asset_name(name)
        if resolved in self.articulations:
            return self.articulations[resolved]
        if resolved in self.rigid_objects:
            return self.rigid_objects[resolved]
        raise KeyError(f"Unknown scene entity '{name}' (resolved to '{resolved}').")

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def add_contact_sensor(
        self,
        *,
        name: str,
        body_names: list[str],
        force_threshold: float = 0.0,
        track_air_time: bool = False,
        dt: float | None = None,
    ) -> CompatContactSensor:
        sensor = CompatContactSensor(
            body_names=body_names,
            force_threshold=force_threshold,
            track_air_time=track_air_time,
            dt=dt,
        )
        self.sensors[name] = sensor
        return sensor

    def add_camera_sensor(
        self, *, name: str, data_types: list[str] | None = None, source_camera: str | None = None
    ) -> CompatCameraSensor:
        sensor = CompatCameraSensor(data_types=data_types, source_camera=source_camera or name)
        self.sensors[name] = sensor
        return sensor

    def add_ray_caster_sensor(self, *, name: str, num_rays: int = 0) -> CompatRayCasterSensor:
        sensor = CompatRayCasterSensor(num_envs=self.num_envs, num_rays=num_rays, device=self.env_origins.device)
        self.sensors[name] = sensor
        return sensor

    def add_imu_sensor(self, *, name: str) -> CompatImuSensor:
        sensor = CompatImuSensor(num_envs=self.num_envs, device=self.env_origins.device)
        self.sensors[name] = sensor
        return sensor

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def update_from_states(self, full_state: TensorState) -> None:
        self.env_origins = self._infer_env_origins(
            device=full_state.robots[next(iter(full_state.robots))].root_state.device if full_state.robots else None
        )

        for art in self.articulations.values():
            art.update_from_states(full_state=full_state, env_origins=self.env_origins)
        for obj in self.rigid_objects.values():
            obj.update_from_states(full_state=full_state, env_origins=self.env_origins)

        # Contact sensor: populate from `extras["contact_forces"]` when available.
        extras = getattr(full_state, "extras", {}) or {}
        contact_payload = extras.get("contact_forces") if isinstance(extras, dict) else None
        if isinstance(contact_payload, dict):
            # Default mapping: apply per-robot payload to every contact sensor.
            for sensor in self.sensors.values():
                if not isinstance(sensor, CompatContactSensor):
                    continue
                if len(self._scenario.robots) != 1 or self._scenario.robots[0].name is None:
                    continue
                robot_name = self._scenario.robots[0].name
                if robot_name in contact_payload:
                    sensor.update_from_extra(
                        contact_payload[robot_name], device=full_state.robots[robot_name].root_state.device
                    )

        # Camera sensors: populate from `full_state.cameras` when available.
        cameras = getattr(full_state, "cameras", None)
        if isinstance(cameras, dict) and cameras:
            for sensor in self.sensors.values():
                if not isinstance(sensor, CompatCameraSensor):
                    continue
                cam_name = sensor.source_camera or ""
                if cam_name and cam_name in cameras:
                    sensor.update_from_camera_state(cameras[cam_name], device=self.env_origins.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _infer_env_origins(self, *, device: torch.device | None = None) -> torch.Tensor:
        # Vectorized IsaacSim handler exposes per-env origins through an IsaacLab scene.
        scene = getattr(self._handler, "scene", None)
        origins = getattr(scene, "env_origins", None) if scene is not None else None
        if origins is None:
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros((self.num_envs, 3), device=dev, dtype=torch.float32)
        return origins
