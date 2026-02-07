from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from metasim.queries.contact_force import ContactForces
from metasim.scenario.scenario import ScenarioCfg

from .capabilities import CAPABILITIES
from .contract import CompatTermError, TermContext, WarnOnce
from .sensors import CompatCameraSensorData, CompatContactSensorData
from .utils import iter_public_fields


@dataclass(frozen=True)
class ContactSensorPlan:
    history_length: int
    query_enabled: bool
    sensor_name: str = "contact_forces"
    track_air_time: bool = False
    force_threshold: float = 0.0


class SensorRegistry:
    """Backend-gated construction of compat sensors and handler optional queries."""

    def __init__(self, *, strict: bool, warn_once: WarnOnce) -> None:
        self._strict = strict
        self._warn_once = warn_once

    def plan_optional_queries(
        self, *, cfg: Any, scenario: ScenarioCfg
    ) -> tuple[dict[str, Any], ContactSensorPlan | None]:
        scene_cfg = getattr(cfg, "scene", None)
        contact_sensor_cfg = getattr(scene_cfg, "contact_forces", None) if scene_cfg is not None else None
        if contact_sensor_cfg is None or getattr(contact_sensor_cfg, "history_length", None) is None:
            return {}, None

        history_len = int(contact_sensor_cfg.history_length)
        track_air_time = bool(getattr(contact_sensor_cfg, "track_air_time", False))
        force_threshold = float(getattr(contact_sensor_cfg, "force_threshold", 0.0))
        backend = scenario.simulator
        support = CAPABILITIES.check(capability=CAPABILITIES.OPTIONAL_QUERY_CONTACT_FORCES, backend=backend)
        if not support.supported:
            if self._strict:
                raise CompatTermError(
                    ctx=TermContext(kind="sensor", name="contact_forces"),
                    backend=backend,
                    message=(support.reason + (f" {support.how_to}" if support.how_to else "")),
                )
            self._warn_once.warning(
                "sensor.contact_forces/unsupported_backend",
                "sensor.contact_forces: requested but contact force query is unavailable for backend '{}'. "
                "Proceeding with a zero/no-op contact sensor (contact-based terms become no-ops).",
                backend,
            )
            return {}, ContactSensorPlan(
                history_length=history_len,
                query_enabled=False,
                sensor_name="contact_forces",
                track_air_time=track_air_time,
                force_threshold=force_threshold,
            )

        return {"contact_forces": ContactForces(history_length=history_len)}, ContactSensorPlan(
            history_length=history_len,
            query_enabled=True,
            sensor_name="contact_forces",
            track_air_time=track_air_time,
            force_threshold=force_threshold,
        )

    def setup_scene_sensors(self, *, env: Any, cfg: Any, plan: ContactSensorPlan | None) -> None:
        """Install compat sensors into env.scene based on IsaacLab-style scene cfg.

        This is a best-effort registry:
        - contact sensor is backed by the MetaSim contact force optional query when available
        - other sensors (camera/raycast/imu) are installed as compatible stubs unless backed by handler state
        """
        scene_cfg = getattr(cfg, "scene", None)
        if scene_cfg is None:
            return

        # ------------------------------------------------------------------
        # Contact sensor (query-backed when available)
        # ------------------------------------------------------------------
        if plan is not None:
            self._setup_contact_sensor(env=env, plan=plan)

        # ------------------------------------------------------------------
        # Camera / other sensor stubs (API surface for imports/terms)
        # ------------------------------------------------------------------
        for sensor_name, sensor_cfg in iter_public_fields(scene_cfg):
            if sensor_cfg is None:
                continue

            cfg_type = getattr(getattr(sensor_cfg, "__class__", None), "__name__", "")
            if cfg_type in {"ContactSensorCfg"}:
                continue

            # Camera (data is populated from handler state when camera exists; otherwise zeros when size known).
            if cfg_type in {"CameraCfg", "TiledCameraCfg"}:
                data_types = getattr(sensor_cfg, "data_types", None) or ["rgb"]
                sensor = env.scene.add_camera_sensor(
                    name=sensor_name, data_types=list(data_types), source_camera=sensor_name
                )

                width = getattr(sensor_cfg, "width", None)
                height = getattr(sensor_cfg, "height", None)
                if (
                    sensor.data is None
                    and isinstance(width, int)
                    and isinstance(height, int)
                    and width > 0
                    and height > 0
                ):
                    output: dict[str, torch.Tensor] = {}
                    if "rgb" in data_types:
                        output["rgb"] = torch.zeros(
                            (env.num_envs, height, width, 3), device=env.device, dtype=torch.uint8
                        )
                    if "depth" in data_types:
                        output["depth"] = torch.zeros(
                            (env.num_envs, height, width), device=env.device, dtype=torch.float32
                        )
                    sensor.data = CompatCameraSensorData(
                        image_shape=(height, width),
                        intrinsic_matrices=None,
                        output=output,
                        info=[{} for _ in range(int(env.num_envs))],
                    )
                continue

            # RayCaster/IMU: no-op stubs for now (installed only for API compatibility).
            if cfg_type in {"RayCasterCfg", "ImuCfg"}:
                if cfg_type == "RayCasterCfg":
                    env.scene.add_ray_caster_sensor(name=sensor_name, num_rays=0)
                else:
                    env.scene.add_imu_sensor(name=sensor_name)
                continue

    def _setup_contact_sensor(self, *, env: Any, plan: ContactSensorPlan) -> None:
        """Install contact sensor into env.scene and attach zero buffers when query is disabled."""
        if len(env.scenario.robots) != 1 or env.scenario.robots[0].name is None:
            if self._strict:
                raise CompatTermError(
                    ctx=TermContext(kind="sensor", name=plan.sensor_name),
                    backend=getattr(env.scenario, "simulator", None),
                    message="contact sensor requires exactly one named robot in ScenarioCfg for MVP compat.",
                )
            self._warn_once.warning(
                "sensor.contact_forces/multi_robot",
                "sensor.contact_forces: requested but scenario has !=1 robot; skipping sensor setup.",
            )
            return

        robot_name = env.scenario.robots[0].name
        sensor = env.scene.add_contact_sensor(
            name=plan.sensor_name,
            body_names=env.scene.articulations[robot_name].body_names,
            force_threshold=plan.force_threshold,
            track_air_time=plan.track_air_time,
            dt=float(getattr(env, "step_dt", 0.0) or 0.0),
        )

        if not plan.query_enabled:
            zeros = torch.zeros(
                (env.num_envs, int(plan.history_length), len(sensor.body_names), 3),
                device=env.device,
                dtype=torch.float32,
            )
            sensor.data = CompatContactSensorData(
                net_forces_w=torch.zeros(
                    (env.num_envs, len(sensor.body_names), 3), device=env.device, dtype=torch.float32
                ),
                net_forces_w_history=zeros,
                last_air_time=torch.zeros(
                    (env.num_envs, len(sensor.body_names)), device=env.device, dtype=torch.float32
                )
                if plan.track_air_time
                else None,
                current_air_time=torch.zeros(
                    (env.num_envs, len(sensor.body_names)), device=env.device, dtype=torch.float32
                )
                if plan.track_air_time
                else None,
                last_contact_time=torch.zeros(
                    (env.num_envs, len(sensor.body_names)), device=env.device, dtype=torch.float32
                )
                if plan.track_air_time
                else None,
                current_contact_time=torch.zeros(
                    (env.num_envs, len(sensor.body_names)), device=env.device, dtype=torch.float32
                )
                if plan.track_air_time
                else None,
            )
