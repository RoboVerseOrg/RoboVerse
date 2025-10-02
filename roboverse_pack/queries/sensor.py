# metasim/queries/sensor_queries.py
from __future__ import annotations

import importlib

from metasim.queries.base import BaseQueryType

_sensor_cache: dict[int, dict[str, tuple[int, int]]] = {}


def _get_sensor_info(mj_model, name: str) -> tuple[int, int]:
    """Get sensor ID and address for a given sensor name."""
    key = id(mj_model)
    sensor_dict = _sensor_cache.setdefault(key, {})
    if name not in sensor_dict:
        sensor_id = mj_model.sensor(name).id
        sensor_adr = mj_model.sensor_adr[sensor_id]
        sensor_dim = mj_model.sensor_dim[sensor_id]
        sensor_dict[name] = (sensor_adr, sensor_dim)
    return sensor_dict[name]


class SensorData(BaseQueryType):
    """MuJoCo sensor data query (works for MJX & raw MuJoCo)."""

    def __init__(self, sensor_name: str):
        super().__init__()
        self.sensor_name = sensor_name
        self._sensor_info: tuple[int, int] | None = None  # (sensor_adr, sensor_dim) resolved during bind

    def bind_handler(self, handler, *args, **kwargs):
        """Remember the sensor info once the handler is known."""
        super().bind_handler(handler, *args, **kwargs)  # Remember to call the base method
        mod = handler.__class__.__module__
        if mod.startswith("metasim.sim.mjx"):
            robot_name = handler._robot.name
            full_name = f"{robot_name}/{self.sensor_name}" if "/" not in self.sensor_name else self.sensor_name
            self._sensor_info = _get_sensor_info(handler._mj_model, full_name)

        elif mod.startswith("metasim.sim.mujoco"):
            robot_name = handler.robot.name
            full_name = f"{robot_name}/{self.sensor_name}" if "/" not in self.sensor_name else self.sensor_name
            self._sensor_info = _get_sensor_info(handler.physics.model, full_name)

        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for SensorData query")

    def __call__(self):
        """Return sensor data whenever `get_extra()` is invoked.

        * Heavy libraries are imported **inside** the relevant branch only.
        """
        mod = self.handler.__class__.__module__

        if mod.startswith("metasim.sim.mjx"):
            # ── MJX branch ────────────────────────────────────────────────
            torch = importlib.import_module("torch")
            jax = importlib.import_module("jax")

            sensor_adr, sensor_dim = self._sensor_info
            val = self.handler._data.sensordata[sensor_adr : sensor_adr + sensor_dim]
            from metasim.sim.mjx.mjx_helper import j2t

            return j2t(val)

        elif mod.startswith("metasim.sim.mujoco"):
            # ── raw MuJoCo branch ────────────────────────────────────────
            torch = importlib.import_module("torch")

            sensor_adr, sensor_dim = self._sensor_info
            data = self.handler.data.sensordata[sensor_adr : sensor_adr + sensor_dim]
            return torch.as_tensor(data).unsqueeze(0)

        raise ValueError(f"Unsupported handler type: {type(self.handler)} for SensorData query")
