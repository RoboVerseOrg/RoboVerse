from __future__ import annotations

from metasim.utils.configclass import configclass

_DEFAULT_FILE_TYPE = {
    "isaaclab": "usd",
    "isaacsim": "usd",
    "pybullet": "urdf",
    "sapien2": "urdf",
    "sapien3": "urdf",
    "genesis": "urdf",
    "isaacgym": "urdf",
    "mujoco": "mjcf",
    "mjx": "mjx_mjcf",
}


@configclass
class SceneCfg:
    """Base config class for scenes."""

    name: str | None = None
    usd_path: str | None = None
    urdf_path: str | None = None
    mjcf_path: str | None = None

    positions: list[tuple[float, float, float]] | None = None
    default_position: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None

    scale: tuple[float, float, float] | None = None

    file_type: dict[str, str] = _DEFAULT_FILE_TYPE.copy()

    def file_name(self, sim_name):
        file_type = self.file_type[sim_name]
        if file_type == "usd":
            return self.usd_path
        elif file_type == "urdf":
            return self.urdf_path
        elif file_type == "mjcf":
            return self.mjcf_path
        else:
            raise ValueError(f"Invalid file type: {file_type}")
