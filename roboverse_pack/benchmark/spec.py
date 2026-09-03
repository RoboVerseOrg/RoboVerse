from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkObjectSpec:
    """Portable object metadata for benchmark task construction."""

    name: str
    kind: str
    pos_cm: tuple[float, float, float]
    half_extents_cm: tuple[float, float, float]
    movable: bool = True


@dataclass(frozen=True)
class BenchmarkCameraPreset:
    """Named benchmark camera pose."""

    name: str
    width: int
    height: int
    pos: tuple[float, float, float]
    look_at: tuple[float, float, float]


@dataclass(frozen=True)
class BenchmarkSceneSpec:
    """Robot-agnostic benchmark scene definition."""

    objects: tuple[BenchmarkObjectSpec, ...]
    camera_presets: tuple[BenchmarkCameraPreset, ...]
    physics_hints: Mapping[str, object]


@dataclass(frozen=True)
class BenchmarkRobotTeleopProfile:
    """Robot-specific teleoperation control metadata."""

    robot: str
    control_joint_count: int
    joint_slices: Mapping[str, tuple[int, int]]
    body_names: Mapping[str, str]


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """Task-level benchmark metadata separated from robot selection."""

    name: str
    family: str
    description: str
    scene: BenchmarkSceneSpec
    supported_simulators: tuple[str, ...]
    supported_robots: tuple[str, ...]
    default_robot: str
    robot_profiles: Mapping[str, BenchmarkRobotTeleopProfile]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.default_robot not in self.supported_robots:
            raise ValueError(f"default robot {self.default_robot!r} is not listed in supported_robots")
        for robot in self.supported_robots:
            if robot not in self.robot_profiles:
                raise ValueError(f"missing robot teleop profile for {robot!r}")

    def robot_profile(self, robot: str | None = None) -> BenchmarkRobotTeleopProfile:
        """Return the robot-specific teleoperation profile."""
        selected = self.default_robot if robot is None else str(robot)
        try:
            return self.robot_profiles[selected]
        except KeyError as exc:
            supported = ", ".join(self.supported_robots)
            raise ValueError(f"task {self.name} does not support robot {selected!r}; supported: {supported}") from exc
