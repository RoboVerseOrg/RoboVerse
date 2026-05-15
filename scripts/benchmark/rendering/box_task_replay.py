from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BUNDLE_ROOT = Path("outputs/assets/box_task_replay_render_bundle_clean")
DEFAULT_TRAJ_REL = Path("assets/traj/task3_meshycup_openarm_wuji_20260513_232823_0_v2.pkl")
DEFAULT_SRC_TOTAL_FRAMES = 849

_REPLAY_ROBOT_FIELDS = (
    "dof_pos",
    "dof_vel",
    "dof_pos_target",
    "dof_vel_target",
    "dof_torque",
)


@dataclass(frozen=True)
class BoxTaskBundlePaths:
    bundle_root: Path
    traj_path: Path
    asset_root: Path
    cardboard_box_usd: Path
    soda_can_usd: Path
    scented_candle_usd: Path

    @classmethod
    def from_root(cls, bundle_root: Path, traj_path: Path | None = None) -> BoxTaskBundlePaths:
        root = bundle_root.resolve()
        resolved_traj_path = (traj_path if traj_path is not None else root / DEFAULT_TRAJ_REL).resolve()
        asset_root = root / "assets" / "local_pack_box"
        paths = cls(
            bundle_root=root,
            traj_path=resolved_traj_path,
            asset_root=asset_root,
            cardboard_box_usd=asset_root / "cardboard_box" / "cardboard_box.usd",
            soda_can_usd=asset_root / "feast_soda_can" / "feast_soda_can.usd",
            scented_candle_usd=asset_root / "feast_scented_candle" / "feast_scented_candle.usd",
        )
        paths.validate()
        return paths

    def validate(self) -> None:
        for path in (
            self.traj_path,
            self.cardboard_box_usd,
            self.soda_can_usd,
            self.scented_candle_usd,
        ):
            if not path.is_file():
                raise FileNotFoundError(str(path))


def normalize_scene_token(scene_token: str) -> str:
    token = scene_token.strip()
    if token.endswith(".usda"):
        token = token[: -len(".usda")]

    if token.startswith("kujiale_scene_"):
        scene_id = token.removeprefix("kujiale_scene_")
    elif token.startswith("usd_"):
        scene_id = token.removeprefix("usd_")
    elif token.isdigit():
        scene_id = token
    else:
        raise ValueError(f"Invalid scene token: {scene_token}")

    if not scene_id.isdigit() or len(scene_id) not in (3, 4):
        raise ValueError("Scene token must be 3 or 4 digits")

    return f"kujiale_scene_{int(scene_id):04d}"


def parse_scene_tokens(scene_text: str) -> list[str]:
    return [normalize_scene_token(token) for token in scene_text.split(",") if token.strip()]


def compute_output_frames(
    src_total_frames: int,
    fps: int,
    duration_sec: float | None,
    out_frames: int | None,
) -> int:
    if duration_sec is not None and out_frames is not None:
        raise ValueError("Use either --out-frames or --duration-sec")
    if out_frames is not None:
        if out_frames <= 0:
            raise ValueError("--out-frames must be positive")
        return out_frames
    if duration_sec is not None:
        if fps <= 0:
            raise ValueError("--fps must be positive")
        if duration_sec <= 0:
            raise ValueError("--duration-sec must be positive")
        return max(1, int(round(duration_sec * fps)))
    if src_total_frames <= 0:
        raise ValueError("src_total_frames must be positive")
    return src_total_frames


def frame_to_source_indices(src_total_frames: int, out_frames: int) -> np.ndarray:
    if src_total_frames <= 0:
        raise ValueError("src_total_frames must be positive")
    if out_frames <= 0:
        raise ValueError("out_frames must be positive")
    if out_frames == 1:
        return np.array([0], dtype=np.int32)
    return np.linspace(0, src_total_frames - 1, out_frames).astype(np.int32)


def scene_frame_bounds(out_frames: int, scene_count: int) -> list[tuple[int, int]]:
    if out_frames <= 0:
        raise ValueError("out_frames must be positive")
    if scene_count <= 0:
        raise ValueError("scene_count must be positive")
    if scene_count > out_frames:
        raise ValueError("scene_count must not exceed out_frames")
    return [
        (int(i * out_frames / scene_count), int((i + 1) * out_frames / scene_count))
        for i in range(scene_count)
    ]


def build_finger_joint_map() -> dict[str, str]:
    return {
        f"{side}_hand_finger{finger}_joint{joint}": f"{side}_finger{finger}_joint{joint}"
        for side in ("left", "right")
        for finger in range(1, 6)
        for joint in range(1, 5)
    }


def patch_state_for_replay(state: dict[str, Any], robot_name: str = "openarm_wuji") -> dict[str, Any]:
    patched = deepcopy(state)
    mapping = build_finger_joint_map()

    robot_states: list[MutableMapping[str, Any]] = []
    if isinstance(patched.get(robot_name), MutableMapping):
        robot_states.append(patched[robot_name])
    if isinstance(patched.get("robots"), MutableMapping) and isinstance(patched["robots"].get(robot_name), MutableMapping):
        robot_states.append(patched["robots"][robot_name])
    if isinstance(patched.get("robot_states"), MutableMapping) and isinstance(
        patched["robot_states"].get(robot_name), MutableMapping
    ):
        robot_states.append(patched["robot_states"][robot_name])
    if any(field in patched for field in _REPLAY_ROBOT_FIELDS):
        robot_states.append(patched)

    seen: set[int] = set()
    for robot_state in robot_states:
        if id(robot_state) in seen:
            continue
        seen.add(id(robot_state))
        for field in _REPLAY_ROBOT_FIELDS:
            field_state = robot_state.get(field)
            if isinstance(field_state, MutableMapping):
                _rewrite_legacy_joint_keys(field_state, mapping)

    return patched


def _rewrite_legacy_joint_keys(field_state: MutableMapping[str, Any], mapping: dict[str, str]) -> None:
    for legacy_name, replay_name in mapping.items():
        if legacy_name in field_state:
            value = field_state.pop(legacy_name)
            field_state.setdefault(replay_name, value)


def _build_scenario_cfg(scenario_cls: type, deferred_attrs: dict[str, Any], **kwargs: Any) -> Any:
    supported_kwargs = _supported_constructor_kwargs(scenario_cls, kwargs)
    scenario = scenario_cls(**supported_kwargs)
    for name, value in deferred_attrs.items():
        if name not in supported_kwargs:
            _set_scenario_attr(scenario, name, value)
    return scenario


def _supported_constructor_kwargs(scenario_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(scenario_cls).parameters.values()
    except (TypeError, ValueError):
        return kwargs

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs

    supported_names = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    supported_names.discard("self")
    return {name: value for name, value in kwargs.items() if name in supported_names}


def _set_scenario_attr(scenario: Any, name: str, value: Any) -> None:
    try:
        setattr(scenario, name, value)
        return
    except (AttributeError, TypeError, ValueError):
        pass

    scenario_dict = getattr(scenario, "__dict__", None)
    if isinstance(scenario_dict, dict):
        scenario_dict[name] = value
        return

    raise AttributeError(f"ScenarioCfg does not allow setting {name!r} after construction")


def build_box_task_scenario(
    *,
    paths: BoxTaskBundlePaths,
    simulator: str,
    scene: str,
    width: int,
    height: int,
    camera_pos: tuple[float, float, float],
    camera_look_at: tuple[float, float, float],
    head_light_intensity: float,
):
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import SphereLightCfg
    from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
    from metasim.scenario.render import RenderCfg
    from metasim.scenario.scenario import ScenarioCfg, SimParamCfg

    scenario = _build_scenario_cfg(
        ScenarioCfg,
        deferred_attrs={"renderer": simulator, "scene": scene, "requested_scene_name": scene},
        objects=[
            PrimitiveCubeCfg(
                name="front_table",
                size=(0.60, 0.70, 0.04),
                mass=80.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.85, 0.78, 0.62),
                fix_base_link=True,
                default_position=(0.55, 0.0, 0.33),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
            ),
            RigidObjCfg(
                name="cardboard_box",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=str(paths.cardboard_box_usd),
            ),
            RigidObjCfg(
                name="feast_soda_can",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=str(paths.soda_can_usd),
            ),
            RigidObjCfg(
                name="feast_scented_candle",
                physics=PhysicStateType.RIGIDBODY,
                usd_path=str(paths.scented_candle_usd),
            ),
        ],
        robots=["openarm_wuji"],
        cameras=[
            PinholeCameraCfg(
                name="camera0",
                data_types=["rgb"],
                width=width,
                height=height,
                pos=camera_pos,
                look_at=camera_look_at,
            )
        ],
        lights=[
            SphereLightCfg(
                name="overhead_light",
                intensity=float(head_light_intensity),
                color=(1.0, 1.0, 1.0),
                radius=0.15,
                pos=(0.55, 0.0, 1.45),
                is_global=False,
            )
        ],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator=simulator,
        renderer=simulator,
        headless=True,
        num_envs=1,
        scene=scene,
        render=RenderCfg(mode="pathtracing"),
    )
    return scenario


def build_decode_scenario():
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
    from metasim.scenario.scenario import ScenarioCfg, SimParamCfg

    return _build_scenario_cfg(
        ScenarioCfg,
        deferred_attrs={"renderer": "isaacsim"},
        objects=[
            PrimitiveCubeCfg(
                name="front_table",
                size=(0.60, 0.70, 0.04),
                mass=80.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.85, 0.78, 0.62),
                fix_base_link=True,
            ),
            RigidObjCfg(name="cardboard_box", physics=PhysicStateType.RIGIDBODY),
            RigidObjCfg(name="feast_soda_can", physics=PhysicStateType.RIGIDBODY),
            RigidObjCfg(name="feast_scented_candle", physics=PhysicStateType.RIGIDBODY),
        ],
        robots=["openarm_wuji"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator="isaacsim",
        renderer="isaacsim",
        headless=True,
        num_envs=1,
    )
