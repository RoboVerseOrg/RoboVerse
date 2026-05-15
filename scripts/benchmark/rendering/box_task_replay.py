from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
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
_DEFAULT_JOINT_LIMIT_MARGIN = 1.0e-3


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
    return np.rint(np.linspace(0, src_total_frames - 1, out_frames)).astype(np.int32)


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


def patch_openarm_wuji_robot_cfg(robot_cfg: Any) -> Any:
    """Rewrite legacy Wuji hand names in robot cfgs to match the generated USD."""
    mapping = build_finger_joint_map()
    for attr_name in ("joint_names", "left_hand_joint_names", "right_hand_joint_names", "ee_joint_names"):
        names = getattr(robot_cfg, attr_name, None)
        if isinstance(names, (list, tuple)):
            _set_object_attr(robot_cfg, attr_name, [_canonical_joint_name(name, mapping) for name in names])

    for attr_name in ("joint_limits", "default_joint_positions", "actuators", "control_type"):
        keyed_values = getattr(robot_cfg, attr_name, None)
        if isinstance(keyed_values, MutableMapping):
            _set_object_attr(robot_cfg, attr_name, _rewrite_legacy_mapping_keys(keyed_values, mapping))
    _set_generated_wuji_hand_defaults(robot_cfg)
    _clamp_robot_default_joint_positions(robot_cfg)

    body_name_map = {
        "left_hand_palm_link": "left_palm_link",
        "right_hand_palm_link": "right_palm_link",
    }
    for attr_name in ("left_ee_body_name", "right_ee_body_name", "ee_body_name"):
        body_name = getattr(robot_cfg, attr_name, None)
        if body_name in body_name_map:
            _set_object_attr(robot_cfg, attr_name, body_name_map[body_name])

    return robot_cfg


def patch_openarm_wuji_scenario_robot_cfgs(scenario: Any) -> Any:
    for robot_cfg in getattr(scenario, "robots", ()) or ():
        if getattr(robot_cfg, "name", None) in {"openarm_wuji", "openarm_bimanual_wuji"}:
            patch_openarm_wuji_robot_cfg(robot_cfg)
    return scenario


def disable_metasim_forced_exit_on_close() -> None:
    os.environ["METASIM_FORCE_EXIT_ON_CLOSE"] = "0"


def tensorize_replay_state(state: dict[str, Any], tensor_factory: Any | None = None) -> dict[str, Any]:
    if tensor_factory is None:
        import torch

        tensor_factory = torch.as_tensor
    return _tensorize_pose_fields(deepcopy(state), tensor_factory)


def close_decode_env_without_closing_kit(env: Any) -> None:
    handler = getattr(env, "handler", None)
    if handler is not None:
        setattr(handler, "_owns_simulation_app", False)
    env.close()


def install_numpy_pickle_aliases() -> list[str]:
    """Install NumPy 1.x/2.x module aliases needed by pickled trajectories."""
    import importlib
    import sys
    import warnings

    inserted: list[str] = []

    def set_alias(name: str, module: Any) -> None:
        if name not in sys.modules:
            sys.modules[name] = module
            inserted.append(name)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core is deprecated.*", category=DeprecationWarning)
        try:
            numpy_public_core = importlib.import_module("numpy.core")
        except ModuleNotFoundError:
            numpy_public_core = None
        try:
            numpy_private_core = importlib.import_module("numpy._core")
        except ModuleNotFoundError:
            numpy_private_core = None

        if numpy_public_core is not None:
            set_alias("numpy._core", numpy_public_core)
            if hasattr(numpy_public_core, "multiarray"):
                set_alias("numpy._core.multiarray", numpy_public_core.multiarray)
            if hasattr(numpy_public_core, "umath"):
                set_alias("numpy._core.umath", numpy_public_core.umath)

        if numpy_private_core is not None:
            set_alias("numpy.core", numpy_private_core)
            if hasattr(numpy_private_core, "multiarray"):
                set_alias("numpy.core.multiarray", numpy_private_core.multiarray)
            if hasattr(numpy_private_core, "umath"):
                set_alias("numpy.core.umath", numpy_private_core.umath)

    return inserted


def load_raw_v2_pickle(path: Path) -> Any:
    import pickle
    import sys

    inserted_aliases = install_numpy_pickle_aliases()
    try:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
    finally:
        for name in reversed(inserted_aliases):
            sys.modules.pop(name, None)


def convert_v2_state_to_v3(state: dict[str, Any], robot_name: str) -> dict[str, Any]:
    if "robots" in state and "objects" in state:
        return {
            "robots": deepcopy(state["robots"]),
            "objects": deepcopy(state["objects"]),
        }

    converted: dict[str, Any] = {"robots": {}, "objects": {}}
    for name, entity_state in state.items():
        if name == robot_name:
            converted["robots"][name] = deepcopy(entity_state)
        elif name not in ("robots", "objects"):
            converted["objects"][name] = deepcopy(entity_state)
    return converted


def _first_v2_episode(payload: Any, robot_name: str) -> dict[str, Any]:
    if not isinstance(payload, MutableMapping) or robot_name not in payload:
        raise ValueError(f"Trajectory payload does not contain robot {robot_name!r}")
    episodes = payload[robot_name]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Trajectory payload for {robot_name!r} has no episodes")
    episode = episodes[0]
    if not isinstance(episode, MutableMapping):
        raise ValueError(f"First trajectory episode for {robot_name!r} is not a mapping")
    return episode


def load_v2_episode_states(traj_path: Path, robot_name: str = "openarm_wuji") -> list[dict[str, Any]]:
    payload = load_raw_v2_pickle(traj_path)
    episode = _first_v2_episode(payload, robot_name)
    episode_states = episode.get("states")
    if not isinstance(episode_states, list) or not episode_states:
        raise ValueError(f"First trajectory episode for {robot_name!r} has no states")
    return [convert_v2_state_to_v3(state, robot_name) for state in episode_states]


def load_v2_init_state(traj_path: Path, robot_name: str = "openarm_wuji") -> dict[str, Any] | None:
    payload = load_raw_v2_pickle(traj_path)
    episode = _first_v2_episode(payload, robot_name)
    init_state = episode.get("init_state")
    if init_state is None:
        return None
    if not isinstance(init_state, MutableMapping):
        raise ValueError(f"Initial trajectory state for {robot_name!r} is not a mapping")
    return convert_v2_state_to_v3(init_state, robot_name)


def decode_trajectory_tensor_states(traj_path: Path, source_indices) -> dict[int, Any]:
    import torch
    from metasim.utils import hf_util
    from metasim.task.base import BaseTaskEnv

    disable_metasim_forced_exit_on_close()
    # The replay bundle is local and complete; avoid recursive Hub downloads during smoke renders.
    hf_util.check_and_download_recursive = lambda filepaths, n_processes=16: None
    _patch_isaacsim_render_settings_for_decode()

    episode_states = load_v2_episode_states(traj_path)
    init_state = load_v2_init_state(traj_path)
    unique_indices = sorted({int(index) for index in source_indices})
    robot_name = "openarm_wuji"
    bundle_paths = BoxTaskBundlePaths.from_root(traj_path.parents[2], traj_path=traj_path)
    decode_scenario = build_decode_scenario(paths=bundle_paths)

    class BoxTaskDecodeEnv(BaseTaskEnv):
        scenario = decode_scenario

        def _get_initial_states(self) -> list[dict[str, Any]]:
            if init_state is not None:
                initial_state = patch_state_for_replay(init_state, robot_name=robot_name)
                initial_state.setdefault("cameras", {})
                initial_state.setdefault("extras", {})
                return [tensorize_replay_state(initial_state)]
            return [{"objects": {}, "robots": {}, "cameras": {}, "extras": {}}]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = None
    try:
        env = BoxTaskDecodeEnv(scenario=BoxTaskDecodeEnv.scenario, device=device)
        tensor_states: dict[int, Any] = {}
        for source_index in unique_indices:
            state = patch_state_for_replay(episode_states[source_index], robot_name=robot_name)
            state = tensorize_replay_state(state)
            env.handler.set_states([state])
            tensor_states[source_index] = env.handler.get_states(mode="tensor")
        return tensor_states
    finally:
        if env is not None:
            close_decode_env_without_closing_kit(env)


def _patch_isaacsim_render_settings_for_decode() -> None:
    import importlib

    isaacsim_module = importlib.import_module("metasim.sim.isaacsim.isaacsim")
    handler_cls = getattr(isaacsim_module, "IsaacsimHandler")
    original = handler_cls._load_render_settings
    if getattr(original, "_box_task_decode_patch", False):
        return

    def load_render_settings_without_replicator(self) -> None:
        try:
            return original(self)
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("omni.replicator"):
                raise
            import carb

            settings = carb.settings.get_settings()
            if self.scenario.render.mode == "pathtracing":
                settings.set_string("/rtx/rendermode", "PathTracing")
            elif self.scenario.render.mode == "raytracing":
                settings.set_string("/rtx/rendermode", "RayTracedLighting")
            elif self.scenario.render.mode == "rasterization":
                raise ValueError("Isaaclab does not support rasterization")
            else:
                raise ValueError(f"Unknown render mode: {self.scenario.render.mode}")

    load_render_settings_without_replicator._box_task_decode_patch = True
    handler_cls._load_render_settings = load_render_settings_without_replicator


def _rewrite_legacy_joint_keys(field_state: MutableMapping[str, Any], mapping: dict[str, str]) -> None:
    for legacy_name, replay_name in mapping.items():
        if legacy_name in field_state:
            value = field_state.pop(legacy_name)
            field_state.setdefault(replay_name, value)


def _canonical_joint_name(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


def _rewrite_legacy_mapping_keys(keyed_values: MutableMapping[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in keyed_values.items():
        rewritten_key = _canonical_joint_name(key, mapping)
        if rewritten_key not in rewritten or rewritten_key == key:
            rewritten[rewritten_key] = value
    return rewritten


def _tensorize_pose_fields(value: Any, tensor_factory: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, MutableMapping):
        return {key: _tensorize_pose_fields(child, tensor_factory, str(key)) for key, child in value.items()}
    if parent_key in {"pos", "rot"} and _is_tensorizable_sequence(value):
        return tensor_factory(value)
    return value


def _is_tensorizable_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, (int, float, np.integer, np.floating)) for item in value)
    return False


def _clamp_robot_default_joint_positions(robot_cfg: Any) -> None:
    defaults = getattr(robot_cfg, "default_joint_positions", None)
    limits = getattr(robot_cfg, "joint_limits", None)
    if not isinstance(defaults, MutableMapping) or not isinstance(limits, MutableMapping):
        return

    clamped = dict(defaults)
    for joint_name, default_value in defaults.items():
        joint_limits = limits.get(joint_name)
        if not isinstance(joint_limits, (list, tuple)) or len(joint_limits) != 2:
            continue
        try:
            value = float(default_value)
            lower = float(joint_limits[0])
            upper = float(joint_limits[1])
        except (TypeError, ValueError):
            continue
        if lower <= value <= upper:
            continue
        clamped[joint_name] = _safe_joint_default(value=value, lower=lower, upper=upper)
    _set_object_attr(robot_cfg, "default_joint_positions", clamped)


def _set_generated_wuji_hand_defaults(robot_cfg: Any) -> None:
    defaults = getattr(robot_cfg, "default_joint_positions", None)
    if not isinstance(defaults, MutableMapping):
        return

    patched_defaults = dict(defaults)
    for joint_name, default_value in defaults.items():
        if not _is_canonical_wuji_hand_joint(joint_name):
            continue
        try:
            value = float(default_value)
        except (TypeError, ValueError):
            continue
        if value == 0.0:
            patched_defaults[joint_name] = 0.1
    _set_object_attr(robot_cfg, "default_joint_positions", patched_defaults)


def _is_canonical_wuji_hand_joint(joint_name: str) -> bool:
    return joint_name.startswith(("left_finger", "right_finger")) and "_joint" in joint_name


def _safe_joint_default(*, value: float, lower: float, upper: float) -> float:
    if upper < lower:
        lower, upper = upper, lower
    if upper - lower > 2.0 * _DEFAULT_JOINT_LIMIT_MARGIN:
        lower += _DEFAULT_JOINT_LIMIT_MARGIN
        upper -= _DEFAULT_JOINT_LIMIT_MARGIN
    return round(min(max(value, lower), upper), 12)


def _set_object_attr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
        return
    except (AttributeError, TypeError, ValueError):
        pass

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        obj_dict[name] = value
        return

    raise AttributeError(f"{type(obj).__name__} does not allow setting {name!r}")


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
    return patch_openarm_wuji_scenario_robot_cfgs(scenario)


def build_decode_scenario(paths: BoxTaskBundlePaths | None = None):
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
    from metasim.scenario.scenario import ScenarioCfg, SimParamCfg

    cardboard_kwargs: dict[str, Any] = {"name": "cardboard_box", "physics": PhysicStateType.RIGIDBODY}
    soda_kwargs: dict[str, Any] = {"name": "feast_soda_can", "physics": PhysicStateType.RIGIDBODY}
    candle_kwargs: dict[str, Any] = {"name": "feast_scented_candle", "physics": PhysicStateType.RIGIDBODY}
    if paths is not None:
        cardboard_kwargs["usd_path"] = str(paths.cardboard_box_usd)
        soda_kwargs["usd_path"] = str(paths.soda_can_usd)
        candle_kwargs["usd_path"] = str(paths.scented_candle_usd)

    scenario = _build_scenario_cfg(
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
            RigidObjCfg(**cardboard_kwargs),
            RigidObjCfg(**soda_kwargs),
            RigidObjCfg(**candle_kwargs),
        ],
        robots=["openarm_wuji"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator="isaacsim",
        renderer="isaacsim",
        headless=True,
        num_envs=1,
    )
    return patch_openarm_wuji_scenario_robot_cfgs(scenario)
