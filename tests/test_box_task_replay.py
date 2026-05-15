from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark.rendering import box_task_replay as replay


def _make_bundle(tmp_path: Path) -> replay.BoxTaskBundlePaths:
    bundle = tmp_path / "bundle"
    for rel in (
        "assets/traj",
        "assets/local_pack_box/cardboard_box",
        "assets/local_pack_box/feast_soda_can",
        "assets/local_pack_box/feast_scented_candle",
    ):
        (bundle / rel).mkdir(parents=True)
    (bundle / "assets/traj/task.pkl").write_bytes(b"pickle-bytes")
    for rel in (
        "assets/local_pack_box/cardboard_box/cardboard_box.usd",
        "assets/local_pack_box/feast_soda_can/feast_soda_can.usd",
        "assets/local_pack_box/feast_scented_candle/feast_scented_candle.usd",
    ):
        (bundle / rel).write_text("#usda\n", encoding="utf-8")
    return replay.BoxTaskBundlePaths.from_root(bundle, traj_path=bundle / "assets/traj/task.pkl")


def _install_fake_metasim_without_renderer_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    class SimpleCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ScenarioCfg(SimpleCfg):
        def __init__(
            self,
            *,
            objects=None,
            robots=None,
            cameras=None,
            lights=None,
            sim_params=None,
            decimation=None,
            simulator=None,
            headless=None,
            num_envs=None,
            render=None,
        ):
            super().__init__(
                objects=objects,
                robots=robots,
                cameras=cameras,
                lights=lights,
                sim_params=sim_params,
                decimation=decimation,
                simulator=simulator,
                headless=headless,
                num_envs=num_envs,
                render=render,
            )

    modules = {
        "metasim": ModuleType("metasim"),
        "metasim.constants": ModuleType("metasim.constants"),
        "metasim.scenario": ModuleType("metasim.scenario"),
        "metasim.scenario.cameras": ModuleType("metasim.scenario.cameras"),
        "metasim.scenario.lights": ModuleType("metasim.scenario.lights"),
        "metasim.scenario.objects": ModuleType("metasim.scenario.objects"),
        "metasim.scenario.render": ModuleType("metasim.scenario.render"),
        "metasim.scenario.scenario": ModuleType("metasim.scenario.scenario"),
    }
    modules["metasim.constants"].PhysicStateType = SimpleNamespace(RIGIDBODY="rigidbody")
    modules["metasim.scenario.cameras"].PinholeCameraCfg = SimpleCfg
    modules["metasim.scenario.lights"].SphereLightCfg = SimpleCfg
    modules["metasim.scenario.objects"].PrimitiveCubeCfg = SimpleCfg
    modules["metasim.scenario.objects"].RigidObjCfg = SimpleCfg
    modules["metasim.scenario.render"].RenderCfg = SimpleCfg
    modules["metasim.scenario.scenario"].ScenarioCfg = ScenarioCfg
    modules["metasim.scenario.scenario"].SimParamCfg = SimpleCfg

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _cfg_names(items: list[object]) -> list[str]:
    return [str(getattr(item, "name", item)) for item in items]


def _cfg_class_name(item: object) -> str:
    return type(item).__name__.lower()


def _render_mode(render: object) -> str:
    mode = getattr(render, "mode", render)
    return str(mode).lower().replace("_", "").replace("-", "")


def _assert_kujiale_scene_0021(scene: object) -> None:
    scene_name = getattr(scene, "name", scene)
    assert scene_name in ("kujiale_scene_0021", "kujiale_0021")
    scene_usd_path = getattr(scene, "usd_path", None)
    if scene_usd_path is not None:
        assert "021.usda" in str(scene_usd_path)


def test_normalize_scene_token_accepts_supported_forms() -> None:
    assert replay.normalize_scene_token("009") == "kujiale_scene_0009"
    assert replay.normalize_scene_token("0009") == "kujiale_scene_0009"
    assert replay.normalize_scene_token("usd_021") == "kujiale_scene_0021"
    assert replay.normalize_scene_token("kujiale_scene_0031") == "kujiale_scene_0031"
    assert replay.normalize_scene_token("kujiale_scene_0031.usda") == "kujiale_scene_0031"


def test_normalize_scene_token_rejects_unknown_form() -> None:
    with pytest.raises(ValueError, match="Scene token must be 3 or 4 digits"):
        replay.normalize_scene_token("21")
    with pytest.raises(ValueError, match="Invalid scene token"):
        replay.normalize_scene_token("kitchen")


def test_parse_scene_tokens_preserves_order() -> None:
    assert replay.parse_scene_tokens("0021,0022,0031") == [
        "kujiale_scene_0021",
        "kujiale_scene_0022",
        "kujiale_scene_0031",
    ]


def test_compute_output_frames_uses_one_source_when_duration_absent() -> None:
    assert replay.compute_output_frames(src_total_frames=849, fps=30, duration_sec=None, out_frames=None) == 849


def test_compute_output_frames_rejects_ambiguous_length() -> None:
    with pytest.raises(ValueError, match="Use either --out-frames or --duration-sec"):
        replay.compute_output_frames(src_total_frames=849, fps=30, duration_sec=1.0, out_frames=10)


def test_frame_to_source_indices_evenly_samples_source() -> None:
    indices = replay.frame_to_source_indices(src_total_frames=9, out_frames=5)
    assert indices.dtype == np.int32
    assert indices.tolist() == [0, 2, 4, 6, 8]


def test_scene_frame_bounds_split_video_evenly() -> None:
    bounds = replay.scene_frame_bounds(out_frames=10, scene_count=3)
    assert bounds == [(0, 3), (3, 6), (6, 10)]


def test_scene_frame_bounds_rejects_more_scenes_than_frames() -> None:
    with pytest.raises(ValueError, match="scene_count must not exceed out_frames"):
        replay.scene_frame_bounds(out_frames=3, scene_count=4)


def test_finger_joint_map_rewrites_legacy_hand_names() -> None:
    mapping = replay.build_finger_joint_map()
    assert mapping["left_hand_finger1_joint1"] == "left_finger1_joint1"
    assert mapping["right_hand_finger5_joint4"] == "right_finger5_joint4"
    assert len(mapping) == 40


def test_patch_state_for_replay_rewrites_nested_robot_legacy_keys() -> None:
    state = {
        "robots": {
            "openarm_wuji": {
                "dof_pos": {"left_hand_finger1_joint1": 1.25},
                "dof_vel": {"right_hand_finger5_joint4": -0.5},
            }
        }
    }

    patched = replay.patch_state_for_replay(state)

    robot_state = patched["robots"]["openarm_wuji"]
    assert robot_state["dof_pos"] == {"left_finger1_joint1": 1.25}
    assert robot_state["dof_vel"] == {"right_finger5_joint4": -0.5}


def test_patch_state_for_replay_does_not_mutate_input() -> None:
    state = {"robots": {"openarm_wuji": {"dof_pos": {"left_hand_finger1_joint1": 1.25}}}}

    replay.patch_state_for_replay(state)

    assert state == {"robots": {"openarm_wuji": {"dof_pos": {"left_hand_finger1_joint1": 1.25}}}}


def test_patch_state_for_replay_preserves_existing_canonical_value() -> None:
    state = {
        "robots": {
            "openarm_wuji": {
                "dof_pos": {
                    "left_hand_finger1_joint1": "legacy",
                    "left_finger1_joint1": "canonical",
                }
            }
        }
    }

    patched = replay.patch_state_for_replay(state)

    dof_pos = patched["robots"]["openarm_wuji"]["dof_pos"]
    assert dof_pos == {"left_finger1_joint1": "canonical"}


def test_patch_state_for_replay_ignores_non_dict_fields() -> None:
    state = {"robots": {"openarm_wuji": {"dof_pos": ["left_hand_finger1_joint1"], "dof_vel": None}}}

    assert replay.patch_state_for_replay(state) == state


def test_patch_openarm_wuji_robot_cfg_rewrites_legacy_joint_config() -> None:
    legacy_actuator = object()
    canonical_actuator = object()
    robot_cfg = SimpleNamespace(
        name="openarm_wuji",
        joint_names=["openarm_left_joint1", "left_hand_finger1_joint1"],
        left_hand_joint_names=["left_hand_finger1_joint1"],
        right_hand_joint_names=["right_hand_finger5_joint4"],
        ee_joint_names=["right_hand_finger5_joint4"],
        joint_limits={
            "left_hand_finger1_joint1": (-0.5, 1.6),
            "right_hand_finger1_joint1": (0.058, 1.6),
        },
        default_joint_positions={"left_hand_finger1_joint1": 0.1, "right_hand_finger1_joint1": 0.0},
        actuators={"left_hand_finger1_joint1": legacy_actuator, "left_finger1_joint1": canonical_actuator},
        control_type={"right_hand_finger5_joint4": "position"},
        left_ee_body_name="left_hand_palm_link",
        right_ee_body_name="right_hand_palm_link",
        ee_body_name="right_hand_palm_link",
    )

    patched = replay.patch_openarm_wuji_robot_cfg(robot_cfg)

    assert patched is robot_cfg
    assert robot_cfg.joint_names == ["openarm_left_joint1", "left_finger1_joint1"]
    assert robot_cfg.left_hand_joint_names == ["left_finger1_joint1"]
    assert robot_cfg.right_hand_joint_names == ["right_finger5_joint4"]
    assert robot_cfg.ee_joint_names == ["right_finger5_joint4"]
    assert robot_cfg.joint_limits == {"left_finger1_joint1": (-0.5, 1.6), "right_finger1_joint1": (0.058, 1.6)}
    assert robot_cfg.default_joint_positions == {"left_finger1_joint1": 0.1, "right_finger1_joint1": 0.1}
    assert robot_cfg.actuators == {"left_finger1_joint1": canonical_actuator}
    assert robot_cfg.control_type == {"right_finger5_joint4": "position"}
    assert robot_cfg.left_ee_body_name == "left_palm_link"
    assert robot_cfg.right_ee_body_name == "right_palm_link"
    assert robot_cfg.ee_body_name == "right_palm_link"


def test_disable_metasim_forced_exit_on_close_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METASIM_FORCE_EXIT_ON_CLOSE", "1")

    replay.disable_metasim_forced_exit_on_close()

    assert os.environ["METASIM_FORCE_EXIT_ON_CLOSE"] == "0"


def test_tensorize_replay_state_converts_pose_arrays_without_mutating_input() -> None:
    state = {
        "objects": {
            "cardboard_box": {
                "pos": np.array([0.1, 0.2, 0.3]),
                "rot": [1.0, 0.0, 0.0, 0.0],
            }
        },
        "robots": {
            "openarm_wuji": {
                "pos": np.array([0.0, 0.0, 0.0]),
                "rot": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": {"left_finger1_joint1": 0.5},
            }
        },
    }

    def fake_tensor(value):
        if isinstance(value, np.ndarray):
            return ("tensor", tuple(value.tolist()))
        return ("tensor", tuple(value))

    tensorized = replay.tensorize_replay_state(state, tensor_factory=fake_tensor)

    assert tensorized["objects"]["cardboard_box"]["pos"] == ("tensor", (0.1, 0.2, 0.3))
    assert tensorized["objects"]["cardboard_box"]["rot"] == ("tensor", (1.0, 0.0, 0.0, 0.0))
    assert tensorized["robots"]["openarm_wuji"]["pos"] == ("tensor", (0.0, 0.0, 0.0))
    assert tensorized["robots"]["openarm_wuji"]["rot"] == ("tensor", (1.0, 0.0, 0.0, 0.0))
    assert tensorized["robots"]["openarm_wuji"]["dof_pos"] == {"left_finger1_joint1": 0.5}
    assert isinstance(state["objects"]["cardboard_box"]["pos"], np.ndarray)


def test_close_decode_env_without_closing_kit_marks_handler_as_shared() -> None:
    class FakeHandler:
        def __init__(self) -> None:
            self._owns_simulation_app = True

    class FakeEnv:
        def __init__(self) -> None:
            self.handler = FakeHandler()
            self.closed = False

        def close(self) -> None:
            assert self.handler._owns_simulation_app is False
            self.closed = True

    env = FakeEnv()

    replay.close_decode_env_without_closing_kit(env)

    assert env.closed is True


def test_load_v2_episode_states_converts_first_episode_to_v3_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flat_state = {
        "openarm_wuji": {
            "pos": [0.1, 0.2, 0.3],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": {"left_hand_finger1_joint1": 0.4},
        },
        "cardboard_box": {"pos": [0.5, 0.0, 0.4], "rot": [1.0, 0.0, 0.0, 0.0]},
        "feast_soda_can": {"pos": [0.6, 0.1, 0.4], "rot": [1.0, 0.0, 0.0, 0.0]},
    }
    flat_init = {
        "openarm_wuji": {"pos": [0.0, 0.0, 0.0]},
        "cardboard_box": {"pos": [0.4, 0.0, 0.4]},
    }
    payload = {"openarm_wuji": [{"states": [flat_state], "init_state": flat_init}]}
    monkeypatch.setattr(replay, "load_raw_v2_pickle", lambda path: payload)

    states = replay.load_v2_episode_states(tmp_path / "traj.pkl")

    assert states == [
        {
            "robots": {"openarm_wuji": flat_state["openarm_wuji"]},
            "objects": {
                "cardboard_box": flat_state["cardboard_box"],
                "feast_soda_can": flat_state["feast_soda_can"],
            },
        }
    ]
    assert states[0]["robots"]["openarm_wuji"]["dof_pos"]["left_hand_finger1_joint1"] == 0.4

    init_state = replay.load_v2_init_state(tmp_path / "traj.pkl")
    assert init_state == {
        "robots": {"openarm_wuji": flat_init["openarm_wuji"]},
        "objects": {"cardboard_box": flat_init["cardboard_box"]},
    }


def test_load_v2_episode_states_rejects_empty_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"openarm_wuji": [{"states": []}]}
    monkeypatch.setattr(replay, "load_raw_v2_pickle", lambda path: payload)

    with pytest.raises(ValueError, match="states"):
        replay.load_v2_episode_states(tmp_path / "traj.pkl")


def test_load_raw_v2_pickle_removes_temporary_numpy_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alias_name = "temporary.numpy.alias"
    traj_path = tmp_path / "traj.pkl"
    traj_path.write_bytes(pickle.dumps({"ok": True}))

    def fake_install_aliases() -> list[str]:
        sys.modules[alias_name] = ModuleType(alias_name)
        return [alias_name]

    monkeypatch.setattr(replay, "install_numpy_pickle_aliases", fake_install_aliases)

    assert replay.load_raw_v2_pickle(traj_path) == {"ok": True}
    assert alias_name not in sys.modules


def test_patch_isaacsim_render_settings_falls_back_without_replicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSettings:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}

        def set_string(self, key: str, value: str) -> None:
            self.values[key] = value

    class FakeHandler:
        def _load_render_settings(self) -> None:
            raise ModuleNotFoundError("No module named 'omni.replicator'", name="omni.replicator")

    fake_settings = FakeSettings()
    fake_carb = ModuleType("carb")
    fake_carb.settings = SimpleNamespace(get_settings=lambda: fake_settings)
    fake_isaacsim = ModuleType("metasim.sim.isaacsim.isaacsim")
    fake_isaacsim.IsaacsimHandler = FakeHandler
    monkeypatch.setitem(sys.modules, "carb", fake_carb)
    monkeypatch.setitem(sys.modules, "metasim.sim.isaacsim.isaacsim", fake_isaacsim)

    replay._patch_isaacsim_render_settings_for_decode()

    handler = FakeHandler()
    handler.scenario = SimpleNamespace(render=SimpleNamespace(mode="pathtracing"))
    handler._load_render_settings()
    assert fake_settings.values["/rtx/rendermode"] == "PathTracing"


def test_bundle_paths_validate_required_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "assets" / "traj").mkdir(parents=True)
    (bundle / "assets" / "local_pack_box" / "cardboard_box").mkdir(parents=True)
    (bundle / "assets" / "local_pack_box" / "feast_soda_can").mkdir(parents=True)
    (bundle / "assets" / "local_pack_box" / "feast_scented_candle").mkdir(parents=True)
    (bundle / "assets" / "traj" / "task.pkl").write_bytes(b"pickle-bytes")
    for rel in (
        "assets/local_pack_box/cardboard_box/cardboard_box.usd",
        "assets/local_pack_box/feast_soda_can/feast_soda_can.usd",
        "assets/local_pack_box/feast_scented_candle/feast_scented_candle.usd",
    ):
        (bundle / rel).write_text("#usda\n", encoding="utf-8")

    paths = replay.BoxTaskBundlePaths.from_root(bundle, traj_path=bundle / "assets/traj/task.pkl")

    assert paths.bundle_root == bundle.resolve()
    assert paths.traj_path.name == "task.pkl"
    assert paths.cardboard_box_usd.name == "cardboard_box.usd"


def test_bundle_paths_report_missing_traj_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="task3_meshycup_openarm_wuji_20260513_232823_0_v2.pkl"):
        replay.BoxTaskBundlePaths.from_root(tmp_path)


def test_bundle_paths_report_missing_cardboard_after_traj(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "assets" / "traj").mkdir(parents=True)
    (bundle / "assets" / "traj" / "task.pkl").write_bytes(b"pickle-bytes")

    with pytest.raises(FileNotFoundError, match="cardboard_box.usd"):
        replay.BoxTaskBundlePaths.from_root(bundle, traj_path=bundle / "assets/traj/task.pkl")


def test_bundle_paths_reject_directory_where_file_required(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "assets" / "traj" / "task.pkl").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="task.pkl"):
        replay.BoxTaskBundlePaths.from_root(bundle, traj_path=bundle / "assets/traj/task.pkl")


def test_build_box_task_scenario_uses_bundle_assets(tmp_path: Path) -> None:
    paths = _make_bundle(tmp_path)

    scenario = replay.build_box_task_scenario(
        paths=paths,
        simulator="blender",
        scene="kujiale_scene_0021",
        width=640,
        height=480,
        camera_pos=(1.45, -1.0, 0.95),
        camera_look_at=(0.55, 0.0, 0.38),
        head_light_intensity=1200.0,
    )

    objects = {obj.name: obj for obj in scenario.objects}
    assert scenario.simulator == "blender"
    assert scenario.renderer == "blender"
    assert _cfg_names(scenario.robots) == ["openarm_wuji"]
    assert scenario.headless is True
    assert scenario.num_envs == 1
    assert scenario.requested_scene_name == "kujiale_scene_0021"
    _assert_kujiale_scene_0021(scenario.scene)
    camera = scenario.cameras[0]
    assert camera.name == "camera0"
    assert camera.data_types == ["rgb"]
    assert camera.width == 640
    assert camera.height == 480
    assert tuple(camera.pos) == (1.45, -1.0, 0.95)
    assert tuple(camera.look_at) == (0.55, 0.0, 0.38)
    assert "primitivecube" in _cfg_class_name(objects["front_table"])
    assert "rigidobj" in _cfg_class_name(objects["cardboard_box"])
    assert "rigidobj" in _cfg_class_name(objects["feast_soda_can"])
    assert "rigidobj" in _cfg_class_name(objects["feast_scented_candle"])
    assert objects["cardboard_box"].usd_path == str(paths.cardboard_box_usd)
    assert objects["feast_soda_can"].usd_path == str(paths.soda_can_usd)
    assert objects["feast_scented_candle"].usd_path == str(paths.scented_candle_usd)
    assert scenario.lights[0].name == "overhead_light"
    assert "spherelight" in _cfg_class_name(scenario.lights[0])
    assert scenario.lights[0].intensity == 1200.0
    assert scenario.sim_params.dt == 0.005
    assert scenario.decimation == 4
    assert _render_mode(scenario.render) == "pathtracing"


def test_build_decode_scenario_uses_isaacsim() -> None:
    scenario = replay.build_decode_scenario()

    assert scenario.simulator == "isaacsim"
    assert scenario.renderer == "isaacsim"
    assert _cfg_names(scenario.robots) == ["openarm_wuji"]
    assert scenario.headless is True
    assert scenario.num_envs == 1
    assert scenario.sim_params.dt == 0.005
    assert scenario.decimation == 4
    assert set(_cfg_names(scenario.objects)) >= {
        "front_table",
        "cardboard_box",
        "feast_soda_can",
        "feast_scented_candle",
    }


def test_build_decode_scenario_can_use_bundle_object_assets(tmp_path: Path) -> None:
    paths = _make_bundle(tmp_path)
    scenario = replay.build_decode_scenario(paths=paths)
    objects = {obj.name: obj for obj in scenario.objects}

    assert objects["cardboard_box"].usd_path == str(paths.cardboard_box_usd)
    assert objects["feast_soda_can"].usd_path == str(paths.soda_can_usd)
    assert objects["feast_scented_candle"].usd_path == str(paths.scented_candle_usd)


def test_scenario_builders_set_renderer_and_scene_when_constructor_lacks_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_metasim_without_renderer_scene(monkeypatch)
    paths = _make_bundle(tmp_path)

    scenario = replay.build_box_task_scenario(
        paths=paths,
        simulator="blender",
        scene="kujiale_scene_0021",
        width=640,
        height=480,
        camera_pos=(1.45, -1.0, 0.95),
        camera_look_at=(0.55, 0.0, 0.38),
        head_light_intensity=1200.0,
    )
    decode_scenario = replay.build_decode_scenario()

    assert scenario.renderer == "blender"
    assert scenario.requested_scene_name == "kujiale_scene_0021"
    assert scenario.scene == "kujiale_scene_0021"
    assert decode_scenario.renderer == "isaacsim"
