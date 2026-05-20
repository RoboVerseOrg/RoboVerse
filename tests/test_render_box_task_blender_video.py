from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark.rendering import render_box_task_blender_video as cli


def _make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    for rel in (
        "assets/traj",
        "assets/local_pack_box/cardboard_box",
        "assets/local_pack_box/feast_soda_can",
        "assets/local_pack_box/feast_scented_candle",
    ):
        (bundle / rel).mkdir(parents=True)
    (bundle / "assets/traj/task3_meshycup_openarm_wuji_20260513_232823_0_v2.pkl").write_bytes(b"pickle-bytes")
    for rel in (
        "assets/local_pack_box/cardboard_box/cardboard_box.usd",
        "assets/local_pack_box/feast_soda_can/feast_soda_can.usd",
        "assets/local_pack_box/feast_scented_candle/feast_scented_candle.usd",
    ):
        (bundle / rel).write_text("#usda\n", encoding="utf-8")
    return bundle


def test_parse_args_accepts_target_video_options(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    args = cli.parse_args(
        [
            "--bundle-root",
            str(bundle),
            "--scenes",
            "0021,0022",
            "--duration-sec",
            "1.5",
            "--fps",
            "30",
            "--width",
            "320",
            "--height",
            "240",
            "--samples",
            "8",
            "--device",
            "CPU",
            "--out-video",
            str(tmp_path / "out.mp4"),
            "--dry-run",
        ]
    )

    assert args.bundle_root == bundle
    assert args.scenes == "0021,0022"
    assert args.duration_sec == 1.5
    assert args.fps == 30
    assert args.dry_run is True
    assert args.scene_task_offset == "none"
    assert args.scene_import_offset == "auto"
    assert args.clear_task_volume is True
    assert args.camera_pos == (1.45, -1.0, 0.95)
    assert args.camera_look_at == (0.55, 0.0, 0.38)
    assert args.head_light_intensity == 400.0
    assert args.exposure == -2.0


@pytest.mark.parametrize("option", ["--fps", "--width", "--height", "--samples", "--out-frames"])
def test_parse_args_rejects_non_positive_integer_options(tmp_path: Path, option: str) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--out-video",
                str(tmp_path / "out.mp4"),
                option,
                "0",
            ]
        )


def test_script_help_runs_when_executed_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/benchmark/rendering/render_box_task_blender_video.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Render box-task replay video in Blender." in result.stdout


def test_dry_run_prints_render_plan(tmp_path: Path, capsys) -> None:
    bundle = _make_bundle(tmp_path)
    exit_code = cli.main(
        [
            "--bundle-root",
            str(bundle),
            "--scenes",
            "0021,0022",
            "--out-frames",
            "6",
            "--fps",
            "30",
            "--out-video",
            str(tmp_path / "out.mp4"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["scenes"] == ["kujiale_scene_0021", "kujiale_scene_0022"]
    assert payload["out_frames"] == 6
    assert payload["scene_frame_bounds"] == [[0, 3], [3, 6]]
    assert payload["out_video"].endswith("out.mp4")


def test_frame_pattern_uses_zero_padded_pngs(tmp_path: Path) -> None:
    assert cli.frame_path(tmp_path, 12).name == "frame_000012.png"


def test_ffmpeg_command_uses_yuv420p_for_compatibility(tmp_path: Path) -> None:
    command = cli.ffmpeg_command(
        ffmpeg_bin="/usr/bin/ffmpeg",
        frame_dir=tmp_path / "frames",
        fps=30,
        out_video=tmp_path / "video.mp4",
    )

    assert command == [
        "/usr/bin/ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(tmp_path / "frames" / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(tmp_path / "video.mp4"),
    ]


def test_work_dir_defaults_under_bundle_root(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    work_dir = cli.select_work_dir(bundle_root=bundle, tmp_dir=None, pid=123)
    assert work_dir == bundle / ".tmp_box_task_blender" / "render_work_123"


def test_render_plan_calls_frame_renderer_once_per_output_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bundle = _make_bundle(tmp_path)
    calls: dict[str, object] = {}

    def fake_render_frames(**kwargs):
        calls["render_frames"] = kwargs
        kwargs["frame_dir"].mkdir(parents=True, exist_ok=True)

    def fake_assemble_video(**kwargs):
        calls["assemble_video"] = kwargs
        kwargs["out_video"].write_bytes(b"fake mp4")

    monkeypatch.setattr(cli, "render_frames", fake_render_frames)
    monkeypatch.setattr(cli, "assemble_video", fake_assemble_video)

    exit_code = cli.main(
        [
            "--bundle-root",
            str(bundle),
            "--scenes",
            "0021,0022",
            "--out-frames",
            "4",
            "--fps",
            "24",
            "--out-video",
            str(tmp_path / "out.mp4"),
        ]
    )

    assert exit_code == 0
    render_kwargs = calls["render_frames"]
    assert render_kwargs["scenes"] == ["kujiale_scene_0021", "kujiale_scene_0022"]
    assert render_kwargs["frame_to_src"].tolist() == [0, 283, 565, 848]
    assert len(render_kwargs["frame_to_src"]) == 4
    assert calls["assemble_video"]["frame_dir"] == render_kwargs["frame_dir"]
    assert calls["assemble_video"]["fps"] == 24
    assert calls["assemble_video"]["out_video"] == (tmp_path / "out.mp4").resolve()
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "ok", "out_video": str((tmp_path / "out.mp4").resolve()), "frames": 4}
    assert (tmp_path / "out.mp4").read_bytes() == b"fake mp4"


def test_render_frames_decodes_unique_source_frames_in_subprocess_before_scene_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = cli.BoxTaskBundlePaths.from_root(_make_bundle(tmp_path))
    source_indices = np.array([5, 2, 5, 9, 2, 9], dtype=np.int32)
    tensor_states = {2: "state-2", 5: "state-5", 9: "state-9"}
    calls: dict[str, object] = {"segments": []}

    def fake_decode_subprocess(**kwargs):
        calls["decode_subprocess"] = kwargs

    def fake_load_decoded(path):
        calls["load_decoded"] = path
        return tensor_states

    def fake_build_scenario(**kwargs):
        return {"scene": kwargs["scene"], "width": kwargs["width"], "height": kwargs["height"]}

    def fake_render_scene_segment(**kwargs):
        calls["segments"].append(kwargs)

    monkeypatch.setattr(cli, "decode_trajectory_tensor_states_subprocess", fake_decode_subprocess)
    monkeypatch.setattr(cli, "load_decoded_tensor_states", fake_load_decoded)
    monkeypatch.setattr(cli, "build_box_task_scenario", fake_build_scenario)
    monkeypatch.setattr(cli, "render_scene_segment", fake_render_scene_segment)

    frame_dir = tmp_path / "work" / "frames"
    cli.render_frames(
        args=SimpleNamespace(
            width=320,
            height=240,
            samples=8,
            device="CPU",
            camera_pos=(1.0, 2.0, 3.0),
            camera_look_at=(0.0, 0.0, 0.0),
            head_light_intensity=100.0,
        ),
        paths=paths,
        scenes=["scene-a", "scene-b"],
        frame_to_src=source_indices,
        frame_dir=frame_dir,
    )

    assert calls["decode_subprocess"]["paths"] is paths
    assert calls["decode_subprocess"]["source_indices"] == [2, 5, 9]
    assert calls["decode_subprocess"]["state_path"] == frame_dir.parent / "decoded_tensor_states.pt"
    assert calls["load_decoded"] == frame_dir.parent / "decoded_tensor_states.pt"
    assert len(calls["segments"]) == 2
    assert calls["segments"][0]["scenario"] == {"scene": "scene-a", "width": 320, "height": 240}
    assert calls["segments"][0]["out_start"] == 0
    assert calls["segments"][0]["out_end"] == 3
    assert calls["segments"][0]["frame_to_src"].tolist() == [5, 2, 5, 9, 2, 9]
    assert calls["segments"][0]["tensor_states_by_src"] is tensor_states
    assert calls["segments"][1]["scenario"]["scene"] == "scene-b"
    assert calls["segments"][1]["out_start"] == 3
    assert calls["segments"][1]["out_end"] == 6


def test_decode_subprocess_command_passes_bundle_traj_indices_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = cli.BoxTaskBundlePaths.from_root(_make_bundle(tmp_path))
    state_path = tmp_path / "states.pt"
    calls: dict[str, object] = {}

    def fake_run(command, *, text, capture_output, check):
        calls["command"] = command
        calls["run_kwargs"] = {"text": text, "capture_output": capture_output, "check": check}
        return SimpleNamespace(returncode=0, stdout="decoded", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.decode_trajectory_tensor_states_subprocess(
        paths=paths,
        source_indices=[5, 2, 5],
        state_path=state_path,
        python_bin="/python",
        script_path=Path("/repo/render.py"),
    )

    assert calls["command"] == [
        "/python",
        "/repo/render.py",
        "--bundle-root",
        str(paths.bundle_root),
        "--traj-path",
        str(paths.traj_path),
        "--decode-source-indices-json",
        "[5, 2, 5]",
        "--decode-states-out",
        str(state_path),
    ]
    assert calls["run_kwargs"] == {"text": True, "capture_output": True, "check": False}


def test_decode_subprocess_reports_worker_output_tail_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = cli.BoxTaskBundlePaths.from_root(_make_bundle(tmp_path))

    def fake_run(command, *, text, capture_output, check):
        return SimpleNamespace(returncode=7, stdout="stdout detail", stderr="stderr detail")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="decode worker failed with exit code 7"):
        cli.decode_trajectory_tensor_states_subprocess(
            paths=paths,
            source_indices=[1],
            state_path=tmp_path / "states.pt",
            python_bin="/python",
            script_path=Path("/repo/render.py"),
        )


def test_decode_worker_mode_writes_decoded_states_without_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    bundle = _make_bundle(tmp_path)
    calls: dict[str, object] = {}

    def fake_decode(traj_path, source_indices, bundle_paths=None):
        calls["decode"] = (traj_path, list(source_indices), bundle_paths)
        return {2: "state-2", 5: "state-5"}

    def fake_save(states, path):
        calls["save"] = (states, path)

    def fail_render_video(args):
        raise AssertionError("decode worker mode must not render")

    monkeypatch.setattr(cli, "decode_trajectory_tensor_states", fake_decode)
    monkeypatch.setattr(cli, "save_decoded_tensor_states", fake_save)
    monkeypatch.setattr(cli, "render_video", fail_render_video)

    state_path = tmp_path / "decoded.pt"
    exit_code = cli.main(
        [
            "--bundle-root",
            str(bundle),
            "--decode-source-indices-json",
            "[5, 2, 5]",
            "--decode-states-out",
            str(state_path),
        ]
    )

    assert exit_code == 0
    decoded_paths = calls["decode"][2]
    assert calls["decode"] == (decoded_paths.traj_path, [5, 2, 5], decoded_paths)
    assert calls["save"] == ({2: "state-2", 5: "state-5"}, state_path)
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "decoded", "states": str(state_path), "source_frames": 3}


def test_camera_object_lookup_prefers_public_then_private_mapping() -> None:
    handler = SimpleNamespace(camera_objs={"camera0": "public-camera"}, _camera_objs={"camera0": "private-camera"})
    assert cli.camera_object_for(handler, "camera0") == "public-camera"

    private_only = SimpleNamespace(_camera_objs={"camera0": "private-camera"})
    assert cli.camera_object_for(private_only, "camera0") == "private-camera"

    with pytest.raises(KeyError, match="missing"):
        cli.camera_object_for(private_only, "missing")


def test_scene_task_offset_auto_uses_negative_scene_default_position() -> None:
    scenario = SimpleNamespace(scene=SimpleNamespace(default_position=(-5.8, 1.8, 0.0)))

    assert cli.resolve_scene_task_offset("auto", scenario) == (5.8, -1.8, -0.0)
    assert cli.resolve_scene_task_offset("none", scenario) == (0.0, 0.0, 0.0)
    assert cli.resolve_scene_task_offset("1.0,2.0,3.0", scenario) == (1.0, 2.0, 3.0)


def test_scene_import_offset_auto_uses_scene_default_position() -> None:
    scenario = SimpleNamespace(scene=SimpleNamespace(default_position=(-5.8, 1.8, 0.0)))

    assert cli.resolve_scene_import_offset("auto", scenario) == (-5.8, 1.8, 0.0)
    assert cli.resolve_scene_import_offset("none", scenario) == (0.0, 0.0, 0.0)
    assert cli.resolve_scene_import_offset("1.0,2.0,3.0", scenario) == (1.0, 2.0, 3.0)


def test_apply_scene_task_offset_moves_cameras_and_lights() -> None:
    camera = SimpleNamespace(pos=(1.45, -1.0, 0.95), look_at=(0.55, 0.0, 0.38))
    light = SimpleNamespace(pos=(0.55, 0.0, 1.45))
    scenario = SimpleNamespace(cameras=[camera], lights=[light])

    cli.apply_scene_task_offset_to_scenario(scenario, (5.8, -1.8, 0.0))

    assert camera.pos == (7.25, -2.8, 0.95)
    assert camera.look_at == (6.35, -1.8, 0.38)
    assert light.pos == (6.35, -1.8, 1.45)


def test_apply_scene_import_offset_moves_blender_scene_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    update_calls: list[str] = []
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(view_layer=SimpleNamespace(update=lambda: update_calls.append("update")))
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    root = SimpleNamespace(location=[10.0, -2.0, 0.0])
    handler = SimpleNamespace(_scene_objs=[root])

    cli.apply_scene_import_offset_to_handler(handler, (-5.8, 1.8, 0.0))

    assert root.location == pytest.approx([4.2, -0.2, 0.0])
    assert update_calls == ["update"]


def test_clear_task_volume_hides_intersecting_scene_meshes_but_keeps_floor() -> None:
    class IdentityMatrix:
        def __matmul__(self, point):
            return point

    def obj(name: str, bounds, obj_type: str = "MESH"):
        return SimpleNamespace(
            name=name,
            type=obj_type,
            bound_box=[
                (bounds[0][0], bounds[0][1], bounds[0][2]),
                (bounds[0][0], bounds[0][1], bounds[1][2]),
                (bounds[0][0], bounds[1][1], bounds[0][2]),
                (bounds[0][0], bounds[1][1], bounds[1][2]),
                (bounds[1][0], bounds[0][1], bounds[0][2]),
                (bounds[1][0], bounds[0][1], bounds[1][2]),
                (bounds[1][0], bounds[1][1], bounds[0][2]),
                (bounds[1][0], bounds[1][1], bounds[1][2]),
            ],
            matrix_world=IdentityMatrix(),
            children=[],
            hide_render=False,
            hide_viewport=False,
        )

    wall = obj("wall", ((-0.1, -1.0, 0.0), (0.1, 1.0, 2.0)))
    floor = obj("floor", ((-2.0, -2.0, -0.02), (2.0, 2.0, 0.02)))
    far_cabinet = obj("cabinet", ((5.0, 5.0, 0.0), (6.0, 6.0, 1.0)))
    handler = SimpleNamespace(_scene_objs=[SimpleNamespace(children=[wall, floor, far_cabinet])])

    status = cli.clear_task_volume_for_replay(handler, center=(0.0, 0.0, 0.0), half_extents=(1.0, 1.0, 2.0))

    assert status == {"status": "applied", "hidden_count": 1}
    assert wall.hide_render is True
    assert wall.hide_viewport is True
    assert floor.hide_render is False
    assert far_cabinet.hide_render is False


def test_translate_tensor_state_offsets_root_and_body_positions_without_mutating_source() -> None:
    state = SimpleNamespace(
        objects={
            "box": SimpleNamespace(
                root_state=np.array([[0.55, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                body_state=None,
            )
        },
        robots={
            "robot": SimpleNamespace(
                root_state=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                body_state=np.array([[[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
            )
        },
        cameras={},
    )

    shifted = cli.translate_tensor_state(state, (5.8, -1.8, 0.0))

    np.testing.assert_allclose(shifted.objects["box"].root_state[0, :3], [6.35, -1.8, 0.42])
    np.testing.assert_allclose(shifted.robots["robot"].root_state[0, :3], [5.8, -1.8, 0.0])
    np.testing.assert_allclose(shifted.robots["robot"].body_state[0, 0, :3], [5.9, -1.6, 0.3])
    np.testing.assert_allclose(state.objects["box"].root_state[0, :3], [0.55, 0.0, 0.42])


def test_configure_blender_for_video_sets_cpu_and_gpu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDevice:
        def __init__(self, device_type: str):
            self.type = device_type
            self.use = False

    class FakePreferences:
        def __init__(self):
            self.compute_device_type = None
            self.devices = [FakeDevice("CPU"), FakeDevice("OPTIX")]
            self.get_devices_calls = 0

        def get_devices(self):
            self.get_devices_calls += 1
            return self.devices

    cycles_prefs = FakePreferences()
    image_settings = SimpleNamespace(file_format=None, color_mode=None)
    scene = SimpleNamespace(
        render=SimpleNamespace(engine=None, image_settings=image_settings),
        cycles=SimpleNamespace(samples=None, device=None),
        view_settings=SimpleNamespace(view_transform=None, exposure=None, gamma=None),
    )
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(scene=scene, preferences=SimpleNamespace(addons={"cycles": SimpleNamespace(preferences=cycles_prefs)}))
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    cli.configure_blender_for_video(samples=4, device="CPU")
    assert scene.render.engine == "CYCLES"
    assert scene.cycles.samples == 4
    assert scene.cycles.device == "CPU"
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.render.image_settings.color_mode == "RGB"
    assert scene.render.use_compositing is False
    assert scene.render.use_sequencer is False
    assert scene.view_settings.view_transform == "Standard"
    assert scene.view_settings.exposure == -2.0
    assert scene.view_settings.gamma == 1.0
    assert scene.cycles.use_denoising is True
    assert scene.cycles.denoiser == "OPTIX"
    assert scene.cycles.denoising_input_passes == "RGB_ALBEDO_NORMAL"
    assert scene.cycles.denoising_prefilter == "ACCURATE"
    assert scene.cycles.denoising_quality == "HIGH"
    assert scene.cycles.use_adaptive_sampling is False
    assert scene.cycles.max_bounces == 10
    assert scene.cycles.diffuse_bounces == 4
    assert scene.cycles.glossy_bounces == 4
    assert scene.cycles.transmission_bounces == 10
    assert scene.cycles.transparent_max_bounces == 10
    assert scene.cycles.caustics_reflective is True
    assert scene.cycles.caustics_refractive is True
    assert scene.cycles.sample_clamp_indirect == 10.0
    assert scene.cycles.blur_glossy == 1.0

    cli.configure_blender_for_video(samples=8, device="AUTO", exposure=-1.25)
    assert scene.cycles.device == "GPU"
    assert scene.cycles.samples == 8
    assert scene.view_settings.exposure == -1.25
    assert cycles_prefs.compute_device_type == "OPTIX"
    assert cycles_prefs.get_devices_calls == 1
    assert all(device.use for device in cycles_prefs.devices)


def test_repair_imported_scene_materials_uses_metasim_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    scene_objects = [object(), object()]
    update_calls: list[str] = []
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(
        scene=SimpleNamespace(objects=scene_objects),
        view_layer=SimpleNamespace(update=lambda: update_calls.append("update")),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    repair_calls: list[tuple[list[object], dict[str, tuple[float, float, float, float]]]] = []
    blender_module = ModuleType("metasim.sim.blender.blender")
    blender_module._repair_imported_materials = lambda objects, overrides: repair_calls.append((objects, overrides))
    monkeypatch.setitem(sys.modules, "metasim", ModuleType("metasim"))
    monkeypatch.setitem(sys.modules, "metasim.sim", ModuleType("metasim.sim"))
    monkeypatch.setitem(sys.modules, "metasim.sim.blender", ModuleType("metasim.sim.blender"))
    monkeypatch.setitem(sys.modules, "metasim.sim.blender.blender", blender_module)

    status = cli.repair_imported_scene_materials()

    assert status["status"] == "applied"
    assert status["object_count"] == 2
    assert status["override_count"] > 0
    assert repair_calls == [(scene_objects, cli.KUJIALE_MATERIAL_OVERRIDES)]
    assert "Wall" in repair_calls[0][1]
    assert update_calls == ["update"]


def test_local_bundle_render_runtime_disables_hf_download_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    hf_util = ModuleType("metasim.utils.hf_util")
    hf_util.check_and_download_single = lambda filepath: calls.append(f"single:{filepath}")
    hf_util.check_and_download_recursive = lambda filepaths, n_processes=16: calls.append("recursive")
    monkeypatch.setitem(sys.modules, "metasim", ModuleType("metasim"))
    monkeypatch.setitem(sys.modules, "metasim.utils", ModuleType("metasim.utils"))
    monkeypatch.setitem(sys.modules, "metasim.utils.hf_util", hf_util)

    original_single = hf_util.check_and_download_single
    original_recursive = hf_util.check_and_download_recursive
    with cli.local_bundle_render_runtime():
        hf_util.check_and_download_single("missing.png")
        hf_util.check_and_download_recursive(["missing.png"])

    assert calls == []
    assert hf_util.check_and_download_single is original_single
    assert hf_util.check_and_download_recursive is original_recursive


def test_render_scene_segment_launches_handler_before_rendering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    created_handlers: list[object] = []

    class FakeHandler:
        def __init__(self, scenario):
            self.scenario = scenario
            created_handlers.append(self)

        def launch(self):
            events.append("launch")

        def close(self):
            events.append("close")

    blender_module = ModuleType("metasim.sim.blender.blender")
    blender_module.BlenderHandler = FakeHandler
    monkeypatch.setitem(sys.modules, "metasim", ModuleType("metasim"))
    monkeypatch.setitem(sys.modules, "metasim.sim", ModuleType("metasim.sim"))
    monkeypatch.setitem(sys.modules, "metasim.sim.blender", ModuleType("metasim.sim.blender"))
    monkeypatch.setitem(sys.modules, "metasim.sim.blender.blender", blender_module)
    monkeypatch.setattr(
        cli,
        "apply_scene_import_offset_to_handler",
        lambda handler, offset: events.append(f"scene_offset:{offset}"),
    )
    monkeypatch.setattr(cli, "configure_blender_for_video", lambda *, samples, device, exposure=-2.0: events.append("configure"))
    monkeypatch.setattr(
        cli,
        "repair_imported_scene_materials",
        lambda: events.append("repair") or {"status": "applied"},
    )
    monkeypatch.setattr(
        cli,
        "clear_task_volume_for_replay",
        lambda handler, center: events.append(f"clear:{center}") or {"status": "applied", "hidden_count": 2},
    )
    monkeypatch.setattr(cli, "apply_state_to_handler", lambda handler, state: events.append(f"state:{state}"))
    monkeypatch.setattr(
        cli,
        "render_png_frame",
        lambda *, handler, camera, path: events.append(f"render:{camera.name}:{path.name}"),
    )

    camera = SimpleNamespace(name="camera0", width=320, height=240)
    scenario = SimpleNamespace(cameras=[camera])
    cli.render_scene_segment(
        scenario=scenario,
        tensor_states_by_src={7: "state-7"},
        frame_to_src=[7],
        out_start=0,
        out_end=1,
        frame_dir=tmp_path,
        samples=8,
        device="CPU",
        scene_import_offset=(1.0, 2.0, 3.0),
    )

    assert created_handlers[0].scenario is scenario
    assert scenario._box_task_room_material_repair == {"status": "applied"}
    assert scenario._box_task_volume_clearance == {"status": "applied", "hidden_count": 2}
    assert events == [
        "launch",
        "scene_offset:(1.0, 2.0, 3.0)",
        "clear:(0.0, 0.0, 0.0)",
        "repair",
        "configure",
        "state:state-7",
        "render:camera0:frame_000000.png",
        "close",
    ]


def test_render_png_frame_sets_camera_resolution_and_filepath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_calls: list[dict[str, object]] = []
    scene = SimpleNamespace(
        camera=None,
        render=SimpleNamespace(
            resolution_x=None,
            resolution_y=None,
            resolution_percentage=None,
            filepath=None,
            image_settings=SimpleNamespace(file_format="BMP", color_mode="RGB"),
        ),
    )
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(scene=scene)
    bpy.ops = SimpleNamespace(render=SimpleNamespace(render=lambda **kwargs: render_calls.append(kwargs)))
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    camera_obj = object()
    handler = SimpleNamespace(camera_objs={"camera0": camera_obj})
    camera = SimpleNamespace(name="camera0", width=320, height=240)
    out_path = tmp_path / "frame.png"

    cli.render_png_frame(handler=handler, camera=camera, path=out_path)

    assert scene.camera is camera_obj
    assert scene.render.resolution_x == 320
    assert scene.render.resolution_y == 240
    assert scene.render.resolution_percentage == 100
    assert scene.render.filepath == str(out_path)
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.render.image_settings.color_mode == "RGB"
    assert render_calls == [{"write_still": True}]


def test_apply_state_to_handler_uses_private_tensor_path_without_readback_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update_calls: list[str] = []
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(view_layer=SimpleNamespace(update=lambda: update_calls.append("update")))
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    class FastTensorHandler:
        def __init__(self):
            self.applied = []
            self.invalidated = 0
            self._last_tensor_state = None
            self._render_dirty = False

        def _invalidate_state_caches(self):
            self.invalidated += 1

        def _apply_tensor_state(self, state):
            self.applied.append(state)

        def set_states(self, state):
            raise AssertionError("fast TensorState path must not call set_states")

        def refresh_render(self):
            raise AssertionError("fast TensorState path must not render readback")

    state = SimpleNamespace(objects={}, robots={}, cameras={})
    handler = FastTensorHandler()

    cli.apply_state_to_handler(handler, state)

    assert handler.applied == [state]
    assert handler.invalidated == 1
    assert handler._last_tensor_state is state
    assert handler._render_dirty is True
    assert update_calls == ["update"]


def test_apply_state_to_handler_uses_blender_readback_format_for_handler_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = SimpleNamespace(render=SimpleNamespace(image_settings=SimpleNamespace(file_format="PNG", color_mode="RGBA")))
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(scene=scene)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    calls: list[tuple[object, str, str]] = []

    class FakeRefreshingHandler:
        set_states_refreshes = True

        def set_states(self, state):
            calls.append(
                (
                    state,
                    scene.render.image_settings.file_format,
                    scene.render.image_settings.color_mode,
                )
            )

        def refresh_render(self):
            raise AssertionError("set_states already refreshes")

    cli.apply_state_to_handler(FakeRefreshingHandler(), "tensor-state")

    assert calls == [("tensor-state", "BMP", "RGB")]


def test_apply_state_to_handler_passes_tensor_state_directly_falls_back_to_list_and_refreshes() -> None:
    class FakeHandler:
        def __init__(self):
            self.calls = []
            self.refresh_calls = 0

        def set_states(self, state):
            if isinstance(state, list):
                raise AssertionError("TensorState must not be wrapped in a list")
            self.calls.append(state)

        def refresh_render(self):
            self.refresh_calls += 1

    handler = FakeHandler()
    cli.apply_state_to_handler(handler, "tensor-state")
    assert handler.calls == ["tensor-state"]
    assert handler.refresh_calls == 1

    class FallbackHandler:
        def __init__(self):
            self.calls = []
            self.refresh_calls = 0

        def set_states(self, state):
            self.calls.append(state)
            if state == "tensor-state":
                raise TypeError("cannot apply tensor")

        def refresh_render(self):
            self.refresh_calls += 1

    fallback_handler = FallbackHandler()
    cli.apply_state_to_handler(fallback_handler, "tensor-state")
    assert fallback_handler.calls == ["tensor-state", ["tensor-state"]]
    assert fallback_handler.refresh_calls == 1
