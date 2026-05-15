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


def test_render_frames_materializes_unique_source_frames_before_scene_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = cli.BoxTaskBundlePaths.from_root(_make_bundle(tmp_path))
    source_indices = np.array([5, 2, 5, 9, 2, 9], dtype=np.int32)
    tensor_states = {2: "state-2", 5: "state-5", 9: "state-9"}
    calls: dict[str, object] = {"segments": []}

    def fake_decode(traj_path, requested_indices):
        calls["decode"] = (traj_path, list(requested_indices))
        return tensor_states

    def fake_build_scenario(**kwargs):
        return {"scene": kwargs["scene"], "width": kwargs["width"], "height": kwargs["height"]}

    def fake_render_scene_segment(**kwargs):
        calls["segments"].append(kwargs)

    monkeypatch.setattr(cli, "decode_trajectory_tensor_states", fake_decode)
    monkeypatch.setattr(cli, "build_box_task_scenario", fake_build_scenario)
    monkeypatch.setattr(cli, "render_scene_segment", fake_render_scene_segment)

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
        frame_dir=tmp_path / "frames",
    )

    assert calls["decode"] == (paths.traj_path, [2, 5, 9])
    assert len(calls["segments"]) == 2
    assert calls["segments"][0]["scenario"] == {"scene": "scene-a", "width": 320, "height": 240}
    assert calls["segments"][0]["out_start"] == 0
    assert calls["segments"][0]["out_end"] == 3
    assert calls["segments"][0]["frame_to_src"].tolist() == [5, 2, 5, 9, 2, 9]
    assert calls["segments"][0]["tensor_states_by_src"] is tensor_states
    assert calls["segments"][1]["scenario"]["scene"] == "scene-b"
    assert calls["segments"][1]["out_start"] == 3
    assert calls["segments"][1]["out_end"] == 6


def test_camera_object_lookup_prefers_public_then_private_mapping() -> None:
    handler = SimpleNamespace(camera_objs={"camera0": "public-camera"}, _camera_objs={"camera0": "private-camera"})
    assert cli.camera_object_for(handler, "camera0") == "public-camera"

    private_only = SimpleNamespace(_camera_objs={"camera0": "private-camera"})
    assert cli.camera_object_for(private_only, "camera0") == "private-camera"

    with pytest.raises(KeyError, match="missing"):
        cli.camera_object_for(private_only, "missing")


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
    scene = SimpleNamespace(
        render=SimpleNamespace(engine=None, image_settings=SimpleNamespace(file_format=None, color_mode=None)),
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
    assert scene.view_settings.view_transform == "Standard"
    assert scene.view_settings.exposure == 0.0
    assert scene.view_settings.gamma == 1.0

    cli.configure_blender_for_video(samples=8, device="AUTO")
    assert scene.cycles.device == "GPU"
    assert scene.cycles.samples == 8
    assert cycles_prefs.compute_device_type == "OPTIX"
    assert cycles_prefs.get_devices_calls == 1
    assert all(device.use for device in cycles_prefs.devices)


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
    monkeypatch.setattr(cli, "configure_blender_for_video", lambda *, samples, device: events.append("configure"))
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
    )

    assert created_handlers[0].scenario is scenario
    assert events == ["launch", "configure", "state:state-7", "render:camera0:frame_000000.png", "close"]


def test_render_png_frame_sets_camera_resolution_and_filepath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_calls: list[dict[str, object]] = []
    scene = SimpleNamespace(
        camera=None,
        render=SimpleNamespace(resolution_x=None, resolution_y=None, resolution_percentage=None, filepath=None),
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
    assert render_calls == [{"write_still": True}]


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
