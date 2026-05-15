from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
