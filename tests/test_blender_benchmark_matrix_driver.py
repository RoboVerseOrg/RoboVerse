from __future__ import annotations

import csv
import shlex
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.benchmark.rendering import blender_render_benchmark as render_benchmark
from scripts.benchmark.rendering import run_blender_benchmark_matrix as matrix_driver


@dataclass
class FakeRobotState:
    joint_pos: object
    joint_pos_target: object
    body_state: object
    body_names: list[str]
    root_state: object = "root"


@dataclass
class FakeTensorState:
    robots: dict[str, FakeRobotState]


def test_run_matrix_records_semantic_child_status(tmp_path, monkeypatch) -> None:
    scene_root = tmp_path / "normal_hand_traj_task1"

    class Completed:
        returncode = 0

    def fake_run(command, stdout, stderr, text, check):
        scene_root.mkdir(parents=True)
        with (scene_root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["command", "status"])
            writer.writeheader()
            writer.writerow({"command": shlex.join(command), "status": "unsupported"})
        stdout.write('{"status": "unsupported"}\n')
        return Completed()

    monkeypatch.setattr(matrix_driver.subprocess, "run", fake_run)

    exit_code = matrix_driver.run_matrix(
        [
            matrix_driver.MatrixRun(
                "normal_hand_traj_task1:denoiser:AUTOMATIC:RGB",
                [
                    "render",
                    "--scene",
                    "normal_hand_traj_task1",
                    "--output-root",
                    str(tmp_path),
                ],
            )
        ],
        tmp_path,
    )

    rows = list(csv.DictReader((tmp_path / "matrix_driver.csv").open(encoding="utf-8")))
    assert exit_code == 0
    assert rows[0]["status"] == "unsupported"


def test_complex_render_uses_recorded_default_initial_pose_state(tmp_path, monkeypatch) -> None:
    import torch

    state_path = tmp_path / "complex_glass_cube_reach" / "states" / "cube_reach_default_initial.pt"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"state")

    applied = []

    class FakeHandler:
        robots = [
            SimpleNamespace(
                name="openarm_bimanual_wuji",
            )
        ]

        def get_states(self, mode):
            raise AssertionError(
                "default initial pose must come from the recorded state, not Blender's stale body state"
            )

        def _apply_tensor_state(self, state):
            applied.append(state)

        def refresh_render(self):  # pragma: no cover - must not be called by the fast path
            raise AssertionError("state application should not trigger an extra pre-render")

    def fake_load(path, weights_only=False):
        assert path == state_path
        assert weights_only is False
        return {
            "states": [
                FakeTensorState(
                    robots={
                        "openarm_bimanual_wuji": FakeRobotState(
                            root_state=torch.tensor([[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                            body_names=["base", "arm"],
                            body_state=torch.tensor([
                                [
                                    [10.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                                    [13.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0],
                                ]
                            ]),
                            joint_pos=torch.tensor([[0.0, -0.7854, 1.5708]]),
                            joint_pos_target=torch.tensor([[0.0, -0.7854, 1.5708]]),
                        )
                    }
                ),
                "target",
            ]
        }

    monkeypatch.setattr(render_benchmark, "_torch_load_state_file", fake_load)

    selected = render_benchmark._apply_recorded_scene_state(
        FakeHandler(),
        root=tmp_path,
        scene="complex_glass_cube_reach",
    )

    assert selected == state_path
    robot_state = applied[0].robots["openarm_bimanual_wuji"]
    assert robot_state.root_state.tolist() == [[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    assert robot_state.body_state[:, :, :3].tolist() == [[[10.0, 2.0, 3.0], [13.0, 5.0, 6.0]]]
    assert robot_state.joint_pos.tolist() == [[pytest.approx(0.0), pytest.approx(-0.7854), pytest.approx(1.5708)]]


def test_complex_render_fails_when_recorded_cube_reach_state_is_missing(tmp_path) -> None:
    try:
        render_benchmark._apply_recorded_scene_state(
            object(),
            root=tmp_path,
            scene="complex_glass_cube_reach",
        )
    except FileNotFoundError as exc:
        assert "refusing to render without the default initial robot pose" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing cube-reach state must not silently render default placement")


def test_glass_quality_matrix_reruns_adaptive_at_low_spp_levels(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="glass_quality",
    )

    adaptive_args = [run.args for run in matrix if run.label.startswith("complex_glass_cube_reach:adaptive:")]
    adaptive_spp = sorted({matrix_driver._arg_value(args, "--samples") for args in adaptive_args})

    assert adaptive_spp == ["16", "64", "8"]
    assert all(matrix_driver._arg_value(args, "--adaptive-sampling") == "on" for args in adaptive_args)


def test_adaptive_low_spp_phase_skips_reference_and_glass_profiles(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="adaptive_low_spp",
    )

    labels = [run.label for run in matrix]

    assert len(labels) == 15
    assert not any(":reference:" in label for label in labels)
    assert not any(":glass:" in label for label in labels)


def test_seed_variance_phase_emits_one_hundred_twenty_rows(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="seed_variance",
    )
    assert len(matrix) == 120


def test_seed_variance_uses_fixed_seed_list(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="seed_variance",
    )
    seeds = sorted({matrix_driver._arg_value(run.args, "--seed") for run in matrix})
    assert seeds == [str(s) for s in range(2026, 2036)]


def test_seed_variance_glass_profiles_use_256_spp(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="seed_variance",
    )
    glass_runs = [run for run in matrix if ":seed_variance:glass:" in run.label]
    assert len(glass_runs) == 50
    assert {matrix_driver._arg_value(run.args, "--samples") for run in glass_runs} == {"256"}


def test_seed_variance_4096spp_uses_denoiser_off(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="seed_variance",
    )
    rows_4096 = [run for run in matrix if matrix_driver._arg_value(run.args, "--samples") == "4096"]
    assert len(rows_4096) == 10
    assert {matrix_driver._arg_value(run.args, "--denoiser") for run in rows_4096} == {"off"}


def test_seed_variance_label_is_setting_plus_seed(tmp_path) -> None:
    matrix = matrix_driver.build_matrix(
        ["complex_glass_cube_reach"],
        tmp_path,
        "OPTIX",
        phase="seed_variance",
    )
    labels = {run.label for run in matrix}
    assert "complex_glass_cube_reach:seed_variance:spp:64spp:seed2026" in labels
    assert "complex_glass_cube_reach:seed_variance:glass:glass_default:seed2031" in labels
    setting_to_seeds: dict[str, set[str]] = {}
    for run in matrix:
        prefix, _, seed_part = run.label.rpartition(":seed")
        setting_to_seeds.setdefault(prefix, set()).add(seed_part)
    for prefix, seed_set in setting_to_seeds.items():
        assert len(seed_set) == 10, prefix
