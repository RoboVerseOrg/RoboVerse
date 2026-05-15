from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.rendering import box_task_replay as replay


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
