"""The converters refuse a demo they cannot convert, before reading any frame, with a reason."""

from __future__ import annotations

import pytest

from roboverse_learn.il.utils.demo_metadata import check_demo_metadata

GOOD = {"joint_qpos": [[0.0] * 9] * 3, "joint_qpos_target": [[0.0] * 9] * 3, "robot_root_state": [[0.0] * 13] * 3}


def test_joint_pos_spaces_accept_a_writer_produced_demo():
    check_demo_metadata(GOOD, observation_space="joint_pos", action_space="joint_pos", demo_dir="demo_0000")


def test_ee_spaces_name_the_keys_the_writer_does_not_produce():
    with pytest.raises(ValueError, match=r"needs .*robot_ee_state.* missing \['robot_ee_state'\]"):
        check_demo_metadata(GOOD, observation_space="ee", action_space="joint_pos", demo_dir="demo_0000")
    with pytest.raises(ValueError, match="robot_ee_state_target"):
        check_demo_metadata(
            {**GOOD, "robot_ee_state": []}, observation_space="joint_pos", action_space="ee", demo_dir="d"
        )


def test_null_or_misaligned_targets_are_refused_up_front():
    with pytest.raises(ValueError, match="missing or null"):
        check_demo_metadata(
            {**GOOD, "joint_qpos_target": None}, observation_space="joint_pos", action_space="joint_pos", demo_dir="d"
        )
    with pytest.raises(ValueError, match="missing or null"):
        check_demo_metadata(
            {**GOOD, "joint_qpos_target": [[0.0] * 9, None, [0.0] * 9]},
            observation_space="joint_pos",
            action_space="joint_pos",
            demo_dir="d",
        )
    with pytest.raises(ValueError, match="2 joint targets for 3 states"):
        check_demo_metadata(
            {**GOOD, "joint_qpos_target": [[0.0] * 9] * 2},
            observation_space="joint_pos",
            action_space="joint_pos",
            demo_dir="d",
        )
    with pytest.raises(ValueError, match="not a RoboVerse demo"):
        check_demo_metadata({}, observation_space="joint_pos", action_space="joint_pos", demo_dir="d")


def test_the_lerobot_converters_key_names_are_honoured():
    meta = {"obs": [[0.0]] * 2, "act": [[0.0]] * 2}
    check_demo_metadata(meta, demo_dir="d", state_key="obs", action_key="act")
    with pytest.raises(ValueError, match="'act' is missing or null"):
        check_demo_metadata({"obs": [[0.0]] * 2}, demo_dir="d", state_key="obs", action_key="act")
