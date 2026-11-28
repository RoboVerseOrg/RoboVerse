"""The robot self-check's pure parts (config, kinematics) on a toy MJCF arm, and the CLI wiring; no simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.utils.robot_check import RobotReport, check_assets, check_config, check_kinematics

pytestmark = pytest.mark.general

_ARM = """
<mujoco model="toy">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="hand" pos="0.3 0 0">
        <joint name="j2" axis="0 1 0" range="-3 3"/>
        <geom type="box" size="0.02 0.02 0.02"/>
        <body name="finger" pos="0 0.03 0">
          <joint name="f" type="slide" axis="0 1 0" range="0 0.04"/>
          <geom type="box" size="0.01 0.005 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _cfg(tmp_path, **over):
    path = tmp_path / "toy.xml"
    path.write_text(_ARM)
    base = dict(
        name="toy",
        mjcf_path=str(path),
        usd_path=None,
        ee_body_name="hand",
        ee_joint_names=["f"],
        actuators={"j1": None, "j2": None, "f": None},
        joint_limits={"j1": (-3.0, 3.0), "j2": (-3.0, 3.0), "f": (0.0, 0.04)},
        default_joint_positions={"j1": 0.0, "j2": 0.5, "f": 0.04},
        fix_base_link=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_config_check_reports_each_problem_by_joint(tmp_path):
    assert check_config(_cfg(tmp_path)).ok
    row = check_config(_cfg(tmp_path, default_joint_positions={"j1": 9.0, "j2": 0.5}))
    assert not row.ok and "no default position for ['f']" in row.detail and "outside limits for ['j1']" in row.detail
    assert not check_config(_cfg(tmp_path, actuators={})).ok


def test_assets_check_sees_local_files_and_names_the_missing_backend(tmp_path):
    assert check_assets(_cfg(tmp_path), "mujoco").ok
    row = check_assets(_cfg(tmp_path), "isaacsim")
    assert not row.ok and "usd_path" in row.detail, "no USD declared is a failed Isaac check, not a quiet skip"


def test_kinematics_check_loads_the_mjcf_and_finds_the_tool(tmp_path):
    row = check_kinematics(_cfg(tmp_path))
    assert row.ok and "ee body 'hand' found" in row.detail and "2 arm joints, 1 gripper joints" in row.detail
    bad = check_kinematics(_cfg(tmp_path, ee_body_name="nope"))
    assert not bad.ok and "no body 'nope'" in bad.detail


def test_report_table_and_verdict(tmp_path):
    report = RobotReport(
        robot="toy", simulator="mujoco", rows=[check_config(_cfg(tmp_path)), check_kinematics(_cfg(tmp_path))]
    )
    assert report.ok and "verdict: pass" in report.table() and report.to_dict()["ok"] is True


def test_cli_has_robot_check():
    import metasim.__main__ as cli

    parser_args = cli.main.__wrapped__ if hasattr(cli.main, "__wrapped__") else None
    assert parser_args is None  # main builds its parser inline; exercising it:
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    cli._add_robot(sub)
    ns = p.parse_args(["robot", "check", "franka", "--no-hold", "--json"])
    assert ns.robot == ["franka"] and ns.no_hold and ns.func is cli._robot_check


def test_limits_check_compares_the_config_with_the_mjcf_ranges(tmp_path):
    from metasim.utils.robot_check import check_limits

    row = check_limits(_cfg(tmp_path))
    assert row.ok and "3 limited joints match" in row.detail
    bad = check_limits(_cfg(tmp_path, joint_limits={"j1": (-3.0, 3.0), "j2": (-1.0, 1.0), "f": (0.0, 0.04)}))
    assert not bad.ok and "j2 cfg [-1.000, 1.000] vs mjcf [-3.000, 3.000]" in bad.detail
    assert check_limits(_cfg(tmp_path, mjcf_path=None)).ok, "no MJCF: not judged"


def test_kinematics_and_hold_rows_do_not_judge_what_is_not_an_arm(tmp_path):
    from metasim.utils.robot_check import check_kinematics

    row = check_kinematics(_cfg(tmp_path, ee_body_name=None))
    assert row.ok and "not an arm" in row.detail
    none = check_kinematics(_cfg(tmp_path, mjcf_path=None))
    assert not none.ok and "no mjcf_path" in none.detail


_ACTUATED_ARM = """
<mujoco model="toy_actuated">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="hand" pos="0.3 0 0">
        <joint name="j2" axis="0 1 0" range="-3 3"/>
        <geom type="box" size="0.02 0.02 0.02"/>
        <site name="grasp_site" pos="0 0 0.05"/>
        <body name="finger_l" pos="0 0.03 0">
          <joint name="f_l" type="slide" axis="0 1 0" range="0 0.04"/>
          <geom type="box" size="0.01 0.005 0.02"/>
        </body>
        <body name="finger_r" pos="0 -0.03 0">
          <joint name="f_r" type="slide" axis="0 1 0" range="-0.04 0"/>
          <geom type="box" size="0.01 0.005 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="f_l" joint2="f_r" polycoef="0 -1 0 0 0"/>
  </equality>
  <actuator>
    <position name="j1" joint="j1" kp="50" kv="5" forcerange="-20 20"/>
    <position name="elbow" joint="j2" kp="30" kv="3"/>
    <position name="f_l" joint="f_l" kp="100" kv="10"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.1 0.7 0.04 -0.04"/>
  </keyframe>
</mujoco>
"""


def test_robot_kit_reads_joints_gains_defaults_and_the_tool_from_the_mjcf(tmp_path):
    from metasim.utils.robot_kit import inspect_mjcf

    path = tmp_path / "toy_actuated.xml"
    path.write_text(_ACTUATED_ARM)
    spec = inspect_mjcf(str(path))
    by_name = {j.name: j for j in spec.joints}
    assert [j.name for j in spec.joints] == ["j1", "j2", "f_l", "f_r"] and not spec.floating_base
    assert (by_name["j1"].stiffness, by_name["j1"].damping, by_name["j1"].effort_limit) == (50.0, 5.0, 20.0)
    assert by_name["j2"].actuator == "elbow" and any("'elbow' drives joint 'j2'" in n for n in spec.notes)
    assert by_name["f_l"].is_gripper and by_name["f_r"].is_gripper
    assert by_name["f_r"].passive and not by_name["f_l"].passive, "the equality-coupled finger is passive"
    assert [j.default for j in spec.joints] == [
        pytest.approx(0.1),
        pytest.approx(0.7),
        pytest.approx(0.04),
        pytest.approx(-0.04),
    ]
    assert spec.ee_body == "hand" and spec.ee_body_source == "the parent body of the gripper joints"
    assert spec.tcp_site == "grasp_site" and spec.tcp_offset == (0.0, 0.0, 0.05)


def test_robot_kit_overrides_defaults_and_mirrors_the_passive_finger(tmp_path):
    from metasim.utils.robot_kit import inspect_mjcf

    path = tmp_path / "toy_actuated.xml"
    path.write_text(_ACTUATED_ARM)
    spec = inspect_mjcf(str(path), defaults={"j2": 0.2, "f_l": 0.01})
    by_name = {j.name: j for j in spec.joints}
    assert by_name["j2"].default == pytest.approx(0.2) and by_name["f_l"].default == pytest.approx(0.01)
    assert by_name["f_r"].default == pytest.approx(-0.01), "the passive finger mirrors the driven one"
    with pytest.raises(ValueError, match="does not have"):
        inspect_mjcf(str(path), defaults={"nope": 0.0})


def test_robot_kit_without_actuators_or_keyframe_guesses_and_says_so(tmp_path):
    from metasim.utils.robot_kit import inspect_mjcf

    path = tmp_path / "toy.xml"
    path.write_text(_ARM)
    spec = inspect_mjcf(str(path))
    by_name = {j.name: j for j in spec.joints}
    assert all(j.guessed_gains for j in spec.joints) and any("placeholder servo gains" in n for n in spec.notes)
    assert by_name["f"].default == pytest.approx(0.04), "a gripper joint without a keyframe defaults to open"
    assert by_name["j1"].default == 0.0 and any("no keyframe" in n for n in spec.notes)


def test_robot_kit_renders_a_config_that_imports_and_passes_the_static_checks(tmp_path):
    from metasim.utils.robot_check import check_actuators, check_config, check_kinematics, check_limits
    from metasim.utils.robot_kit import add_robot, load_cfg_from_file

    path = tmp_path / "toy_actuated.xml"
    path.write_text(_ACTUATED_ARM)
    out = tmp_path / "toy_actuated_cfg.py"
    spec, cls = add_robot(mjcf=str(path), name="toy_actuated", out=str(out))
    assert cls == "ToyActuatedCfg"
    source = out.read_text()
    assert '"f_r": (-0.04, 0),  # passive' in source and 'ee_joint_names: list[str] = ["f_l"]' in source
    assert "TODO" in source and "Generator notes" in source
    assert "\"j2\": BaseActuatorCfg(stiffness=30, damping=3),  # MJCF actuator 'elbow'" in source
    cfg = load_cfg_from_file(str(out), cls)
    assert cfg.name == "toy_actuated" and cfg.num_joints == 4 and cfg.fix_base_link
    assert sorted(cfg.actuators) == ["f_l", "j1", "j2"] and cfg.actuators["f_l"].is_ee
    assert cfg.curobo_tcp_rel_pos == (0.0, 0.0, 0.05)
    assert check_config(cfg).ok and check_limits(cfg).ok and check_kinematics(cfg).ok
    row = check_actuators(cfg)
    assert not row.ok and "'elbow' drives 'j2'" in row.detail, "the misnamed actuator is a named failure"
    with pytest.raises(FileExistsError):
        add_robot(mjcf=str(path), name="toy_actuated", out=str(out))


def test_actuators_check_names_misnamed_and_unknown_joints(tmp_path):
    from metasim.utils.robot_check import check_actuators

    good = check_actuators(_cfg(tmp_path))
    assert good.ok and "3 joint(s) driven by a servo synthesized" in good.detail
    path = tmp_path / "toy_actuated.xml"
    path.write_text(_ACTUATED_ARM)
    bad = check_actuators(
        _cfg(tmp_path, mjcf_path=str(path), actuators={"j1": None, "j2": None, "f_l": None, "ghost": None})
    )
    assert not bad.ok and "'elbow' drives 'j2'" in bad.detail and "['ghost']" in bad.detail
    assert check_actuators(_cfg(tmp_path, mjcf_path=None)).ok, "no MJCF: not judged"


def test_cli_has_robot_add():
    import argparse

    import metasim.__main__ as cli

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    cli._add_robot(sub)
    ns = p.parse_args(["robot", "add", "--mjcf", "x.xml", "--name", "toy", "--default", "j1=0.5", "--no-hold"])
    assert ns.func is cli._robot_add and ns.default == ["j1=0.5"] and ns.out is None and not ns.no_check


def test_canned_trajectory_starts_near_the_defaults_and_stays_inside_the_limits(tmp_path):
    import numpy as np

    from metasim.utils.robot_check import canned_trajectory

    cfg = _cfg(tmp_path)
    joints = sorted(cfg.actuators)
    q = canned_trajectory(cfg, joints, steps=40, period_steps=20, amplitude_frac=0.5)
    assert q.shape == (40, 3)
    for k, j in enumerate(joints):
        lo, hi = cfg.joint_limits[j]
        assert np.all(q[:, k] >= lo - 1e-9) and np.all(q[:, k] <= hi + 1e-9)
    assert q[0, 0] == pytest.approx(cfg.default_joint_positions[joints[0]]), "phase 0: joint 0 starts at its default"
    assert np.ptp(q[:, joints.index("j2")]) > 0.5, "the trajectory actually moves"


def test_compare_traces_judges_each_joint_against_its_unit_tolerance(tmp_path):
    import numpy as np

    from metasim.utils.robot_check import TrackingTrace, canned_trajectory, compare_traces

    cfg = _cfg(tmp_path)
    joints = sorted(cfg.actuators)  # ["f", "j1", "j2"]
    cmd = canned_trajectory(cfg, joints, steps=30, period_steps=15)
    a = TrackingTrace("mujoco", joints, cmd, cmd.copy(), control_dt=0.015)
    b_meas = cmd.copy()
    b_meas[:, joints.index("j2")] += 0.03  # inside the 0.05 rad hinge tolerance
    b_meas[:, joints.index("f")] += 0.01  # outside the 0.005 m slide tolerance
    b = TrackingTrace("isaacsim", joints, cmd, b_meas)
    report = compare_traces(cfg, a, b)
    rows = {r.joint: r for r in report.rows}
    assert rows["j2"].ok and rows["j2"].kind == "hinge" and rows["j2"].max_diff == pytest.approx(0.03)
    assert not rows["f"].ok and rows["f"].kind == "slide" and rows["f"].tol == 0.005
    assert not report.ok and "verdict: FAIL" in report.table() and "| f | m |" in report.table()
    assert rows["j2"].lag_a == 0 and rows["j2"].lag_b == 0
    late = TrackingTrace("late", joints, cmd, np.roll(cmd, 3, axis=0))
    assert list(late.lag_steps()) == [3, 3, 3], "a trace that follows three steps behind reports lag 3"
    assert report.to_dict()["rows"][0]["ok"] in (True, False) and report.steps == 30
    round_trip = TrackingTrace.from_dict(b.to_dict())
    assert np.allclose(round_trip.measured, b.measured) and round_trip.simulator == "isaacsim"
    floating = compare_traces(_cfg(tmp_path, fix_base_link=False), a, b)
    assert floating.ok and not floating.judged and "not judged" in floating.table() and "reported" in floating.table()
    with pytest.raises(ValueError, match="same command"):
        compare_traces(cfg, a, TrackingTrace("x", joints, cmd + 1.0, cmd))
    with pytest.raises(ValueError, match="joint order"):
        compare_traces(cfg, a, TrackingTrace("x", list(reversed(joints)), cmd, cmd))


def test_cli_has_robot_parity():
    import argparse

    import metasim.__main__ as cli

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    cli._add_robot(sub)
    ns = p.parse_args(["robot", "parity", "franka", "--sims", "mujoco", "isaacsim", "--tol", "0.02"])
    assert ns.func is cli._robot_parity and ns.sims == ["mujoco", "isaacsim"] and ns.tol == 0.02 and ns.steps == 400
