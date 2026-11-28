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
