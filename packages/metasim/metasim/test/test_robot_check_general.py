"""The robot self-check's pure parts (config, kinematics) on a toy MJCF arm, and the CLI wiring; no simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
    assert none.ok and none.applicable and "not judged" in none.detail, "no MJCF: the backend's own checks decide"


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
    assert row.ok and "'elbow' drives 'j2'" in row.detail, "a differently named actuator is matched by transmission"
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
    assert not bad.ok and "['ghost']" in bad.detail
    renamed = check_actuators(_cfg(tmp_path, mjcf_path=str(path), actuators={"j1": None, "j2": None, "f_l": None}))
    assert renamed.ok and "'elbow' drives 'j2'" in renamed.detail, "an actuator is matched by transmission, not name"
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
    for k, j in enumerate(joints):
        assert q[0, k] == pytest.approx(cfg.default_joint_positions[j]), "every joint starts at its default (ramp)"
    assert np.ptp(q[20:, joints.index("j2")]) > 0.5, "after the ramp the trajectory actually moves"


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
    ns = p.parse_args([
        "robot",
        "parity",
        "franka",
        "--sims",
        "mujoco",
        "isaacsim",
        "--tol",
        "0.02",
        "--amplitude-max",
        "0",
    ])
    assert ns.func is cli._robot_parity and ns.sims == ["mujoco", "isaacsim"] and ns.tol == 0.02 and ns.steps == 400
    assert ns.amplitude_max == 0.0


def test_robot_parity_forwards_the_trajectory_flags(monkeypatch, tmp_path):
    """The CLI hands every trajectory flag to track_trajectory (``--amplitude-max`` was once parsed and ignored)."""
    import argparse

    import metasim.__main__ as cli
    from metasim.utils import robot_check as rc
    from metasim.utils import setup_util

    cfg = _cfg(tmp_path)
    seen = []

    def fake_track(cfg_, sim, **kwargs):
        seen.append((sim, kwargs))
        joints = sorted(cfg_.actuators)
        cmd = rc.canned_trajectory(cfg_, joints, steps=kwargs["steps"], period_steps=kwargs["period_steps"])
        return rc.TrackingTrace(sim, joints, cmd, cmd.copy(), control_dt=0.015)

    monkeypatch.setattr(setup_util, "get_robot", lambda name: cfg)
    monkeypatch.setattr(rc, "track_trajectory", fake_track)
    ns = argparse.Namespace(
        robot="toy",
        sims=["a", "b"],
        load=None,
        save=None,
        steps=30,
        period=15,
        amplitude=0.1,
        amplitude_max=0.0,
        settle=5,
        tol=0.05,
        tol_slide=0.005,
        json=False,
    )
    assert cli._robot_parity(ns) == 0
    assert [s for s, _ in seen] == ["a", "b"]
    assert seen[0][1] == {
        "steps": 30,
        "period_steps": 15,
        "amplitude_frac": 0.1,
        "amplitude_max": 0.0,
        "settle_steps": 5,
    }


def test_headless_gl_decision_is_one_pure_rule():
    """Linux, no display, nothing set, a loadable offscreen library -> egl (or osmesa); every other case leaves MuJoCo alone."""
    from metasim.sim.gl import apply_headless_gl, choose_headless_gl

    def pick(env, platform="linux", mujoco_loaded=False, is_loadable=lambda lib: True):
        return choose_headless_gl(environ=env, platform=platform, mujoco_loaded=mujoco_loaded, is_loadable=is_loadable)

    assert pick({})[0] == "egl"
    assert pick({}, is_loadable=lambda lib: lib == "libOSMesa.so.8")[0] == "osmesa"
    none = pick({}, is_loadable=lambda lib: False)
    assert none[0] is None and "no offscreen GL library" in none[1]
    assert pick({"MUJOCO_GL": "disable"})[0] is None and "explicitly" in pick({"MUJOCO_GL": "disable"})[1]
    assert pick({"DISPLAY": ":0"})[0] is None and pick({"WAYLAND_DISPLAY": "wayland-0"})[0] is None
    assert pick({}, platform="darwin")[0] is None and "Linux only" in pick({}, platform="darwin")[1]
    late = pick({}, mujoco_loaded=True)
    assert late[0] is None and "imported before" in late[1]
    env: dict[str, str] = {}
    chosen = apply_headless_gl(env)
    assert (chosen is None) == ("MUJOCO_GL" not in env), (
        "apply sets exactly what choose picked (this process's real state)"
    )


def test_mujoco_backend_import_applies_the_headless_rule_once():
    """One subprocess: after importing the backend package, MUJOCO_GL is what the rule says for that process."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL")}
    code = (
        "import os; from metasim.sim.gl import choose_headless_gl; expected = choose_headless_gl()[0];"
        " import metasim.sim.mujoco; print(os.environ.get('MUJOCO_GL') == expected)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180, check=False
    )
    assert out.stdout.strip() == "True", out.stderr[-800:]


def test_list_robots_walks_every_module_of_the_robot_packages():
    from metasim.utils.setup_util import list_robots

    robots = list_robots(["metasim.example.example_pack.robots"])
    assert "franka" in robots and robots["franka"].endswith(":FrankaCfg")
    assert "g1_dof12" in robots and "g1_dof23" in robots, "two classes in one module are both listed"
    assert "robot_template" not in robots and list(robots) == sorted(robots)
    assert list_robots(["not.a.package"]) == {}, "a package that does not import is skipped, not fatal"


def test_robot_check_all_sweeps_the_list_and_writes_the_table(monkeypatch, tmp_path, capsys):
    import metasim.__main__ as cli
    from metasim.utils import robot_check as rc
    from metasim.utils import setup_util

    monkeypatch.setattr(setup_util, "list_robots", lambda: {"good": "m:Good", "bad": "m:Bad", "broken": "m:Broken"})

    def fake_check(name, *, sim, steps, hold, static=True):
        if name == "broken":
            raise ValueError("no such robot")
        ok = name == "good"
        return rc.RobotReport(
            robot=name,
            simulator=sim,
            rows=[rc.CheckRow("config", True, "fine"), rc.CheckRow("hold", ok, "drift" if not ok else "still")],
        )

    monkeypatch.setattr(rc, "check_robot", fake_check)
    monkeypatch.setattr(cli.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    out = tmp_path / "sweep"
    with pytest.raises(SystemExit) as exc:
        cli.main(["robot", "check", "--all", "--sim", "mujoco", "--out-dir", str(out)])
    assert exc.value.code == 1
    table = capsys.readouterr().out
    assert (
        "| good | ok | - | - | - | - | ok | pass |" in table and "| bad | ok | - | - | - | - | FAIL | FAIL |" in table
    )
    assert "| broken | FAIL |" in table and "ValueError: no such robot" in table
    assert "3 robot(s), 1 pass, 2 fail" in table
    data = json.loads((out / "robots.json").read_text())
    assert data["simulator"] == "mujoco" and [r["robot"] for r in data["robots"]] == ["good", "bad", "broken"]
    assert (out / "robots.md").read_text().startswith("| robot | config |")
    assert cli.main(["robot", "check"]) == 2, "no names and no --all is a usage error"
    cli.main(["robot", "list", "--json"])
    assert json.loads(capsys.readouterr().out) == {"good": "m:Good", "bad": "m:Bad", "broken": "m:Broken"}


def test_robot_check_isolated_spawns_a_child_only_when_the_cheap_rows_pass(monkeypatch, tmp_path, capsys):
    """Isaac Sim boots once per process: the sweep runs one child per robot, and skips the boot for a robot whose
    assets are already missing; a child that dies without a report is a named hold failure."""
    import argparse
    import subprocess
    from types import SimpleNamespace

    import metasim.__main__ as cli
    from metasim.utils import robot_check as rc
    from metasim.utils import setup_util

    cheap = {
        "ready": rc.RobotReport(
            "ready", "isaacsim", [rc.CheckRow("config", True, ""), rc.CheckRow("assets", True, "")]
        ),
        "nousd": rc.RobotReport(
            "nousd", "isaacsim", [rc.CheckRow("config", True, ""), rc.CheckRow("assets", False, "no usd_path")]
        ),
        "dies": rc.RobotReport("dies", "isaacsim", [rc.CheckRow("config", True, ""), rc.CheckRow("assets", True, "")]),
    }
    monkeypatch.setattr(setup_util, "get_robot", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(rc, "check_robot", lambda name, *, sim, steps, hold: cheap[name])
    spawned = []

    def fake_run(cmd, **kwargs):
        name = cmd[cmd.index("check") + 1]
        spawned.append(name)
        assert cmd[cmd.index("--isolate") + 1] == "never" and "--hold-only" in cmd, "the child runs only the hold"
        if name == "dies":
            return subprocess.CompletedProcess(
                cmd, 1, stdout="[INFO] booting\n", stderr="Kit aborted\nSegmentation fault"
            )
        report = rc.RobotReport(
            name, "isaacsim", [rc.CheckRow("config", True, "c"), rc.CheckRow("hold", True, "still")]
        )
        child_dir = cmd[cmd.index("--out-dir") + 1]
        with open(os.path.join(child_dir, "robots.json"), "w") as fh:
            json.dump({"simulator": "isaacsim", "robots": [report.to_dict()]}, fh)
        return subprocess.CompletedProcess(cmd, 0, stdout="[INFO] lots of [bracketed] log lines\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(sim="isaacsim", steps=200, no_hold=False, out_dir=str(tmp_path), json=False, all=True)
    stale_dir = tmp_path / "children" / "dies"
    stale_dir.mkdir(parents=True)
    (stale_dir / "robots.json").write_text(json.dumps({"simulator": "isaacsim", "robots": [cheap["ready"].to_dict()]}))
    ready = cli._robot_check_isolated("ready", args)
    nousd = cli._robot_check_isolated("nousd", args)
    dies = cli._robot_check_isolated("dies", args)
    assert not (stale_dir / "robots.json").exists(), "a stale report from the last run is removed before the child runs"
    assert spawned == ["ready", "dies"], "no simulator boot for a robot whose assets are missing"
    assert ready.ok and [r.check for r in ready.rows] == ["config", "assets", "hold"], "parent rows + the child's hold"
    assert not nousd.ok and nousd.rows[-1].check == "hold" and "not attempted" in nousd.rows[-1].detail
    assert not dies.ok and "child exited 1 without a report: Segmentation fault" in dies.rows[-1].detail
    assert (tmp_path / "logs" / "dies.log").read_text().endswith("Segmentation fault")

    monkeypatch.setattr(setup_util, "list_robots", lambda: {"ready": "m:R", "nousd": "m:N"})
    monkeypatch.setattr(cli.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as exc:
        cli.main(["robot", "check", "--all", "--sim", "isaacsim", "--out-dir", str(tmp_path)])
    assert exc.value.code == 1 and spawned == ["ready", "dies", "ready"], (
        "auto isolation on isaacsim with several robots"
    )
    err = capsys.readouterr().err
    assert "ready" in err and "FAIL" in err, "the sweep reports each robot as it finishes"


def test_a_robot_without_an_asset_for_the_backend_is_not_applicable(tmp_path):
    """A URDF-only robot on MuJoCo is reported n/a: it neither passes nor fails the sweep on that backend."""
    from metasim.utils.robot_check import RobotReport, check_assets, check_robot_cfg

    cfg = _cfg(tmp_path, mjcf_path=None, urdf_path="x.urdf")
    row = check_assets(cfg, "mujoco")
    assert not row.ok and not row.applicable and "no mjcf_path declared" in row.detail
    report = check_robot_cfg(cfg, sim="mujoco", hold=True)
    assert not report.applicable and report.ok and report.verdict == "n/a"
    hold = report.rows[-1]
    assert hold.check == "hold" and not hold.applicable and hold.detail.startswith("n/a")
    assert "verdict: n/a" in report.table() and report.to_dict()["verdict"] == "n/a"
    failing = RobotReport("x", "mujoco", [check_assets(_cfg(tmp_path), "mujoco")])
    assert failing.applicable and failing.verdict == "pass"
    urdf_only_with_bad_config = check_robot_cfg(
        _cfg(tmp_path, mjcf_path=None, urdf_path="x.urdf", actuators={}), sim="mujoco"
    )
    assert urdf_only_with_bad_config.verdict == "FAIL", "a config defect fails the robot on every backend"
    (tmp_path / "toy.usda").write_text("#usda 1.0\n")
    usd_only = check_robot_cfg(
        _cfg(tmp_path, mjcf_path=None, usd_path=str(tmp_path / "toy.usda")), sim="isaacsim", hold=False
    )
    assert usd_only.applicable and usd_only.ok, "a USD-only robot on Isaac Sim is a candidate: its hold check must run"
    hold_only = check_robot_cfg(_cfg(tmp_path), sim="mujoco", hold=False, static=False)
    assert [r.check for r in hold_only.rows] == ["assets"]


def test_asset_fields_must_name_their_format():
    from metasim.utils.asset_check import check_asset_fields

    cfg = SimpleNamespace(mjcf_path="r/robot.xml", usd_path="r/robot.xml", urdf_path=None)
    problems = check_asset_fields(cfg)
    assert len(problems) == 1 and problems[0].startswith("usd_path='r/robot.xml' is not a .usd")
    assert check_asset_fields(SimpleNamespace(mjcf_path="r/robot.xml", usd_path="r/robot.usda")) == []
    # the config row fails on it (a defect on every backend), with the joints still consistent
    cfg = SimpleNamespace(
        actuators={"j": object()},
        joint_limits={"j": (-1.0, 1.0)},
        default_joint_positions={"j": 0.0},
        mjcf_path="r/robot.xml",
        usd_path="r/robot.xml",
    )
    row = check_config(cfg)
    assert not row.ok and "usd_path" in row.detail


def test_usd_dependencies_are_not_judged_without_pxr(monkeypatch):
    import sys

    from metasim.utils.asset_check import fetch_usd_dependencies, usd_dependencies

    monkeypatch.setitem(sys.modules, "pxr", None)  # ``from pxr import Sdf`` raises ImportError
    assert usd_dependencies("whatever.usd") is None
    assert fetch_usd_dependencies("whatever.usd", fetch=lambda p: None) is None


def test_usd_dependencies_follow_payloads_and_fetch_the_missing_ones(tmp_path):
    pytest.importorskip("pxr")
    from metasim.utils.asset_check import fetch_usd_dependencies, usd_dependencies

    (tmp_path / "configuration").mkdir()
    (tmp_path / "configuration" / "robot_physics.usda").write_text('#usda 1.0\ndef Xform "robot" {}\n')
    (tmp_path / "robot.usda").write_text(
        '#usda 1.0\ndef Xform "robot" (payload = @./configuration/robot_physics.usda@)\n{}\n'
        'def Xform "sensors" (references = @./configuration/robot_sensor.usda@)\n{}\n'
    )
    deps = usd_dependencies(str(tmp_path / "robot.usda"))
    assert [os.path.basename(p) for p in deps.resolved] == ["robot_physics.usda"]
    assert [os.path.basename(p) for p in deps.unresolved] == ["robot_sensor.usda"]
    assert not deps.ok

    fetched = []

    def fetch(path):
        fetched.append(os.path.basename(path))
        Path(path).write_text('#usda 1.0\ndef Xform "s" {}\n')

    deps = fetch_usd_dependencies(str(tmp_path / "robot.usda"), fetch=fetch)
    assert deps.ok and fetched == ["robot_sensor.usda"]
    # a fetch that produces nothing leaves the file unresolved instead of looping
    (tmp_path / "configuration" / "robot_sensor.usda").unlink()
    deps = fetch_usd_dependencies(str(tmp_path / "robot.usda"), fetch=lambda p: None)
    assert [os.path.basename(p) for p in deps.unresolved] == ["robot_sensor.usda"]


def test_actuators_check_names_two_drivers_and_tendon_actuators(tmp_path):
    from metasim.utils.robot_check import check_actuators

    twice = _ACTUATED_ARM.replace("</actuator>", '<motor name="j2_boost" joint="j2"/></actuator>')
    path = tmp_path / "twice.xml"
    path.write_text(twice)
    row = check_actuators(_cfg(tmp_path, mjcf_path=str(path), actuators={"j1": None, "j2": None, "f_l": None}))
    assert not row.ok and "two actuators" in row.detail and "'j2': ['elbow', 'j2_boost']" in row.detail
    stray = _ACTUATED_ARM.replace(
        "<actuator>",
        '<tendon><fixed name="split"><joint joint="f_l" coef="1"/></fixed></tendon><actuator><general tendon="split"/>',
    )
    path.write_text(stray)
    row = check_actuators(_cfg(tmp_path, mjcf_path=str(path), actuators={"j1": None, "j2": None, "f_l": None}))
    assert not row.ok and "drive no joint" in row.detail and "'#0'" in row.detail
