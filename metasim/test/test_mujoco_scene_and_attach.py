"""Regression tests for the MuJoCo backend's scene/attach pipeline.

Covers three small fixes landed together:

1. ``ScenarioCfg.add_default_ground`` opt-out — when ``ground is None`` and
   ``add_default_ground=False``, ``MujocoHandler._init_mujoco`` must not
   inject the default ``//unnamed_geom_0`` plane. Needed for loading
   complete scene-style MJCFs (dm_control cartpole, mjlab playground)
   whose own MJCF already declares a floor; the extra plane caused
   ground-penetration contacts and broke physics parity.

2. ``_add_robots_to_model`` — robots whose root body has no explicit
   ``<inertial>`` tag (URDFs rely on MuJoCo's geom-mass auto-inertia)
   must not raise AttributeError when MetaSim builds its dummy wrapper
   inertial. Falls back to (0, 0, 0) for the dummy inertial's pos.

3. ``_add_robots_to_model`` freejoint promotion — when a robot MJCF
   already ships a ``<freejoint>`` inside its root body (mjlab unitree
   go1/g1 do this), wrapping the robot makes that body non-top-level
   and MuJoCo errors out with "free joint can only be used on top
   level". The fix removes the inner freejoint and re-adds one to the
   attachment wrapper (which is top-level by construction).

Tests use real dm_control mjcf so they exercise the actual code path.
They're marked ``@pytest.mark.mujoco`` so the autotest selection drives
them only when a mujoco-enabled env is active.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


pytestmark = pytest.mark.mujoco


def _write_xml(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


# ---------------------------------------------------------------------------
# Fix 1: add_default_ground opt-out
# ---------------------------------------------------------------------------


def test_add_default_ground_opt_out_skips_unnamed_ground(tmp_path):
    """A scene loaded with add_default_ground=False must not get an
    extra default plane geom on top of whatever the scene already has."""
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco import MujocoHandler

    # Minimal cartpole-style scene with its own floor.
    scene_xml = _write_xml(
        tmp_path,
        "scene.xml",
        """
        <mujoco model="mini_cart">
          <option timestep="0.01"/>
          <worldbody>
            <geom name="own_floor" type="plane" size="4 4 0.1" pos="0 0 -0.05"/>
            <body name="cart" pos="0 0 1">
              <joint name="slider" type="slide" axis="1 0 0"/>
              <geom name="cart_box" type="box" size="0.1 0.1 0.1" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="drive" joint="slider" gear="1"/>
          </actuator>
        </mujoco>
        """,
    )

    scenario = ScenarioCfg(
        scene=SceneCfg(mjcf_path=str(scene_xml)),
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = MujocoHandler(scenario)
    handler.launch()
    try:
        import mujoco

        model = handler.physics.model.ptr
        geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        assert any("own_floor" in n for n in geom_names), f"scene-own floor missing: {geom_names}"
        # The MetaSim default ground is added as an unnamed geom (e.g. "//unnamed_geom_0"
        # or similar dm_control auto-name). When opted out, no auto plane should appear.
        unnamed_planes = [
            i
            for i in range(model.ngeom)
            if model.geom_type[i] == int(mujoco.mjtGeom.mjGEOM_PLANE)
            and (geom_names[i].startswith("//") or geom_names[i] == "")
        ]
        assert unnamed_planes == [], f"unexpected auto-ground plane(s): {[geom_names[i] for i in unnamed_planes]}"
    finally:
        try:
            handler.close()
        except Exception:
            pass


def test_add_default_ground_default_keeps_legacy_behavior(tmp_path):
    """Backward compat: default add_default_ground=True still attaches
    the unnamed plane (no behavior change for existing callers)."""
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco import MujocoHandler

    scenario = ScenarioCfg(
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    # Explicit default — keeps prior contract.
    assert scenario.add_default_ground is True

    handler = MujocoHandler(scenario)
    handler.launch()
    try:
        import mujoco

        model = handler.physics.model.ptr
        plane_count = sum(
            1
            for i in range(model.ngeom)
            if model.geom_type[i] == int(mujoco.mjtGeom.mjGEOM_PLANE)
        )
        assert plane_count >= 1, "expected at least one ground plane under default behavior"
    finally:
        try:
            handler.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fix 2: missing-inertial fallback
# ---------------------------------------------------------------------------


def test_robot_without_root_inertial_loads_cleanly(tmp_path):
    """A robot MJCF that omits <inertial> on its root body (and lets
    MuJoCo compute inertia from the geom mass) must load without
    AttributeError."""
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco import MujocoHandler

    robot_xml = _write_xml(
        tmp_path,
        "noinertial_robot.xml",
        """
        <mujoco model="noinertial_robot">
          <worldbody>
            <body name="root" pos="0 0 0">
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom name="g1" type="box" size="0.05 0.05 0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="m1" joint="j1" gear="1"/>
          </actuator>
        </mujoco>
        """,
    )

    robot = RobotCfg(
        name="noinertial_robot",
        num_joints=1,
        fix_base_link=True,
        mjcf_path=str(robot_xml),
        enabled_gravity=True,
        enabled_self_collisions=True,
        control_type={},
    )
    scenario = ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.01),
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = MujocoHandler(scenario)
    handler.launch()  # would have raised AttributeError before the fix
    handler.close()


# ---------------------------------------------------------------------------
# Fix 3: inner freejoint promotion
# ---------------------------------------------------------------------------


def test_inner_freejoint_is_promoted_to_wrapper(tmp_path):
    """A robot MJCF with <freejoint> inside its root body must compile
    after attach — MuJoCo rejects free joints on non-top-level bodies."""
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco import MujocoHandler

    robot_xml = _write_xml(
        tmp_path,
        "floating_robot.xml",
        """
        <mujoco model="floating_robot">
          <worldbody>
            <body name="trunk" pos="0 0 0.5">
              <freejoint name="floating_base_joint"/>
              <geom name="trunk_g" type="box" size="0.1 0.1 0.05" mass="1"/>
              <body name="leg" pos="0.1 0 -0.1">
                <joint name="hip" type="hinge" axis="0 1 0"/>
                <geom name="leg_g" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.2"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    robot = RobotCfg(
        name="floating_robot",
        num_joints=2,  # freejoint counts as 1 joint; hip = 1
        fix_base_link=False,
        mjcf_path=str(robot_xml),
        enabled_gravity=True,
        enabled_self_collisions=True,
        control_type={},
    )
    scenario = ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.01),
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = MujocoHandler(scenario)
    handler.launch()  # would have raised "free joint can only be used on top level"
    try:
        import mujoco

        model = handler.physics.model.ptr
        # Exactly one freejoint should exist.
        free_jnts = [i for i in range(model.njnt) if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        assert len(free_jnts) == 1, f"expected exactly one freejoint, got {len(free_jnts)}"
        # The freejoint must be on a body that is a direct child of world (id 0).
        free_body = int(model.jnt_bodyid[free_jnts[0]])
        free_parent = int(model.body_parentid[free_body])
        assert free_parent == 0, (
            f"freejoint body {free_body} parented to {free_parent}, not world; promotion failed"
        )
    finally:
        handler.close()
