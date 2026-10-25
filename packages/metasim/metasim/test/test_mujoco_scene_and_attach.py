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
        plane_count = sum(1 for i in range(model.ngeom) if model.geom_type[i] == int(mujoco.mjtGeom.mjGEOM_PLANE))
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
        assert free_parent == 0, f"freejoint body {free_body} parented to {free_parent}, not world; promotion failed"
    finally:
        handler.close()


# ---------------------------------------------------------------------------
# Actuation contract: cfg-synthesized actuators and passive (unactuated) joints
# ---------------------------------------------------------------------------


_TWO_JOINT_ARM = """
<mujoco model="two_joint_arm">
  <compiler angle="radian"/>
  <option integrator="implicitfast"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_g" type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="link1_g" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="link2_g" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  {actuators}
</mujoco>
"""


def _arm_scenario(tmp_path, actuators_xml: str, **robot_kwargs):
    from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg

    robot_xml = _write_xml(tmp_path, "arm.xml", _TWO_JOINT_ARM.format(actuators=actuators_xml))
    defaults = dict(
        name="arm",
        num_joints=2,
        fix_base_link=True,
        mjcf_path=str(robot_xml),
        enabled_gravity=False,
        enabled_self_collisions=False,
        actuators={
            "shoulder": BaseActuatorCfg(stiffness=50.0, damping=5.0, effort_limit_sim=20.0),
            "elbow": BaseActuatorCfg(stiffness=50.0, damping=5.0, effort_limit_sim=20.0),
        },
        joint_limits={"shoulder": (-1.5, 1.5), "elbow": (-1.5, 1.5)},
        default_joint_positions={"shoulder": 0.0, "elbow": 0.0},
        control_type={"shoulder": "position", "elbow": "position"},
    )
    defaults.update(robot_kwargs)
    robot = RobotCfg(**defaults)
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.002),
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


def test_cfg_actuators_are_synthesized_for_an_unactuated_mjcf(tmp_path):
    """A robot MJCF without an ``<actuator>`` block is driven through the gains its RobotCfg declares."""
    import torch

    from metasim.sim.sim_context import HandlerContext

    with HandlerContext(_arm_scenario(tmp_path, actuators_xml="")) as handler:
        model = handler.physics.model
        assert model.nu == 2, "one position actuator per cfg-controlled joint"
        prefix = handler._mujoco_robot_names[0]
        act = model.actuator(f"{prefix}elbow")
        assert float(model.actuator_gainprm[act.id, 0]) == pytest.approx(50.0)  # kp from the cfg
        assert float(model.actuator_biasprm[act.id, 2]) == pytest.approx(-5.0)  # kv from the cfg
        assert list(model.actuator_ctrlrange[act.id]) == pytest.approx([-1.5, 1.5])  # the cfg joint limits
        assert list(model.actuator_forcerange[act.id]) == pytest.approx([-20.0, 20.0])

        # The synthesized servo tracks a target like an asset-authored one would.
        target = torch.tensor([[0.3, -0.4]])  # sorted joints: elbow, shoulder
        for _ in range(400):
            handler.set_dof_targets(target)
            handler.simulate()
        state = handler.get_states(mode="tensor").robots["arm"]
        assert torch.allclose(state.joint_pos, target, atol=0.02)
        assert torch.allclose(state.joint_pos_target, target)


def test_cfg_actuator_without_gains_or_mjcf_actuator_is_rejected(tmp_path):
    """Nothing can drive a cfg-controlled joint with neither gains nor an asset actuator: fail at construction."""
    from metasim.scenario.robot import BaseActuatorCfg
    from metasim.sim.mujoco import MujocoHandler

    scenario = _arm_scenario(
        tmp_path,
        actuators_xml="",
        actuators={"shoulder": BaseActuatorCfg(), "elbow": BaseActuatorCfg()},
    )
    with pytest.raises(ValueError, match="ships no actuator for joint 'shoulder'"):
        MujocoHandler(scenario).launch()


def test_cfg_actuator_for_a_joint_the_mjcf_lacks_is_rejected(tmp_path):
    from metasim.scenario.robot import BaseActuatorCfg
    from metasim.sim.mujoco import MujocoHandler

    scenario = _arm_scenario(
        tmp_path,
        actuators_xml="",
        actuators={"shoulder": BaseActuatorCfg(stiffness=1.0), "wrist": BaseActuatorCfg(stiffness=1.0)},
        joint_limits={"shoulder": (-1.5, 1.5), "elbow": (-1.5, 1.5), "wrist": (-1.0, 1.0)},
        control_type={"shoulder": "position", "wrist": "position"},
    )
    with pytest.raises(ValueError, match="names joint 'wrist', which the MJCF"):
        MujocoHandler(scenario).launch()


def test_passive_joint_is_in_the_state_and_takes_no_target(tmp_path):
    """A joint without an actuator (and without a cfg actuator) is passive.

    It keeps its slot in ``joint_pos`` / ``joint_vel`` / the action tensor, no ctrl is written for it, and
    its ``joint_pos_target`` reports the position it has.
    """
    import torch

    from metasim.scenario.robot import BaseActuatorCfg
    from metasim.sim.sim_context import HandlerContext

    scenario = _arm_scenario(
        tmp_path,
        actuators_xml='<actuator><position name="shoulder" joint="shoulder" kp="50" kv="5"/></actuator>',
        actuators={"shoulder": BaseActuatorCfg(stiffness=50.0, damping=5.0)},
        control_type={"shoulder": "position"},
        enabled_gravity=True,
    )
    with HandlerContext(scenario) as handler:
        assert handler.get_joint_names("arm", sort=True) == ["elbow", "shoulder"]
        assert handler._passive_joint_names("arm") == ["elbow"]
        assert handler.physics.model.nu == 1

        # Action tensor keeps one column per joint; the passive column is ignored.
        target = torch.tensor([[1.0, 0.5]])  # elbow (passive), shoulder
        for _ in range(300):
            handler.set_dof_targets(target)
            handler.simulate()
        state = handler.get_states(mode="tensor").robots["arm"]
        assert state.joint_pos.shape == (1, 2)
        assert float(state.joint_pos[0, 1]) == pytest.approx(0.5, abs=0.02), "the actuated joint tracks"
        assert float(state.joint_pos[0, 0]) != pytest.approx(1.0, abs=0.1), "the passive joint received no target"
        assert float(state.joint_pos_target[0, 0]) == pytest.approx(float(state.joint_pos[0, 0]))
        assert float(state.joint_pos_target[0, 1]) == pytest.approx(0.5)
        assert float(state.joint_effort_target[0, 0]) == 0.0

        dict_state = handler.get_states(mode="dict")[0]["robots"]["arm"]
        assert set(dict_state["dof_pos_target"]) == {"elbow", "shoulder"}
        assert dict_state["dof_pos_target"]["elbow"] == pytest.approx(float(state.joint_pos[0, 0]))


def test_actuator_not_named_after_a_joint_is_rejected(tmp_path):
    """RoboVerse drives joints through same-named actuators; a stray actuator name is a config error."""
    from metasim.sim.mujoco import MujocoHandler

    scenario = _arm_scenario(
        tmp_path,
        actuators_xml=(
            '<actuator><position name="shoulder" joint="shoulder" kp="50"/>'
            '<position name="elbow_drive" joint="elbow" kp="50"/></actuator>'
        ),
    )
    handler = MujocoHandler(scenario)
    handler.launch()
    try:
        with pytest.raises(ValueError, match=r"actuators \['elbow_drive'\] are not named after a joint"):
            handler.get_states(mode="tensor")
    finally:
        handler.close()
