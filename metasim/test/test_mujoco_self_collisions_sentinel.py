"""Regression tests for the new ``enabled_self_collisions = "mujoco_default"`` sentinel.

Background: the MuJoCo backend used to read this flag as a strict
``bool``. ``True`` flipped the global ``filterparent`` flag to
``"disable"`` (all-pair self-collisions, including parent-child).
``False`` left ``filterparent`` at MuJoCo's default but added a
``<exclude>`` for every body pair inside the robot (no self-collision
at all). Neither matches MuJoCo's own default for an unwrapped robot
MJCF — and both biased the contact set in opposite directions, leaving
0.16 rad / 1.6 rad residuals on the mjlab go1 / yam parity sweeps.

The sentinel string keeps both knobs at MuJoCo defaults: parent-child
contacts filtered, all other intra-robot pairs collide naturally. With
this in place every mjlab task drops to machine-epsilon parity against
a raw ``mujoco.MjModel.from_xml_path`` rollout.

These tests use the same minimal-XML approach as
``test_mujoco_scene_and_attach.py`` so they run anywhere without a GPU.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


pytestmark = pytest.mark.mujoco


def _two_body_robot(tmp_path: Path) -> Path:
    p = tmp_path / "two_body.xml"
    p.write_text(
        textwrap.dedent(
            """
            <mujoco model="two_body">
              <worldbody>
                <body name="trunk" pos="0 0 0.5">
                  <freejoint/>
                  <geom name="trunk_g" type="box" size="0.05 0.05 0.05" mass="1"/>
                  <body name="arm" pos="0.05 0 0">
                    <joint name="elbow" type="hinge" axis="0 1 0"/>
                    <geom name="arm_g" type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        ).strip()
        + "\n"
    )
    return p


def _build(scenario, tmp_path):
    from metasim.sim.mujoco import MujocoHandler

    handler = MujocoHandler(scenario)
    handler.launch()
    try:
        import mujoco

        model = handler.physics.model.ptr
        excludes = int(model.nexclude)
        # Modern mujoco exposes filterparent as a disable-bit in opt.disableflags;
        # the bit is *set* when filterparent is *disabled*. Return 1 for
        # "default-enabled" so the test reads naturally.
        disabled = bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        filterparent = 0 if disabled else 1
        return excludes, filterparent
    finally:
        handler.close()


def _make_scenario(robot_xml: str, self_coll, robot_name: str = "robot"):
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg

    robot = RobotCfg(
        name=robot_name,
        num_joints=1,
        fix_base_link=False,
        mjcf_path=robot_xml,
        enabled_gravity=True,
        enabled_self_collisions=self_coll,
        control_type={},
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.01),
        decimation=2,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


def test_true_disables_filterparent_no_excludes(tmp_path):
    """enabled_self_collisions=True: filterparent off + no excludes."""
    xml = str(_two_body_robot(tmp_path))
    excludes, filterparent = _build(_make_scenario(xml, True), tmp_path)
    assert excludes == 0, "True branch must not add body-pair excludes"
    # MJ_FLG values: 1=enabled, 0=disabled.
    assert filterparent == 0, f"True branch must disable filterparent, got {filterparent}"


def test_false_adds_excludes_keeps_filterparent(tmp_path):
    """enabled_self_collisions=False: every intra-robot body pair excluded."""
    xml = str(_two_body_robot(tmp_path))
    excludes, filterparent = _build(_make_scenario(xml, False), tmp_path)
    # Two-body robot → C(2,2)=1 exclude pair.
    assert excludes >= 1, f"False branch must add excludes, got {excludes}"
    assert filterparent == 1, f"False branch must keep filterparent at default-enabled, got {filterparent}"


def test_mujoco_default_leaves_both_knobs(tmp_path):
    """enabled_self_collisions='mujoco_default': no excludes, filterparent stays default-enabled."""
    xml = str(_two_body_robot(tmp_path))
    excludes, filterparent = _build(_make_scenario(xml, "mujoco_default"), tmp_path)
    assert excludes == 0, f"sentinel must not add excludes, got {excludes}"
    assert filterparent == 1, f"sentinel must keep filterparent default-enabled, got {filterparent}"
