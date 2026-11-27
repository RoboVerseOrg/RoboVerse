"""MuJoCo handler regressions found on LIBERO assets.

1. An attached asset that declares the legacy ``<size njmax=...>`` (LIBERO's stove and cabinets do) used to
   fail compilation: dm_control merges it into the root, which already sizes the arena with ``memory``, and
   MuJoCo rejects a model with both.
2. Scaling an object collapsed a box site's three half-extents to one number.
3. ``set_states`` with a joint value stored as a 1-element array (LIBERO demo initial states) raised
   "setting an array element with a sequence" when writing the joint's actuator ``ctrl``.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

_DRAWER = """
<mujoco model="drawer">
  <size njmax="500" nconmax="100"/>
  <worldbody>
    <body name="frame">
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="tray" pos="0 0 0.15">
        <joint name="slide" type="slide" axis="0 1 0" range="-0.1 0.1"/>
        <geom type="box" size="0.05 0.05 0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="slide" joint="slide" kp="10"/>
  </actuator>
</mujoco>
"""


def _launch(tmp_path, xml: str):
    from metasim.scenario.objects import ArticulationObjCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.mujoco import MujocoHandler

    path = tmp_path / "drawer.xml"
    path.write_text(textwrap.dedent(xml).strip() + "\n")
    scenario = ScenarioCfg(
        objects=[ArticulationObjCfg(name="drawer", mjcf_path=str(path), fix_base_link=True)],
        robots=[],
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    handler = MujocoHandler(scenario)
    handler.launch()
    return handler


def test_attached_njmax_does_not_conflict_with_the_arena(tmp_path):
    from metasim.sim.mujoco.mujoco import MUJOCO_ARENA_MEMORY

    handler = _launch(tmp_path, _DRAWER)
    try:
        assert int(handler.physics.model.narena) >= int(MUJOCO_ARENA_MEMORY[:-1]) * 2**20
    finally:
        handler.close()


def test_set_states_accepts_one_element_array_joint_values(tmp_path):
    handler = _launch(tmp_path, _DRAWER.replace('<size njmax="500" nconmax="100"/>', ""))
    try:
        _write_one_element_state(handler)
    finally:
        handler.close()


def _write_one_element_state(handler):
    handler.set_states([
        {
            "objects": {"drawer": {"pos": [0, 0, 0], "rot": [1, 0, 0, 0], "dof_pos": {"slide": np.array([0.05])}}},
            "robots": {},
        }
    ])
    state = handler.get_states(mode="tensor").objects["drawer"]
    assert float(state.joint_pos[0, 0]) == pytest.approx(0.05)
    assert float(handler.physics.data.ctrl[0]) == pytest.approx(0.05)


def test_scaling_keeps_a_box_site_box_shaped(tmp_path):
    from metasim.scenario.objects import RigidObjCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.mujoco import MujocoHandler

    path = tmp_path / "tray.xml"
    path.write_text(
        '<mujoco model="tray"><worldbody><body name="object"><geom type="box" size="0.1 0.1 0.01" mass="1"/>'
        '<site name="contain_region" type="box" pos="0 0 0.02" size="0.04 0.08 0.01"/></body></worldbody></mujoco>'
    )
    obj = RigidObjCfg(name="tray", mjcf_path=str(path), scale=(2.0, 0.5, 3.0))
    handler = MujocoHandler(ScenarioCfg(objects=[obj], robots=[], simulator="mujoco", num_envs=1, headless=True))
    handler.launch()
    try:
        size = handler.physics.named.model.site_size["tray/contain_region"]
        np.testing.assert_allclose(size, [0.08, 0.04, 0.03])
        from metasim.utils.site_util import site_pose

        states = handler.get_states(mode="tensor")
        assert site_pose(states, obj=obj, site="contain_region").size == pytest.approx(tuple(size))
    finally:
        handler.close()
