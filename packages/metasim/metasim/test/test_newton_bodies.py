"""Newton bodies: named, floated and weighed like on MuJoCo. Needs the newton engine (GPU env).

1. Newton >= 1.2 labels a body with its source-file path; the handler reports the last segment, so body
   names and the order of ``body_state`` / contact rows match MuJoCo.
2. An MJCF robot with ``fix_base_link=False`` whose root body declares no joint (H1, G1) was welded to the
   world: Newton's parser fixes a jointless root unless asked for a floating base.
3. ``set_body_masses`` reaches ``SolverMuJoCo``'s copy of the model and leaves the bodies it did not change
   bit-exact.
4. ``ContactForces`` works on the 1.2+ ``SensorContact`` API and names the same bodies as MuJoCo.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
import torch

pytest.importorskip("metasim.sim.newton.newton")

_WALKER = """
<mujoco model="walker">
  <worldbody>
    <body name="torso">
      <geom type="box" size="0.1 0.1 0.1" mass="4"/>
      <body name="leg_l" pos="0 0.08 -0.1">
        <joint name="hip_l" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
      </body>
      <body name="leg_r" pos="0 -0.08 -0.1">
        <joint name="hip_r" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _launch(tmp_path, simulator, *, fix_base_link=False, num_envs=1, queries=None, xml=_WALKER, height=1.0):
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.utils.setup_util import get_handler

    path = tmp_path / "walker.xml"
    path.write_text(textwrap.dedent(xml).strip() + "\n")
    robot = RobotCfg(
        name="walker",
        num_joints=2,
        fix_base_link=fix_base_link,
        mjcf_path=str(path),
        default_position=(0.0, 0.0, height),
        enabled_gravity=True,
        enabled_self_collisions=False,
        control_type={},
    )
    scenario = ScenarioCfg(robots=[robot], simulator=simulator, num_envs=num_envs, headless=True)
    return get_handler(scenario, optional_queries=queries)


def _run(handler, seconds):
    for _ in range(int(seconds / handler.physics_dt)):
        handler.simulate()


@pytest.fixture
def walker(tmp_path):
    handler = _launch(tmp_path, "newton", num_envs=2)
    yield handler
    handler.close()


def test_bodies_are_named_like_on_mujoco(walker):
    assert walker.get_body_names("walker") == ["leg_l", "leg_r", "torso"]
    assert walker.get_states(mode="tensor").robots["walker"].body_names == ["leg_l", "leg_r", "torso"]
    assert walker.get_joint_names("walker") == ["hip_l", "hip_r"]


def test_a_jointless_root_floats_unless_the_base_is_fixed(tmp_path, walker):
    _run(walker, 0.3)
    assert float(walker.get_states(mode="tensor").robots["walker"].root_state[0, 2]) < 0.8  # fell
    fixed = _launch(tmp_path, "newton", fix_base_link=True)
    try:
        _run(fixed, 0.3)
        assert float(fixed.get_states(mode="tensor").robots["walker"].root_state[0, 2]) == pytest.approx(1.0)
    finally:
        fixed.close()


def test_masses_reach_the_solver_and_spare_the_other_bodies(walker):
    model = walker._model
    inv_inertia, inv_mass = model.body_inv_inertia.numpy().copy(), model.body_inv_mass.numpy().copy()
    masses = walker.get_body_masses("walker")
    torch.testing.assert_close(masses, torch.tensor([[1.0, 1.0, 4.0]] * 2))
    heavy = masses[1:].clone()
    heavy[0, 2] = 8.0
    walker.set_body_masses("walker", heavy, env_ids=[1])
    torch.testing.assert_close(walker.get_body_masses("walker"), torch.tensor([[1.0, 1.0, 4.0], [1.0, 1.0, 8.0]]))
    torso_1 = walker._env_body_ids("walker")[1, walker.get_body_names("walker", sort=False).index("torso")]
    solver_mass = walker._solver.mjw_model.body_mass.numpy()
    assert sorted(solver_mass[1]) == pytest.approx(sorted([0.0, 1.0, 1.0, 8.0]))  # world 1, incl. its world body
    assert sorted(solver_mass[0]) == pytest.approx(sorted([0.0, 1.0, 1.0, 4.0]))
    others = np.arange(model.body_count) != torso_1
    assert np.array_equal(model.body_inv_inertia.numpy()[others], inv_inertia[others])
    assert np.array_equal(model.body_inv_mass.numpy()[others], inv_mass[others])
    assert model.body_inv_mass.numpy()[torso_1] == pytest.approx(1 / 8)


def test_contact_forces_name_the_same_bodies_as_mujoco(tmp_path):
    from metasim.queries import ContactForces

    # fixed torso; the long left leg reaches 5 cm into the ground, the short right one stays 15 cm above it
    xml = _WALKER.replace(
        'fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>\n      </body>\n      <body name="leg_r"',
        'fromto="0 0 0 0 0 -0.4" size="0.03" mass="1"/>\n      </body>\n      <body name="leg_r"',
    )
    xml = xml.replace(
        '<body name="leg_r" pos="0 -0.08 -0.1">\n        <joint name="hip_r" type="hinge" axis="0 1 0" range="-1 1"/>\n        <geom type="capsule" fromto="0 0 0 0 0 -0.3"',
        '<body name="leg_r" pos="0 -0.08 -0.1">\n        <joint name="hip_r" type="hinge" axis="0 1 0" range="-1 1"/>\n        <geom type="capsule" fromto="0 0 0 0 0 -0.2"',
    )
    assert "-0.4" in xml and "-0.2" in xml
    for sim in ("mujoco", "newton"):
        queries = {"contact_forces": ContactForces(history_length=1)}
        handler = _launch(tmp_path, sim, fix_base_link=True, queries=queries, xml=xml, height=0.48)
        try:
            _run(handler, 0.05)
            names = handler.get_body_names("walker")
            forces = handler.get_states(mode="tensor").extras["contact_forces"]["walker"].contact_forces[0]
            touching = {names[i] for i in torch.nonzero(forces.norm(dim=-1) > 0.1).flatten().tolist()}
            assert touching == {"leg_l"}, (sim, touching, forces)
        finally:
            handler.close()
