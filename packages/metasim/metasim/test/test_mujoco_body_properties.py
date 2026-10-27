"""MuJoCo body properties: ``set_body_masses`` and ``set_body_friction`` change what the engine simulates.

Friction: a box slid across a frictionless slab stops after ``v^2 / (2 mu g)``; the coefficient lands on the
named body's collision geoms only, a round trip leaves the model bit-identical, and restitution (which MuJoCo
does not model) warns and changes nothing.

Masses: ``set_body_masses`` changes what the engine simulates, for the named body only.

Content used to write ``physics.model.body_mass[i]`` with ``i`` an index into the *object's* body list,
which on a scene with more than one model is another body (the world body for a robot's first link),
and to rescale ``body_inertia`` from its current value each call. These pin the handler-level setter:
the right body, inertia scaled by the mass ratio, derived constants refreshed, and a write that restores
the defaults restores the trajectory exactly -- also through the multi-env ``ParallelHandler``.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.mujoco

# a spring-loaded hinge under gravity: the deflection grows with the link's mass
_ARM = """
<mujoco model="arm">
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="link" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0" stiffness="5" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _scenario(tmp_path, num_envs=1):
    from metasim.scenario.objects import ArticulationObjCfg
    from metasim.scenario.scenario import ScenarioCfg

    path = tmp_path / "arm.xml"
    path.write_text(textwrap.dedent(_ARM).strip() + "\n")
    return ScenarioCfg(
        objects=[ArticulationObjCfg(name="arm", mjcf_path=str(path), fix_base_link=True)],
        robots=[],
        simulator="mujoco",
        num_envs=num_envs,
        headless=True,
    )


def _hinge_after(handler, steps=150):
    handler.set_states(handler._initial)
    for _ in range(steps):
        handler.simulate()
    return handler.get_states(mode="tensor").objects["arm"].joint_pos[:, 0].clone()


@pytest.fixture
def arm(tmp_path):
    from metasim.utils.setup_util import get_handler

    handler = get_handler(_scenario(tmp_path))
    handler._initial = handler.get_states(mode="tensor")
    yield handler
    handler.close()


def test_writes_the_named_body_and_scales_its_inertia(arm):
    model = arm.physics.model.ptr
    fields = ["body_mass", "body_inertia", "body_subtreemass", "dof_invweight0", "body_invweight0"]
    before = {f: getattr(model, f).copy() for f in fields}
    assert arm.get_body_names("arm") == ["base", "link"]
    masses = arm.get_body_masses("arm")
    torch.testing.assert_close(masses, torch.tensor([[1.0, 0.5]]))

    arm.set_body_masses("arm", masses)  # written back unchanged: the model is bit-for-bit the same
    assert all(np.array_equal(before[f], getattr(model, f)) for f in fields)

    arm.set_body_masses("arm", torch.tensor([[1.0, 1.25]]))
    link = model.body("arm/link").id
    assert model.body_mass[link] == 1.25
    np.testing.assert_allclose(model.body_inertia[link], before["body_inertia"][link] * 2.5)
    assert model.body_subtreemass[model.body("arm/base").id] == pytest.approx(2.25)
    assert model.dof_invweight0[0] < before["dof_invweight0"][0]  # mj_setConst ran
    others = np.arange(model.nbody) != link
    assert np.array_equal(model.body_mass[others], before["body_mass"][others])

    arm.set_body_masses("arm", torch.tensor([[1.0, 1.25]]))  # same values again: no further scaling
    np.testing.assert_allclose(model.body_inertia[link], before["body_inertia"][link] * 2.5)


def test_a_heavier_link_sags_further_and_the_default_comes_back(arm):
    default = _hinge_after(arm)
    masses = arm.get_body_masses("arm")
    heavy = masses.clone()
    heavy[0, 1] *= 3
    arm.set_body_masses("arm", heavy)
    assert abs(float(_hinge_after(arm))) > 1.5 * abs(float(default))
    arm.set_body_masses("arm", masses)
    assert torch.equal(_hinge_after(arm), default)


def test_parallel_envs_are_written_one_by_one(tmp_path):
    from metasim.utils.setup_util import get_handler

    handler = get_handler(_scenario(tmp_path, num_envs=2))
    try:
        handler._initial = handler.get_states(mode="tensor")
        default = _hinge_after(handler)
        masses = handler.get_body_masses("arm")
        assert masses.shape == (2, 2)
        handler.set_body_masses("arm", torch.tensor([[1.0, 1.5]]), env_ids=[1])
        torch.testing.assert_close(handler.get_body_masses("arm"), torch.tensor([[1.0, 0.5], [1.0, 1.5]]))
        after = _hinge_after(handler)
        assert after[0] == default[0] and abs(float(after[1])) > 1.5 * abs(float(default[1]))
    finally:
        handler.close()


# a frictionless fixed slab and a free box on it; the box's "handle" body has no collision geom
_SLAB = '<mujoco model="slab"><worldbody><body name="slab"><geom type="box" size="5 5 0.05" friction="0 0 0"/></body></worldbody></mujoco>'
_BOX = """<mujoco model="box"><worldbody><body name="block">
<geom type="box" size="0.1 0.1 0.1" mass="1" friction="1 0.005 0.0001"/>
<body name="handle" pos="0 0 0.2"><geom type="sphere" size="0.01" mass="0.01" contype="0" conaffinity="0"/></body>
</body></worldbody></mujoco>"""
_G, _V = 9.81, 2.0


def _slide_scenario(tmp_path, num_envs=1, simulator="mujoco"):
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg

    (tmp_path / "slab.xml").write_text(_SLAB)
    (tmp_path / "box.xml").write_text(_BOX)
    return ScenarioCfg(
        robots=[
            RobotCfg(name="slab", mjcf_path=str(tmp_path / "slab.xml"), fix_base_link=True, actuators={}),
            RobotCfg(
                name="box",
                mjcf_path=str(tmp_path / "box.xml"),
                fix_base_link=False,
                default_position=(0.0, 0.0, 0.152),
                actuators={},
            ),
        ],
        objects=[],
        simulator=simulator,
        num_envs=num_envs,
        headless=True,
    )


def _slide(handler, seconds=1.5):
    """Distance the box slides from rest-on-the-slab after a 2 m/s push along x."""
    for _ in range(50):
        handler.simulate()
    states = handler.get_states(mode="tensor")
    x0 = states.robots["box"].root_state[:, 0].clone()
    states.robots["box"].root_state[:, 7] = _V
    handler.set_states(states)
    for _ in range(int(seconds / handler.physics_dt)):
        handler.simulate()
    return (handler.get_states(mode="tensor").robots["box"].root_state[:, 0] - x0).cpu()


def test_friction_sets_the_stopping_distance_and_round_trips_exactly(tmp_path):
    from metasim.utils.log import reset_warn_once
    from metasim.utils.setup_util import get_handler

    reset_warn_once()
    handler = get_handler(_slide_scenario(tmp_path))
    try:
        model = handler.physics.model.ptr
        friction = handler.get_body_friction("box")
        assert handler.get_body_names("box") == ["block", "handle"]
        assert friction[0, 0] == 1.0 and torch.isnan(friction[0, 1])  # handle: no collision geom
        before = model.geom_friction.copy()
        handler.set_body_friction("box", friction)
        assert np.array_equal(model.geom_friction, before)

        friction[0, 0] = 0.25
        handler.set_body_friction("box", friction)
        block = model.geom(model.body("box/block").geomadr[0]).id
        assert model.geom_friction[block, 0] == 0.25
        assert np.array_equal(model.geom_friction[block, 1:], before[block, 1:]), "torsion / roll keep the asset's"
        others = np.arange(model.ngeom) != block
        assert np.array_equal(model.geom_friction[others], before[others])

        assert torch.isnan(handler.get_body_restitution("box")).all()
        handler.set_body_restitution("box", torch.zeros(1, 2))  # MuJoCo has none: warns, writes nothing
        assert float(_slide(handler)[0]) == pytest.approx(_V**2 / (2 * 0.25 * _G), rel=0.01)
    finally:
        handler.close()


def test_friction_is_per_env_through_the_parallel_handler(tmp_path):
    from metasim.utils.setup_util import get_handler

    handler = get_handler(_slide_scenario(tmp_path, num_envs=2))
    try:
        friction = handler.get_body_friction("box")
        handler.set_body_friction("box", torch.tensor([[0.5, float("nan")]]), env_ids=[1])
        friction[1, 0] = 0.5
        torch.testing.assert_close(handler.get_body_friction("box"), friction, equal_nan=True)
        handler.set_body_friction("box", torch.tensor([[0.25, float("nan")]]), env_ids=[0])
        dist = _slide(handler)
        assert dist.tolist() == pytest.approx([_V**2 / (2 * mu * _G) for mu in (0.25, 0.5)], rel=0.01)
    finally:
        handler.close()
