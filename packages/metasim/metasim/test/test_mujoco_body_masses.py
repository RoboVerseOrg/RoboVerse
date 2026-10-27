"""MuJoCo ``set_body_masses`` changes what the engine simulates, for the named body only.

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
