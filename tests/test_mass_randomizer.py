"""``MassRandomizer`` randomizes the body it names, around its default, on the handler's mass interface.

On MuJoCo it used to index ``model.body_mass`` with the body's position in the object's own list, so
``body_names="pelvis"`` on a G1 scene added mass to the world body and left the pelvis alone, and
recomputed the inertia from 0/0. It now goes through ``set_body_masses``.
"""

from __future__ import annotations

import textwrap

import pytest

_ARM = """
<mujoco model="arm">
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="link" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def arm(tmp_path):
    from metasim.scenario.objects import ArticulationObjCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.utils.setup_util import get_handler

    path = tmp_path / "arm.xml"
    path.write_text(textwrap.dedent(_ARM).strip() + "\n")
    scenario = ScenarioCfg(
        objects=[ArticulationObjCfg(name="arm", mjcf_path=str(path), fix_base_link=True)],
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    handler = get_handler(scenario)
    yield handler
    handler.close()


@pytest.mark.requires_optional("mujoco")
def test_randomizes_the_named_body_around_its_default(arm):
    from roboverse_pack.randomization.humanoid import MassRandomizer

    model = arm.physics.model.ptr
    world_mass, base_mass = float(model.body_mass[0]), float(model.body_mass[model.body("arm/base").id])
    link_inertia = model.body_inertia[model.body("arm/link").id].copy()
    rand = MassRandomizer(obj_name="arm", body_names="link", mass_distribution_params=(1.0, 2.0), operation="add")
    rand.bind_handler(arm)
    for _ in range(5):
        rand()
        link = float(arm.get_body_masses("arm")[0, 1])
        assert 1.5 <= link <= 2.5  # default 0.5 plus U(1, 2), not accumulated over calls
        ratio = model.body_inertia[model.body("arm/link").id] / link_inertia
        assert ratio == pytest.approx([link / 0.5] * 3, rel=1e-5)
    assert float(model.body_mass[0]) == world_mass
    assert float(model.body_mass[model.body("arm/base").id]) == base_mass


@pytest.mark.requires_optional("mujoco")
def test_an_unknown_body_name_fails_at_bind(arm):
    from roboverse_pack.randomization.humanoid import MassRandomizer

    with pytest.raises(ValueError, match=r"has no bodies \['pelvis'\]"):
        MassRandomizer(obj_name="arm", body_names="pelvis").bind_handler(arm)


def test_beyondmimic_uses_the_same_randomizers():
    from roboverse_pack.randomization import humanoid
    from roboverse_pack.tasks.beyondmimic.metasim.configs import cfg_randomizers

    assert cfg_randomizers.MassRandomizer is humanoid.MassRandomizer
    assert cfg_randomizers.MaterialRandomizer is humanoid.MaterialRandomizer
    assert cfg_randomizers.randomize_prop_by_op is humanoid.randomize_prop_by_op
