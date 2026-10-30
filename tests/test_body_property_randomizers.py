"""The body-property randomizers run on the handler's body-property interface, on a real MuJoCo scene.

``MaterialRandomizer`` used to branch per backend: on MuJoCo it wrote "static friction" into ``solimp`` and
indexed geoms with object-local offsets, on Newton it never told the solver. It now draws one friction per
body from its buckets and writes it with ``set_body_friction``; restitution on MuJoCo warns.
mjlab's ``body_mass`` DR now goes through ``set_body_masses`` as well.

``MassRandomizer`` randomizes the body it names, around its default, on the handler's mass interface.

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
        <body name="tip" pos="0.3 0 0"/>
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


@pytest.mark.requires_optional("mujoco")
def test_a_range_that_can_go_negative_fails_at_bind(arm):
    from roboverse_pack.randomization.humanoid import MassRandomizer

    with pytest.raises(ValueError, match="negative mass"):
        MassRandomizer(obj_name="arm", body_names="link", mass_distribution_params=(-1.0, 3.0)).bind_handler(arm)


@pytest.mark.requires_optional("mujoco")
def test_scaling_every_body_leaves_a_massless_one_massless(arm):
    from roboverse_pack.randomization.humanoid import MassRandomizer

    assert arm.get_body_names("arm") == ["base", "link", "tip"]
    rand = MassRandomizer(obj_name="arm", mass_distribution_params=(0.8, 1.2), operation="scale")
    rand.bind_handler(arm)
    rand()
    masses = arm.get_body_masses("arm")[0]
    assert 0.8 <= float(masses[0]) <= 1.2 and 0.4 <= float(masses[1]) <= 0.6 and float(masses[2]) == 0.0


@pytest.mark.requires_optional("mujoco")
def test_material_randomizer_draws_bucket_friction_per_body(arm):
    import torch

    from metasim.utils.log import reset_warn_once
    from roboverse_pack.randomization.humanoid import MaterialRandomizer

    reset_warn_once()
    model = arm.physics.model.ptr
    solimp, solref = model.geom_solimp.copy(), model.geom_solref.copy()
    rand = MaterialRandomizer(obj_name="arm", friction_range=(0.2, 0.4), restitution_range=(0.0, 0.5), num_buckets=4)
    rand.bind_handler(arm)
    assert rand.set_body_ids.tolist() == [0, 1], "tip has no collision shape and is skipped"
    seen = set()
    for _ in range(10):
        rand()
        friction = arm.get_body_friction("arm")[0]
        assert torch.isnan(friction[2])
        assert all(0.2 <= float(v) <= 0.4 and float(v) in rand.buckets["friction"].tolist() for v in friction[:2])
        seen.update(friction[:2].tolist())
    assert 1 < len(seen) <= 4
    assert (model.geom_solimp == solimp).all() and (model.geom_solref == solref).all(), "contact softness untouched"


@pytest.mark.requires_optional("mujoco")
def test_material_randomizer_rejects_what_it_cannot_do(arm):
    from roboverse_pack.randomization.humanoid import MaterialRandomizer

    with pytest.raises(ValueError, match="no collision shape"):
        MaterialRandomizer(obj_name="arm", body_names="tip", friction_range=(0.5, 1.0)).bind_handler(arm)
    with pytest.raises(ValueError, match="restitution_range"):
        MaterialRandomizer(obj_name="arm", restitution_range=(0.0, 1.5))
    with pytest.raises(ValueError, match="friction_range"):
        MaterialRandomizer(obj_name="arm", friction_range=(1.0, 0.5))
    legacy = MaterialRandomizer(obj_name="arm", static_friction_range=(0.3, 1.6), dynamic_friction_range=(0.3, 1.2))
    assert legacy.friction_range == (0.3, 1.2)


@pytest.mark.requires_optional("mujoco")
def test_mjlab_body_mass_scales_the_assets_own_body(arm):
    from types import SimpleNamespace

    from roboverse_pack.tasks.mjlab.mdp.events_dr import body_mass
    from roboverse_pack.tasks.mjlab.mdp.scene_entity import SceneEntityCfg

    model = arm.physics.model.ptr
    world, base, link = 0, model.body("arm/base").id, model.body("arm/link").id
    before = model.body_mass.copy()
    invweight = model.dof_invweight0.copy()
    env = SimpleNamespace(handler=arm, scenario=arm.scenario, num_envs=1)
    body_mass(env, asset_cfg=SceneEntityCfg("arm", body_names=("link",)), operation="mul", ranges=(2.0, 2.0))
    assert model.body_mass[link] == pytest.approx(2 * before[link])
    assert model.body_mass[world] == before[world] and model.body_mass[base] == before[base]
    assert model.dof_invweight0[0] < invweight[0], "mj_setConst ran"


@pytest.mark.requires_optional("mujoco")
def test_mjlab_body_mass_gives_a_massless_body_mass_without_scaling_its_inertia(arm):
    from types import SimpleNamespace

    from roboverse_pack.tasks.mjlab.mdp.events_dr import body_mass
    from roboverse_pack.tasks.mjlab.mdp.scene_entity import SceneEntityCfg

    model = arm.physics.model.ptr
    tip, link = model.body("arm/tip").id, model.body("arm/link").id
    assert model.body_mass[tip] == 0.0
    link_mass, link_inertia = float(model.body_mass[link]), model.body_inertia[link].copy()
    env = SimpleNamespace(handler=arm, scenario=arm.scenario, num_envs=1)
    body_mass(env, asset_cfg=SceneEntityCfg("arm"), operation="add", ranges=(0.5, 0.5))  # every body, the tip too
    assert model.body_mass[tip] == pytest.approx(0.5)
    assert model.body_mass[link] == pytest.approx(link_mass + 0.5)
    import numpy as np

    np.testing.assert_allclose(model.body_inertia[link], link_inertia * (link_mass + 0.5) / link_mass)
