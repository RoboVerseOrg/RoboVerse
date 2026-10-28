"""Primitive objects weigh ``cfg.mass`` and have the config's full size on MuJoCo, MJX's MJCF, and Newton.

MuJoCo and Newton derived a primitive's mass from their default density (1000 kg/m^3), so a 10 cm cube of
``mass=0.3`` weighed 1 kg there and 0.3 kg on Isaac Sim, PyBullet and SAPIEN. MuJoCo's cylinder ``size`` is
the half height, and the handler passed the full height, so cylinders were twice as tall; Newton built a
capsule. A settled cylinder now rests on its centre at ``height / 2`` on both.
"""

from __future__ import annotations

import pytest

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg

_R = PhysicStateType.RIGIDBODY


def _objects():
    return [
        PrimitiveCubeCfg(
            name="cube", size=(0.1, 0.1, 0.1), mass=0.3, color=[1, 0, 0], physics=_R, default_position=(0, 0, 0.06)
        ),
        PrimitiveSphereCfg(
            name="ball", radius=0.05, mass=0.2, color=[0, 1, 0], physics=_R, default_position=(0.5, 0, 0.06)
        ),
        PrimitiveCylinderCfg(
            name="can", radius=0.05, height=0.2, mass=0.4, color=[0, 0, 1], physics=_R, default_position=(1, 0, 0.11)
        ),
    ]


def test_primitive_xml_carries_mass_and_half_height():
    import xml.etree.ElementTree as ET

    from metasim.sim.mujoco.primitives import primitive_xml

    cube, ball, can = _objects()
    geoms = {o.name: ET.fromstring(primitive_xml(o)).find(".//geom").attrib for o in (cube, ball, can)}
    assert geoms["cube"]["size"] == "0.05 0.05 0.05" and geoms["cube"]["mass"] == "0.3"
    assert geoms["ball"]["size"] == "0.05" and geoms["ball"]["mass"] == "0.2"
    assert geoms["can"]["type"] == "cylinder" and geoms["can"]["size"] == "0.05 0.1"


def _masses(handler, sim, obj):
    if sim == "mujoco":
        model = handler.physics.model
        prefix = handler.mj_objects[obj].model + "/"
        return sum(model.body_mass[i] for i in range(model.nbody) if model.body(i).name.startswith(prefix))
    return float(handler._model.body_mass.numpy()[handler._get_body_indices(0, obj)].sum())


@pytest.mark.parametrize("sim", [pytest.param("mujoco", marks=pytest.mark.mujoco), "newton"])
def test_primitives_weigh_their_mass_and_rest_at_their_half_height(sim):
    if sim == "newton":
        pytest.importorskip("metasim.sim.newton.newton")
    from metasim.utils.setup_util import get_handler

    objects = _objects()
    handler = get_handler(ScenarioCfg(robots=[], objects=objects, simulator=sim, num_envs=1, headless=True))
    try:
        for obj in objects:
            assert _masses(handler, sim, obj.name) == pytest.approx(obj.mass, rel=1e-5), obj.name
        for _ in range(400):
            handler.simulate()
        state = handler.get_states(mode="tensor")
        for obj, half in zip(objects, (0.05, 0.05, 0.1), strict=True):
            assert state.objects[obj.name].root_state[0, 2].item() == pytest.approx(half, abs=2e-3), obj.name
    finally:
        handler.close()
