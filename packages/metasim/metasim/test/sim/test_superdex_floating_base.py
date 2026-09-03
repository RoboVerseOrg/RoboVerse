"""Floating-base articulation on SuperDex: root state must follow the body, not the spawn frame.

SuperDex's ``Actor.get_root_transform`` is the *articulation root frame*; with a FREE base joint the
base motion lives in the first six DoFs, so a handler that reports the root frame as the root state
shows the spawn pose forever (found in review). This suite builds a small two-link URDF with
explicit inertials, drops it, and checks that ``root_state`` tracks link 0 and that
``set_states`` with only ``dof_pos`` does not teleport the body back to its spawn pose.
"""

from __future__ import annotations

import os

import pytest
import torch

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

_URDF = """<?xml version="1.0"?>
<robot name="pendulum_box">
  <link name="base">
    <inertial><mass value="2.0"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><origin xyz="0 0 0.1"/><mass value="0.5"/><inertia ixx="0.002" iyy="0.002" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><box size="0.04 0.04 0.2"/></geometry></visual>
    <collision><origin xyz="0 0 0.1"/><geometry><box size="0.04 0.04 0.2"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
    <limit lower="-1.5" upper="1.5" effort="5" velocity="5"/>
  </joint>
</robot>
"""


def _scenario(tmp_path) -> ScenarioCfg:
    urdf = tmp_path / "pendulum_box.urdf"
    urdf.write_text(_URDF, encoding="utf-8")
    return ScenarioCfg(
        simulator="superdex",
        headless=True,
        num_envs=1,
        objects=[
            ArticulationObjCfg(name="pend", fix_base_link=False, urdf_path=str(urdf), default_position=(0.0, 0.0, 1.0))
        ],
        sim_params=SimParamCfg(dt=0.005),
        decimation=20,
    )


@pytest.fixture
def superdex_handler(tmp_path, monkeypatch):
    pytest.importorskip("superdex.physics", reason="superdex wheels (Python >= 3.12) not installed")
    from metasim.utils.setup_util import get_handler

    monkeypatch.setenv("METASIM_SUPERDEX_CACHE", str(tmp_path / "cache"))
    handler = get_handler(_scenario(tmp_path))
    yield handler
    handler.close()


@pytest.mark.superdex
def test_root_state_follows_a_falling_free_base(superdex_handler):
    handler = superdex_handler
    z0 = float(handler.get_states(mode="tensor").objects["pend"].root_state[0, 2])
    assert abs(z0 - 1.0) < 1e-3
    for _ in range(30):  # 3 s of fall + settle
        handler.simulate()
    state = handler.get_states(mode="tensor").objects["pend"]
    z_root = float(state.root_state[0, 2])
    z_base_link = float(state.body_state[0, state.body_names.index("base"), 2])
    assert z_root < 0.5, f"root_state.z stayed at {z_root}: the root frame is being reported instead of link 0"
    assert abs(z_root - z_base_link) < 1e-4, "root_state must equal the world pose of link 0"
    assert abs(z_base_link - 0.1) < 0.02, f"base should rest on the ground (half size 0.1), got z={z_base_link}"


@pytest.mark.superdex
def test_set_states_dof_only_keeps_the_body_where_it_is(superdex_handler):
    handler = superdex_handler
    for _ in range(30):
        handler.simulate()
    before = handler.get_states(mode="dict")[0]["objects"]["pend"]
    handler.set_states([{"objects": {"pend": {"dof_pos": {"hinge": 0.3}}}, "robots": {}}])
    after = handler.get_states(mode="dict")[0]["objects"]["pend"]
    assert torch.allclose(torch.as_tensor(after["pos"]), torch.as_tensor(before["pos"]), atol=1e-4), (
        "dof-only set_states teleported the base back to its spawn pose"
    )
    assert abs(float(after["dof_pos"]["hinge"]) - 0.3) < 1e-4
    # and an explicit pose still moves it
    handler.set_states([
        {"objects": {"pend": {"pos": torch.tensor([0.5, 0.0, 0.6]), "rot": torch.tensor([1.0, 0, 0, 0])}}, "robots": {}}
    ])
    moved = handler.get_states(mode="dict")[0]["objects"]["pend"]
    assert torch.allclose(torch.as_tensor(moved["pos"]), torch.tensor([0.5, 0.0, 0.6]), atol=1e-4)


@pytest.mark.superdex
def test_free_base_without_inertials_fails_fast(tmp_path, monkeypatch):
    pytest.importorskip("superdex.physics", reason="superdex wheels (Python >= 3.12) not installed")
    from metasim.utils.setup_util import get_handler

    monkeypatch.setenv("METASIM_SUPERDEX_CACHE", str(tmp_path / "cache"))
    urdf = tmp_path / "massless.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="massless">
  <link name="base"><inertial><mass value="1"/></inertial><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="tip"><collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>
  <joint name="j" type="revolute"><parent link="base"/><child link="tip"/><origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    scenario = ScenarioCfg(
        simulator="superdex",
        headless=True,
        objects=[ArticulationObjCfg(name="m", fix_base_link=False, urdf_path=str(urdf), default_position=(0, 0, 1.0))],
    )
    with pytest.raises(ValueError, match="free base"):
        get_handler(scenario)
    assert os.path.isdir(tmp_path / "cache")
