"""End to end on the MuJoCo handler: the opt-in light rig replaces the asset light and changes the frame."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

pytestmark = pytest.mark.mujoco


def _renderer_available() -> bool:
    import mujoco

    try:
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 8, 8)
        r.close()
        return True
    except Exception:
        return False


def _frame(lights, use_rig: bool):
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler

    scenario = ScenarioCfg(
        robots=[FrankaCfg()],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[0.8, 0.1, 0.1],
                default_position=[0.3, -0.2, 0.05],
                physics=PhysicStateType.RIGIDBODY,
            )
        ],
        cameras=[PinholeCameraCfg(name="cam", width=96, height=96, pos=(1.5, -1.5, 1.2), look_at=(0.2, -0.2, 0.2))],
        lights=lights,
        sim_params=SimParamCfg(mujoco_use_scenario_lights=use_rig),
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    h = MujocoHandler(scenario)
    h.launch()
    try:
        h.simulate()
        rgb = h.get_states(mode="tensor").cameras["cam"].rgb[0].cpu().numpy().astype(np.float32)
        return rgb, int(h._mj_model.nlight), [int(a) for a in h._mj_model.light_active]
    finally:
        h.close()


def test_mujoco_scenario_light_rig_is_opt_in_and_reaches_the_frame():
    if not _renderer_available():
        pytest.skip("no MuJoCo offscreen renderer in this environment")
    rig = [DistantLightCfg(name="sun", intensity=2000.0, polar=30.0, azimuth=45.0)]
    default_rgb, n_default, active_default = _frame(rig, use_rig=False)
    assert sum(active_default) == n_default  # flag off: asset lights untouched, rig not added
    rig_rgb, n_rig, active_rig = _frame(rig, use_rig=True)
    assert n_rig == n_default + 1 and sum(active_rig) == 1, (n_default, active_default, n_rig, active_rig)
    assert np.abs(rig_rgb - default_rgb).mean() > 5.0, "the declared rig did not change the frame"
