"""``scenario.lights`` reach the MuJoCo model as explicit MJCF lights.

Pure MJCF assembly (no rendering): the rules that produced a wrong picture when missed are pinned
here, and the model must still compile in MuJoCo.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("dm_control")
from dm_control import mjcf

from metasim.scenario.lights import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, SphereLightCfg
from metasim.sim.mujoco.lights import (
    AMBIENT_RATIO,
    AREA_INTENSITY_TO_DIFFUSE,
    DISTANT_INTENSITY_TO_DIFFUSE,
    add_scenario_lights,
)

pytestmark = pytest.mark.general

# +90° about +Y, (w, x, y, z): rotates local -Z (the UsdLux emission axis) onto -X
_ROT_MINUS_Z_TO_MINUS_X = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


def _compiled(model: mjcf.RootElement):
    import mujoco

    return mujoco.MjModel.from_xml_string(model.to_xml_string())


def test_no_lights_leaves_the_model_untouched():
    model = mjcf.RootElement()
    assert add_scenario_lights(model, []) == []
    assert model.worldbody.find_all("light") == []
    assert model.visual.headlight.diffuse is None  # MuJoCo default headlight stays


def test_every_light_writes_an_explicit_type():
    """An MJCF light without ``type`` is a 45° downward spot; a bulb is a hemispherical spot, never ``point``."""
    model = mjcf.RootElement()
    created = add_scenario_lights(
        model,
        [
            SphereLightCfg(name="bulb", pos=(0.0, 0.0, 2.0)),
            DiskLightCfg(name="panel", pos=(0.0, 0.0, 2.0)),
            DistantLightCfg(name="sun"),
            CylinderLightCfg(name="tube", pos=(1.0, 0.0, 2.0)),
        ],
    )
    types = {light.name: light.type for light in created}
    assert types == {"bulb": "spot", "panel": "spot", "sun": "directional", "tube": "spot"}
    assert all(light.type != "point" for light in created), "point lights render black in mujoco.Renderer"
    _compiled(model)


def test_disk_light_points_along_its_rotated_minus_z():
    """The same rig aimed at a workpiece in Isaac Sim must not point at the floor in MuJoCo."""
    model = mjcf.RootElement()
    (down,) = add_scenario_lights(model, [DiskLightCfg(name="down", pos=(0.0, 0.0, 2.0))])
    assert np.allclose(down.dir, [0.0, 0.0, -1.0])
    model = mjcf.RootElement()
    (sideways,) = add_scenario_lights(
        model, [DiskLightCfg(name="side", pos=(0.0, 0.0, 1.0), rot=_ROT_MINUS_Z_TO_MINUS_X)]
    )
    assert np.allclose(sideways.dir, [-1.0, 0.0, 0.0], atol=1e-6), sideways.dir
    assert sideways.cutoff == 90.0 and sideways.exponent == 0.0  # hemispherical, flat lobe


def test_distant_light_direction_follows_polar_and_azimuth():
    model = mjcf.RootElement()
    (straight_down,) = add_scenario_lights(model, [DistantLightCfg()])
    assert np.allclose(straight_down.dir, [0.0, 0.0, -1.0])
    model = mjcf.RootElement()
    (tilted,) = add_scenario_lights(model, [DistantLightCfg(polar=90.0)])
    assert abs(tilted.dir[2]) < 1e-6 and abs(np.linalg.norm(tilted.dir) - 1.0) < 1e-6


def test_area_lights_honour_normalize():
    """``normalize=False`` (UsdLux: intensity per unit area) scales the output with the emitting area."""
    unit = SphereLightCfg(name="a", radius=0.5, intensity=500.0)
    per_area = SphereLightCfg(name="b", radius=0.5, intensity=500.0, normalize=False)
    model = mjcf.RootElement()
    normalized, unnormalized = add_scenario_lights(model, [unit, per_area])
    assert np.allclose(normalized.diffuse, [500.0 * AREA_INTENSITY_TO_DIFFUSE] * 3)
    area = 4.0 * math.pi * 0.5**2
    assert np.allclose(unnormalized.diffuse, np.array(normalized.diffuse) * area)


def test_declared_rig_replaces_the_headlight_and_keeps_constant_attenuation():
    model = mjcf.RootElement()
    (sun,) = add_scenario_lights(model, [DistantLightCfg(intensity=2000.0, color=(1.0, 0.5, 0.25))])
    assert np.allclose(model.visual.headlight.diffuse, [0.0, 0.0, 0.0])
    assert np.allclose(sun.diffuse, np.array([1.0, 0.5, 0.25]) * 2000.0 * DISTANT_INTENSITY_TO_DIFFUSE)
    # the bounce-light stand-in is global ambient in the light's colour, not per-light ambient
    assert np.allclose(sun.ambient, [0.0, 0.0, 0.0])
    assert np.allclose(model.visual.headlight.ambient, np.array(sun.diffuse) * AMBIENT_RATIO)
    # the documented mismatch: MuJoCo keeps constant attenuation (its default) rather than UsdLux 1/d²
    assert sun.attenuation is None
    compiled = _compiled(model)
    assert compiled.nlight == 1


def test_declared_rig_deactivates_embedded_lights_but_keeps_scene_lights():
    """A robot MJCF that ships its own light must not add to the rig (USD assets have none); a light
    authored in the scene MJCF stays, as a scene USD's lights do on Isaac Sim."""
    model = mjcf.RootElement()
    robot = mjcf.RootElement(model="bot")
    robot.worldbody.add("light", name="top", pos=[0, 0, 2])
    model.attach(robot)
    model.worldbody.add("light", name="room", pos=[0, 0, 3])  # authored in the scene itself: stays
    (sun,) = add_scenario_lights(model, [DistantLightCfg(name="sun")])
    by_name = {light.full_identifier: light for light in model.find_all("light")}
    assert by_name["bot/top"].active == "false"
    assert by_name["room"].active is None and sun.active is None
    assert model.visual.headlight.active == 1
    compiled = _compiled(model)
    assert compiled.nlight == 3 and sum(int(a) for a in compiled.light_active) == 2


def test_dome_light_becomes_ambient_plus_zenith_light():
    model = mjcf.RootElement()
    (sky,) = add_scenario_lights(model, [DomeLightCfg(name="sky", intensity=1000.0, color=(0.5, 0.5, 1.0))])
    ambient = model.visual.headlight.ambient
    assert ambient[2] > ambient[0] > 0.0  # global ambient in the dome colour ...
    assert sky.type == "directional" and sky.castshadow == "false"  # ... plus a shadow-less zenith light
    assert np.allclose(sky.diffuse, ambient)
    assert np.allclose(model.visual.headlight.diffuse, [0.0, 0.0, 0.0])
    _compiled(model)


def test_unknown_light_type_is_rejected():
    class _Weird(DistantLightCfg.__mro__[1]):  # BaseLightCfg
        pass

    with pytest.raises(TypeError, match="unsupported light config"):
        add_scenario_lights(mjcf.RootElement(), [_Weird(name="x")])


def test_duplicate_rig_names_are_rejected_and_asset_clashes_renamed():
    with pytest.raises(ValueError, match=r"duplicate names \['panel'\]"):
        add_scenario_lights(mjcf.RootElement(), [DiskLightCfg(name="panel"), DiskLightCfg(name="panel", pos=(1, 0, 2))])
    model = mjcf.RootElement()
    model.worldbody.add("light", name="light1", pos=[0, 0, 2])  # a scene MJCF light: kept, name is taken
    (renamed,) = add_scenario_lights(model, [SphereLightCfg(name="light1", pos=(0, 0, 1))])
    assert renamed.name == "light1_scenario"
    compiled = _compiled(model)
    assert compiled.nlight == 2 and all(int(a) == 1 for a in compiled.light_active)  # scene light stays on
    # unnamed lights get a type-prefixed identifier; an explicit name that looks like one is not a duplicate
    model = mjcf.RootElement()
    a, b, c = add_scenario_lights(
        model,
        [
            SphereLightCfg(pos=(0, 0, 1)),
            SphereLightCfg(name="spherelight_2", pos=(1, 0, 1)),
            SphereLightCfg(pos=(2, 0, 1)),
        ],
    )
    assert (a.name, b.name, c.name) == ("spherelight_0", "spherelight_2", "spherelight_2_scenario")


def test_non_unit_rotation_is_normalised_and_garbage_rejected():
    model = mjcf.RootElement()
    (panel,) = add_scenario_lights(model, [DiskLightCfg(name="p", pos=(0, 0, 1), rot=(1.0, 1.0, 0.0, 0.0))])
    assert np.allclose(panel.dir, [0.0, 1.0, 0.0], atol=1e-6), panel.dir  # +90° about X, un-normalised input
    with pytest.raises(ValueError, match="not a usable"):
        add_scenario_lights(mjcf.RootElement(), [DiskLightCfg(name="q", rot=(0.0, 0.0, 0.0, 0.0))])


def test_saturation_and_light_count_are_warned_about(loguru_warnings):
    add_scenario_lights(
        mjcf.RootElement(), [DiskLightCfg(name="flood", intensity=20000.0, radius=1.2, pos=(0, 0, 4.5))]
    )
    add_scenario_lights(mjcf.RootElement(), [SphereLightCfg(name=f"bulb{i}", pos=(i, 0, 2)) for i in range(9)])
    text = "\n".join(loguru_warnings)
    assert "clip to white" in text and "flood" in text
    assert "evaluates at most 8" in text and "bulb7" in text


def test_flag_off_means_the_rig_is_not_consulted():
    """``SimParamCfg.mujoco_use_scenario_lights`` defaults to False: the default rig must not re-light scenes."""
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco.lights import scenario_lights_enabled

    assert scenario_lights_enabled(ScenarioCfg()) is False
    assert scenario_lights_enabled(ScenarioCfg(sim_params=SimParamCfg(mujoco_use_scenario_lights=True))) is True


def test_flag_on_with_empty_rig_warns_and_changes_nothing(loguru_warnings):
    model = mjcf.RootElement()
    assert add_scenario_lights(model, []) == []
    assert model.visual.headlight.diffuse is None
    assert any("scenario.lights is empty" in m for m in loguru_warnings)
