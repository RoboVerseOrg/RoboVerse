"""Portable recipes are deterministic, validated and isolated from caller RNGs."""

from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.randomization import (
    EnvironmentRandomCfg,
    LightingRandomCfg,
    SensorRandomCfg,
    SurfaceRandomCfg,
    TextureSetCfg,
    ViewRandomCfg,
    VisualRandomizationCfg,
    VisualRandomizer,
    VisualRecipe,
    color_temperature_to_rgb,
)
from metasim.randomization.visual import _distort, _fit_geometry, _sensor_geometry, _undistort
from metasim.scenario.render import RenderCfg

pytestmark = pytest.mark.general


def _downgrade(recipe, version):
    """Strip the fields later schema versions added, as an older writer would have serialized."""
    recipe = copy.deepcopy(recipe)
    recipe.schema_version = version
    recipe.sensors = {}
    for material in recipe.materials.values():
        del material["uv_projection"]
        del material["textures"]["orm"]
        if version < 2:
            for name in ("uv_scale", "uv_rotation", "ior"):
                del material[name]
    for light in recipe.lights.values():
        del light["color_temperature"]
    for camera in recipe.cameras.values():
        del camera["f_stop"]
        del camera["focus_scale"]
    return recipe


def _identity_sensor():
    return {
        "exposure_ev": 0.0,
        "white_balance": [1.0, 1.0],
        "vignetting": 0.0,
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "blur_sigma": 0.0,
        "shot_noise": 0.0,
        "read_noise": 0.0,
    }


@pytest.mark.parametrize("sim,num_envs", [("blender", 1), ("isaacsim", 2)])
def test_visual_render_fixture_constructs_valid_primitives(sim, num_envs):
    """Invalid scene configs must fail independently of backend availability."""
    from metasim.test.randomization.conftest import get_visual_render_scenario

    scenario = get_visual_render_scenario(sim, num_envs)
    assert scenario.objects[0].name == "cube"
    assert scenario.objects[0].color == [0.6, 0.2, 0.1]
    assert scenario.num_envs == num_envs


def _cfg():
    return VisualRandomizationCfg(
        materials={"cube": SurfaceRandomCfg()},
        lights={"key": LightingRandomCfg()},
        cameras={"view": ViewRandomCfg()},
        environment=EnvironmentRandomCfg(),
        sensors={"view": SensorRandomCfg()},
    )


def test_order_sharding_and_added_targets_preserve_named_streams():
    sampler = VisualRandomizer(_cfg(), seed=0)
    serial = {i: sampler.sample(sample_id=i).to_json() for i in range(12)}
    sharded = {}
    for worker in range(3):
        independent = VisualRandomizer(_cfg(), seed=0)
        for i in reversed(range(worker, 12, 3)):
            sharded[i] = independent.sample(sample_id=i).to_json()
    assert sharded == serial
    cfg = _cfg()
    cfg.materials["sphere"] = SurfaceRandomCfg()
    augmented = VisualRandomizer(cfg, seed=0).sample(sample_id=0)
    original = sampler.sample(sample_id=0)
    assert augmented.materials["cube"] == original.materials["cube"]
    assert augmented.lights == original.lights
    assert sampler.sample(sample_id=0, variant_id=1).to_json() != serial[0]
    assert VisualRandomizer(_cfg(), seed=1).sample(sample_id=0).to_json() != serial[0]


def test_sampling_and_construction_preserve_global_rngs():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    sampler = VisualRandomizer(_cfg(), seed=0)
    sampler.sample(sample_id=3)
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_recipe_round_trip_and_config_copy():
    cfg = _cfg()
    sampler = VisualRandomizer(cfg)
    original = sampler.sample(sample_id=4)
    cfg.materials["cube"].roughness = (0, 0)
    assert sampler.sample(sample_id=4) == original
    assert VisualRecipe.from_json(original.to_json()) == original


def test_material_details_and_legacy_recipe_replay():
    cfg = _cfg()
    palette = ((0.2, 0.1, 0.05), (0.7, 0.6, 0.5))
    cfg.materials["cube"] = SurfaceRandomCfg(
        color_palette=palette, uv_scale=((2, 3), (4, 5)), uv_rotation=(-1, 1), ior=(1.3, 1.6)
    )
    sampler = VisualRandomizer(cfg)
    for index in range(12):
        recipe = sampler.sample(sample_id=index)
        material = recipe.materials["cube"]
        assert tuple(material["color"]) in palette
        assert 2 <= material["uv_scale"][0] <= 3
        assert 4 <= material["uv_scale"][1] <= 5
        assert -1 <= material["uv_rotation"] <= 1
        assert 1.3 <= material["ior"] <= 1.6
        assert recipe.schema_version == 3
        assert VisualRecipe.from_json(recipe.to_json()) == recipe
    for version in (1, 2):
        legacy = _downgrade(recipe, version)
        assert VisualRecipe.from_json(legacy.to_json()) == legacy
        # An older writer never serialized the sensor stage at all.
        data = json.loads(legacy.to_json())
        del data["sensors"]
        assert VisualRecipe.from_json(json.dumps(data)) == legacy
        legacy.sensors = {"view": _identity_sensor()}
        with pytest.raises(ValueError, match="schema version 3"):
            legacy.validate()


def test_new_streams_leave_earlier_schema_values_unchanged():
    """Adding v3 fields must not change the v2 values a seed already produced."""
    recipe = VisualRandomizer(_cfg(), seed=7).sample(sample_id=3)
    assert recipe.materials["cube"]["color"] == pytest.approx([
        0.41845285502820123,
        0.416529575444393,
        0.748529973827197,
    ])
    assert recipe.cameras["view"]["focal_scale"] == pytest.approx(0.9958995221592183)
    assert recipe.cameras["view"]["f_stop"] == 0.0
    assert recipe.lights["key"]["color_temperature"] is None
    assert recipe.materials["cube"]["uv_projection"] == "mesh"


def test_color_temperature_realizes_planckian_light_color():
    assert all(0.9 <= v <= 1.0 for v in color_temperature_to_rgb(6500))
    warm, cool = color_temperature_to_rgb(2700), color_temperature_to_rgb(12000)
    assert warm[0] == 1.0 and warm[2] < 0.5
    assert cool[2] == 1.0 and cool[0] < 0.9
    with pytest.raises(ValueError):
        color_temperature_to_rgb(1000)
    cfg = _cfg()
    cfg.lights["key"] = LightingRandomCfg(color_temperature=(3000, 3000))
    recipe = VisualRandomizer(cfg).sample(sample_id=0)
    assert recipe.lights["key"]["color_temperature"] == pytest.approx(3000)
    assert recipe.lights["key"]["color"] == pytest.approx(list(color_temperature_to_rgb(3000)))
    with pytest.raises(ValueError):
        LightingRandomCfg(color_temperature=(500, 3000))


def test_one_kelvin_converter_serves_recipes_and_legacy_randomizers():
    """A Kelvin value must mean the same colour in a recipe and in the legacy light randomizer."""
    from metasim.randomization.light_randomizer import LightRandomizer

    convert = LightRandomizer._temperature_to_rgb
    for kelvin in (2700, 3000, 5000, 6500, 12000):
        assert convert(None, kelvin) == pytest.approx(color_temperature_to_rgb(kelvin))
    # The legacy API accepted 1000..40000 K; out-of-range values clamp instead of raising.
    assert convert(None, 500) == pytest.approx(color_temperature_to_rgb(1667))
    assert convert(None, 40000) == pytest.approx(color_temperature_to_rgb(25000))


def test_depth_of_field_and_projection_are_sampled_and_validated():
    cfg = _cfg()
    cfg.cameras["view"] = ViewRandomCfg(f_stop=(2.8, 5.6), focus_scale=(0.9, 1.1))
    cfg.materials["cube"] = SurfaceRandomCfg(uv_projection="box")
    recipe = VisualRandomizer(cfg).sample(sample_id=0)
    assert 2.8 <= recipe.cameras["view"]["f_stop"] <= 5.6
    assert 0.9 <= recipe.cameras["view"]["focus_scale"] <= 1.1
    assert recipe.materials["cube"]["uv_projection"] == "box"
    recipe.cameras["view"]["f_stop"] = -1
    with pytest.raises(ValueError):
        recipe.validate()
    with pytest.raises(ValueError):
        ViewRandomCfg(focus_scale=(0, 1))
    with pytest.raises(ValueError):
        SurfaceRandomCfg(uv_projection="sphere")


def test_packed_orm_excludes_separate_scalar_maps(tmp_path):
    texture = tmp_path / "orm.png"
    texture.touch()
    with pytest.raises(ValueError, match="orm"):
        VisualRandomizer(
            VisualRandomizationCfg(
                materials={
                    "cube": SurfaceRandomCfg(textures=(TextureSetCfg(orm=str(texture), roughness=str(texture)),))
                }
            )
        )
    recipe = VisualRandomizer(
        VisualRandomizationCfg(materials={"cube": SurfaceRandomCfg(textures=(TextureSetCfg(orm=str(texture)),))})
    ).sample(sample_id=0)
    assert recipe.materials["cube"]["textures"]["orm"] == str(texture.resolve())
    assert VisualRecipe.from_json(recipe.to_json()) == recipe


def test_sensor_stage_is_identity_without_effects_and_reproducible_per_frame():
    recipe = VisualRandomizer(_cfg()).sample(sample_id=0)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    intrinsics = np.array([[70.0, 0.0, 32.0], [0.0, 70.0, 24.0], [0.0, 0.0, 1.0]])
    identity = copy.deepcopy(recipe)
    identity.sensors["view"] = _identity_sensor()
    capture = identity.apply_sensor("view", image, intrinsics)
    np.testing.assert_array_equal(capture.rgb, image)
    assert capture.intrinsics == intrinsics.tolist()
    assert capture.distortion == [0.0] * 5 and capture.model == "opencv_pinhole"
    first = recipe.apply_sensor("view", torch.from_numpy(image), torch.from_numpy(intrinsics))
    assert first.rgb.shape == image.shape and first.rgb.dtype == np.uint8
    np.testing.assert_array_equal(first.rgb, recipe.apply_sensor("view", image, intrinsics).rgb)
    assert not np.array_equal(first.rgb, recipe.apply_sensor("view", image, intrinsics, frame=1).rgb)
    assert first.distortion[:4] == recipe.sensors["view"]["distortion"] and first.distortion[4] == 0.0
    for camera, rgb, matrix in [
        ("other", image, intrinsics),
        ("view", image.astype(np.float32), intrinsics),
        ("view", image[..., :1], intrinsics),
        ("view", image, intrinsics[:2]),
        ("view", image, intrinsics * np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]])),
        ("view", image, intrinsics * np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]])),
    ]:
        with pytest.raises(ValueError):
            recipe.apply_sensor(camera, rgb, matrix)
    huge = copy.deepcopy(identity)
    huge.sensors["view"]["blur_sigma"] = 40.0
    with pytest.raises(ValueError, match="blur_sigma"):
        huge.apply_sensor("view", image, intrinsics)


def test_distortion_keeps_every_output_pixel_inside_the_render():
    x, y = np.meshgrid(np.linspace(-0.6, 0.6, 9), np.linspace(-0.45, 0.45, 7))
    coefficients = [-0.12, 0.01, 0.001, -0.002]
    xu, yu = _undistort(*_distort(x, y, coefficients), coefficients)
    assert np.abs(xu - x).max() < 1e-9 and np.abs(yu - y).max() < 1e-9
    intrinsics = np.array([[80.0, 0.0, 48.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]])
    barrel, su, sv, _, _ = _sensor_geometry(intrinsics, 64, 96, [-0.3, 0.0, 0.0, 0.0])
    assert barrel[0, 0] > intrinsics[0, 0] and barrel[0, 2] == intrinsics[0, 2]
    # The unclipped fit must already be inside the render, and tight: a slightly smaller zoom overflows.
    zoom, (raw_u, raw_v, _, _) = _fit_geometry(intrinsics, 64, 96, (-0.3, 0.0, 0.0, 0.0))
    assert 0 <= raw_u.min() and raw_u.max() <= 95 and 0 <= raw_v.min() and raw_v.max() <= 63
    assert zoom == pytest.approx(barrel[0, 0] / intrinsics[0, 0])
    # A 1% smaller zoom is no longer admissible: some pixel leaves the render or crosses
    # the lens fold, where the inverse distortion stops converging.
    us, vs = np.meshgrid(np.arange(96, dtype=float), np.arange(64, dtype=float))
    xd, yd = (us - 48.0) / (80.0 * zoom * 0.99), (vs - 32.0) / (80.0 * zoom * 0.99)
    x, y = _undistort(xd, yd, (-0.3, 0.0, 0.0, 0.0))
    back_x, back_y = _distort(x, y, (-0.3, 0.0, 0.0, 0.0))
    loose_u, loose_v = 80.0 * x + 48.0, 80.0 * y + 32.0
    outside = loose_u.min() < 0 or loose_u.max() > 95 or loose_v.min() < 0 or loose_v.max() > 63
    diverged = max(np.abs(back_x - xd).max(), np.abs(back_y - yd).max()) >= 1e-7
    assert outside or diverged
    pincushion, su, sv, _, _ = _sensor_geometry(intrinsics, 64, 96, [0.1, 0.0, 0.0, 0.0])
    assert pincushion[0, 0] == intrinsics[0, 0]
    _, (raw_u, raw_v, _, _) = _fit_geometry(intrinsics, 64, 96, (0.1, 0.0, 0.0, 0.0))
    assert 0 <= raw_u.min() and raw_u.max() <= 95 and 0 <= raw_v.min() and raw_v.max() <= 63
    # Vignetting, gain and a mild lens leave a flat gray frame flat at the center and darker at corners.
    recipe = VisualRandomizer(_cfg()).sample(sample_id=0)
    recipe.sensors["view"] = {**_identity_sensor(), "vignetting": 0.5, "distortion": [-0.1, 0.0, 0.0, 0.0]}
    flat = np.full((64, 96, 3), 128, np.uint8)
    capture = recipe.apply_sensor("view", flat, intrinsics)
    assert capture.rgb[32, 48].tolist() == [128, 128, 128]
    assert capture.rgb[0, 0, 0] < 120


def test_auto_exposure_meets_target_luminance_and_is_optional_in_recipes():
    from metasim.randomization.visual import _srgb_to_linear

    cfg = _cfg()
    cfg.sensors["view"] = SensorRandomCfg(auto_exposure=(0.18, 0.18))
    recipe = VisualRandomizer(cfg).sample(sample_id=0)
    recipe.sensors["view"].update(_identity_sensor(), auto_exposure=0.18)
    intrinsics = np.array([[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]])
    capture = recipe.apply_sensor("view", np.full((32, 32, 3), 230, np.uint8), intrinsics)
    luminance = _srgb_to_linear(capture.rgb / 255.0) @ np.array([0.2126, 0.7152, 0.0722])
    assert capture.exposure_gain < 1 and luminance.mean() == pytest.approx(0.18, abs=0.01)
    dark = recipe.apply_sensor("view", np.full((32, 32, 3), 2, np.uint8), intrinsics)
    assert dark.exposure_gain == 16
    data = json.loads(recipe.to_json())
    del data["sensors"]["view"]["auto_exposure"]
    legacy = VisualRecipe.from_json(json.dumps(data))
    assert legacy.apply_sensor("view", np.full((32, 32, 3), 230, np.uint8), intrinsics).exposure_gain == 1.0
    recipe.sensors["view"]["auto_exposure"] = 1.5
    with pytest.raises(ValueError):
        recipe.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_exposure": (0, 0.5)},
        {"white_balance": ((0, 1), (1, 1))},
        {"distortion": ((0, 0), (0, 0))},
        {"vignetting": (0, 1.5)},
        {"blur_sigma": (-1, 0)},
        {"read_noise": (0, float("nan"))},
    ],
)
def test_invalid_sensor_ranges(kwargs):
    with pytest.raises(ValueError):
        SensorRandomCfg(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uv_scale": ((0, 1), (1, 1))},
        {"uv_rotation": (0, float("inf"))},
        {"ior": (0.5, 1.5)},
        {"color_palette": ((1.1, 0, 0),)},
    ],
)
def test_invalid_material_detail_ranges(kwargs):
    with pytest.raises(ValueError):
        VisualRandomizer(VisualRandomizationCfg(materials={"cube": SurfaceRandomCfg(**kwargs)}))


@pytest.mark.parametrize(
    "field,value",
    [("seed", -1), ("sample_id", True), ("variant_id", 0.5), ("schema_version", True), ("schema_version", 4)],
)
def test_invalid_recipe_identifiers(field, value):
    recipe = VisualRandomizer(_cfg()).sample(sample_id=0)
    setattr(recipe, field, value)
    with pytest.raises(ValueError):
        recipe.validate()


@pytest.mark.parametrize("value", [None, [], {"": {}}, {3: {}}])
def test_malformed_target_maps(value):
    recipe = VisualRecipe(0, 0, 0, materials=value)
    with pytest.raises(ValueError):
        recipe.validate()


@pytest.mark.parametrize(
    "category,field,value",
    [
        ("materials", "roughness", float("nan")),
        ("materials", "metallic", 1.01),
        ("materials", "color", [True, 0, 0]),
        ("materials", "color", "rgb"),
        ("materials", "textures", {}),
        ("lights", "intensity_scale", -1),
        ("lights", "position_delta", None),
        ("lights", "color_temperature", 100),
        ("cameras", "focal_scale", 0),
        ("cameras", "focus_scale", 0),
        ("sensors", "white_balance", [1.0]),
        ("sensors", "distortion", [0.0, 0.0, 0.0]),
        ("sensors", "shot_noise", -0.1),
    ],
)
def test_invalid_realized_parameters(category, field, value):
    recipe = VisualRandomizer(_cfg()).sample(sample_id=0)
    next(iter(getattr(recipe, category).values()))[field] = value
    with pytest.raises(ValueError):
        recipe.validate()


def test_unknown_and_missing_json_keys_fail():
    data = json.loads(VisualRecipe(0, 0, 0).to_json())
    data["future"] = 1
    with pytest.raises(ValueError, match="exactly"):
        VisualRecipe.from_json(json.dumps(data))
    del data["future"]
    del data["materials"]
    with pytest.raises(ValueError, match="exactly"):
        VisualRecipe.from_json(json.dumps(data))


def test_texture_sets_remain_coherent_and_assets_are_checked(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    first.touch()
    second.touch()
    cfg = VisualRandomizationCfg(
        materials={
            "cube": SurfaceRandomCfg(
                textures=(
                    TextureSetCfg(base_color=str(first), roughness=str(first)),
                    TextureSetCfg(base_color=str(second), roughness=str(second)),
                )
            )
        }
    )
    sampler = VisualRandomizer(cfg)
    for i in range(10):
        textures = sampler.sample(sample_id=i).materials["cube"]["textures"]
        assert textures["base_color"] == textures["roughness"]
    first.unlink()
    with pytest.raises(FileNotFoundError):
        VisualRandomizer(cfg)
    with pytest.raises(ValueError, match="HDRI"):
        VisualRandomizer(VisualRandomizationCfg(environment=EnvironmentRandomCfg(hdri_paths=(str(second),))))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"samples": True},
        {"seed": -1},
        {"max_bounces": 0},
        {"exposure": float("inf")},
        {"exposure": "1"},
        {"denoise": 1},
        {"settle_frames": 0},
        {"asset_timeout_s": float("nan")},
    ],
)
def test_render_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RenderCfg(**kwargs)


def test_render_configuration_preserves_positional_hdri():
    assert RenderCfg("pathtracing", 16, "CPU", "studio.hdr").hdri_path == "studio.hdr"


class _Adapter:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.fail = False

    def validate(self, recipe, ids):
        self.calls.append(("validate", ids))

    def apply(self, recipe, ids):
        self.calls.append(("apply", ids))
        if self.fail:
            raise RuntimeError("backend edit failed")

    def invalidate(self):
        self.calls.append(("invalidate", None))

    def render(self):
        self.calls.append(("render", None))


def _bound(monkeypatch):
    from metasim.randomization.core import visual_adapter

    monkeypatch.setattr(visual_adapter, "BlenderVisualAdapter", _Adapter)
    invalidations = []
    renderer = SimpleNamespace(num_envs=2, scenario=SimpleNamespace(simulator="blender"))
    handler = SimpleNamespace(render_handler=renderer, _invalidate_state_caches=lambda: invalidations.append(1))
    sampler = VisualRandomizer(_cfg()).bind_handler(handler)
    return sampler, invalidations


def test_application_preflights_batches_and_invalidates_on_error(monkeypatch):
    sampler, invalidations = _bound(monkeypatch)
    recipe = sampler.sample(sample_id=0)
    sampler.apply(recipe, render=False)
    assert sampler._adapter.calls == [("validate", [0, 1]), ("apply", [0, 1]), ("invalidate", None)]
    assert invalidations == [1]
    sampler._adapter.fail = True
    with pytest.raises(RuntimeError, match="backend edit"):
        sampler.apply(recipe)
    assert sampler._adapter.calls[-1][0] == "invalidate"
    assert len(invalidations) == 2


@pytest.mark.parametrize("ids", [[-1], [2], [True], [0, 0], 0])
def test_invalid_env_selection_never_reaches_adapter(monkeypatch, ids):
    sampler, _ = _bound(monkeypatch)
    with pytest.raises(ValueError):
        sampler.apply(sampler.sample(sample_id=0), env_ids=ids)
    assert sampler._adapter.calls == []


def test_shared_lights_cannot_be_applied_to_subset_and_empty_selection_is_noop(monkeypatch):
    sampler, _ = _bound(monkeypatch)
    recipe = sampler.sample(sample_id=0)
    with pytest.raises(ValueError, match="shared"):
        sampler.apply(recipe, env_ids=[0])
    sampler.apply(recipe, env_ids=[])
    assert sampler._adapter.calls == []
    recipe = copy.deepcopy(recipe)
    recipe.lights = {}
    recipe.environment = None
    sampler.apply(recipe, env_ids=[1])
    assert sampler._adapter.calls[0] == ("validate", [1])
    assert sampler._adapter.calls[-1] == ("render", None)


def test_unsupported_renderer_fails_before_importing_optional_libraries():
    with pytest.raises(NotImplementedError, match="mujoco"):
        VisualRandomizer(_cfg()).bind_handler(SimpleNamespace(scenario=SimpleNamespace(simulator="mujoco")))
