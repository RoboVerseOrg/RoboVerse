"""Test light randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.light_randomizer import LightRandomCfg, LightRandomizer


def light_intensity(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test light intensity randomization with reproducible seed."""
    from metasim.randomization.light_randomizer import LightIntensityRandomCfg

    # Create light randomizer with intensity randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        intensity=LightIntensityRandomCfg(
            intensity_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, light_path, _ = _get_light_prim_from_randomizer(randomizer)

    # Get current intensity using adapter
    current_intensity = randomizer.adapter.get_light_intensity(light_path)

    # Apply randomization
    randomizer()
    new_intensity = randomizer.adapter.get_light_intensity(light_path)

    assert current_intensity != new_intensity, "Light intensity should have changed after randomization"
    assert common_range[0] <= new_intensity <= common_range[1], "Intensity should be within specified range"

    log.info(f"Light intensity randomization (Type: {distribution}) test passed")


def light_color(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test light color randomization."""
    from metasim.randomization.light_randomizer import LightColorRandomCfg

    # Create light randomizer with RGB color randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        color=LightColorRandomCfg(
            color_range=(common_range, common_range, common_range),
            use_temperature=False,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = _get_light_prim_from_randomizer(randomizer)

    # Get current color
    color_attr = light_prim.GetAttribute("inputs:color")
    current_color = color_attr.Get()

    # Apply randomization
    randomizer()
    new_color = color_attr.Get()

    assert current_color != new_color, "Light color should have changed after randomization"
    # Check color values are within range
    assert all(common_range[0] <= c <= common_range[1] for c in new_color), (
        "Color values should be within specified range"
    )

    log.info(f"Light color randomization (Type: {distribution}) test passed")


def light_color_temperature(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test light color temperature randomization."""
    from metasim.randomization.light_randomizer import LightColorRandomCfg

    # Create light randomizer with color temperature
    common_range = (common_range[0] + 1000, common_range[1] + 1000)  # Shift to valid temperature range
    cfg = LightRandomCfg(
        light_name="test_light",
        color=LightColorRandomCfg(
            temperature_range=common_range,
            use_temperature=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = _get_light_prim_from_randomizer(randomizer)

    # Get current color
    color_attr = light_prim.GetAttribute("inputs:color")
    current_color = color_attr.Get()

    # Apply randomization
    randomizer()
    new_color = color_attr.Get()

    assert current_color != new_color, "Light color should have changed after temperature randomization"

    log.info(f"Light color temperature randomization (Type: {distribution}) test passed")


def light_position(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test light position randomization."""
    # Create light randomizer with position randomization
    from metasim.randomization.light_randomizer import LightPositionRandomCfg

    cfg = LightRandomCfg(
        light_name="test_light",
        position=LightPositionRandomCfg(
            position_range=(common_range, common_range, common_range),
            relative_to_origin=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, light_path, light_type = _get_light_prim_from_randomizer(randomizer)

    # Skip for distant lights
    if light_type == "distant":
        log.info(f"Skipping position randomization for distant light (Type: {distribution})")
        return

    # Get current position using adapter
    current_pos, _, _ = randomizer.adapter.get_transform(light_path)

    # Apply randomization
    randomizer()
    new_pos, _, _ = randomizer.adapter.get_transform(light_path)

    assert current_pos != new_pos, "Light position should have changed after randomization"
    # Check position changes are within delta range
    delta = torch.tensor([abs(new_pos[i] - current_pos[i]) for i in range(3)])
    max_delta = common_range[1] - common_range[0]
    assert torch.all(delta <= max_delta), f"Position delta should be within specified range, got delta: {delta}"

    log.info(f"Light position randomization (Type: {distribution}) test passed")


def light_orientation(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test light orientation randomization."""
    from metasim.randomization.light_randomizer import LightOrientationRandomCfg

    # Create light randomizer with orientation randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        orientation=LightOrientationRandomCfg(
            angle_range=(common_range, common_range, common_range),
            relative_to_origin=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, light_path, _ = _get_light_prim_from_randomizer(randomizer)

    # Get current rotation using adapter (returns quaternion)
    _, current_rot, _ = randomizer.adapter.get_transform(light_path)

    # Apply randomization
    randomizer()
    _, new_rot, _ = randomizer.adapter.get_transform(light_path)

    assert current_rot != new_rot, "Light orientation should have changed after randomization"
    # Check that quaternion values changed
    from metasim.utils.math import euler_xyz_from_quat

    n_r, n_p, n_y = euler_xyz_from_quat(torch.tensor(new_rot).unsqueeze(0))
    # Check that each rotation axis change is within specified delta range
    ranges = torch.tensor(cfg.orientation.angle_range)
    assert torch.all(torch.abs(torch.stack([n_r, n_p, n_y], dim=-1)) <= (ranges[:, 1] - ranges[:, 0])), (
        f"Orientation delta should be within range {cfg.orientation.angle_range}, got angles: {[n_r, n_p, n_y]}"
    )

    log.info(f"Light orientation randomization (Type: {distribution}) test passed")


def light_seed(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test that light randomization is reproducible with same seed."""
    from metasim.randomization.light_randomizer import LightIntensityRandomCfg

    # Create light randomizer
    cfg = LightRandomCfg(
        light_name="test_light",
        intensity=LightIntensityRandomCfg(intensity_range=common_range, enabled=True, distribution=distribution),
    )

    # Test reproducibility
    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    light_prim, light_path, _ = _get_light_prim_from_randomizer(randomizer)

    # Apply randomization twice with same seed - should give same results
    randomizer()
    intensity_val1 = randomizer.adapter.get_light_intensity(light_path)
    randomizer.set_seed(789)
    randomizer()
    intensity_val2 = randomizer.adapter.get_light_intensity(light_path)

    assert intensity_val1 == intensity_val2, "Same seed should produce same random values"
    log.info("Light seed reproducibility test passed")


TEST_FUNCTIONS = [
    light_intensity,
    light_color,
    light_color_temperature,
    light_position,
    light_orientation,
    light_seed,
]


@pytest.mark.isaacsim
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_light_randomizers(handler, test_func, distribution):
    """Run light randomizer checks inside the shared handler process."""
    common_range = (1e-8, 20.0)
    test_func(handler, distribution=distribution, common_range=common_range)


def _get_light_prim_from_randomizer(randomizer: LightRandomizer):
    """Helper function to get light prim and attributes from randomizer."""
    # Get prim path from registry
    prim_paths = randomizer.registry.get_prim_paths(randomizer.cfg.light_name)
    if not prim_paths:
        raise ValueError(f"Light '{randomizer.cfg.light_name}' not found in the scene")

    light_path = prim_paths[0]  # Get first light

    # Get prim from adapter's stage
    light_prim = randomizer.adapter.stage.GetPrimAtPath(light_path)
    if not light_prim or not light_prim.IsValid():
        raise ValueError(f"Invalid light prim at path: {light_path}")

    # Determine light type from prim type
    light_type = light_prim.GetTypeName()
    if "Disk" in light_type:
        light_type = "disk"
    elif "Sphere" in light_type:
        light_type = "sphere"
    elif "Distant" in light_type:
        light_type = "distant"
    else:
        light_type = "unknown"

    return light_prim, light_path, light_type


if __name__ == "__main__":
    pytest.main([__file__, "-k isaacsim"])
