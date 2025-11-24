"""Test light randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.light_randomizer import LightRandomCfg, LightRandomizer


def get_light_prim_from_randomizer(randomizer: LightRandomizer):
    """Helper function to get light prim and attributes from randomizer."""
    light_prim, light_path, light_type = randomizer._get_light_prim(randomizer.cfg.light_name)
    if not light_prim:
        raise ValueError("Light not found in the scene")
    return light_prim, light_path, light_type


def light_intensity(handler, distribution="uniform"):
    """Test light intensity randomization with reproducible seed."""
    from metasim.randomization.light_randomizer import LightIntensityRandomCfg

    # Create light randomizer with intensity randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        intensity=LightIntensityRandomCfg(
            intensity_range=(10000.0, 20000.0),
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = get_light_prim_from_randomizer(randomizer)

    # Get current intensity
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    current_intensity = intensity_attr.Get()

    # Apply randomization
    randomizer()
    new_intensity = intensity_attr.Get()

    assert current_intensity != new_intensity, "Light intensity should have changed after randomization"
    assert 10000.0 <= new_intensity <= 20000.0, "Intensity should be within specified range"

    log.info(f"Light intensity randomization (Type: {distribution}) test passed")


def light_color(handler, distribution="uniform"):
    """Test light color randomization."""
    from metasim.randomization.light_randomizer import LightColorRandomCfg

    # Create light randomizer with RGB color randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        color=LightColorRandomCfg(
            color_range=((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
            use_temperature=False,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = get_light_prim_from_randomizer(randomizer)

    # Get current color
    color_attr = light_prim.GetAttribute("inputs:color")
    current_color = color_attr.Get()

    # Apply randomization
    randomizer()
    new_color = color_attr.Get()

    assert current_color != new_color, "Light color should have changed after randomization"
    # Check color values are within range
    assert all(0.5 <= c <= 1.0 for c in new_color), "Color values should be within specified range"

    log.info(f"Light color randomization (Type: {distribution}) test passed")


def light_color_temperature(handler, distribution="uniform"):
    """Test light color temperature randomization."""
    from metasim.randomization.light_randomizer import LightColorRandomCfg

    # Create light randomizer with color temperature
    cfg = LightRandomCfg(
        light_name="test_light",
        color=LightColorRandomCfg(
            temperature_range=(2700.0, 6500.0),  # Warm to cool white
            use_temperature=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = get_light_prim_from_randomizer(randomizer)

    # Get current color
    color_attr = light_prim.GetAttribute("inputs:color")
    current_color = color_attr.Get()

    # Apply randomization
    randomizer()
    new_color = color_attr.Get()

    assert current_color != new_color, "Light color should have changed after temperature randomization"

    log.info(f"Light color temperature randomization (Type: {distribution}) test passed")


def light_position(handler, distribution="uniform"):
    """Test light position randomization."""
    # Create light randomizer with position randomization
    from metasim.randomization.light_randomizer import LightPositionRandomCfg

    cfg = LightRandomCfg(
        light_name="test_light",
        position=LightPositionRandomCfg(
            position_range=((0.1, 2.0), (0.1, 2.0), (0.1, 2.0)),
            relative_to_origin=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, light_type = get_light_prim_from_randomizer(randomizer)

    # Skip for distant lights
    if light_type == "distant":
        log.info(f"Skipping position randomization for distant light (Type: {distribution})")
        return

    # Get current position
    translate_attr = light_prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        from pxr import Sdf

        translate_attr = light_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
    current_pos = translate_attr.Get()
    # Apply randomization
    randomizer()
    new_pos = translate_attr.Get()
    assert current_pos != new_pos, "Light position should have changed after randomization"
    # Check position changes are within delta range
    delta = torch.tensor([abs(new_pos[i] - current_pos[i]) for i in range(3)])
    assert torch.all(delta <= torch.tensor([1.9, 1.9, 1.9])), "Position delta should be within specified range"

    log.info(f"Light position randomization (Type: {distribution}) test passed")


def light_orientation(handler, distribution="uniform"):
    """Test light orientation randomization."""
    from metasim.randomization.light_randomizer import LightOrientationRandomCfg

    # Create light randomizer with orientation randomization
    cfg = LightRandomCfg(
        light_name="test_light",
        orientation=LightOrientationRandomCfg(
            angle_range=((0.1, 5.0), (0.1, 5.0), (0.1, 5.0)),
            relative_to_origin=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    light_prim, _, _ = get_light_prim_from_randomizer(randomizer)

    # Get current rotation
    rotate_attr = light_prim.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr:
        # Create rotation attribute if it doesn't exist
        from pxr import Sdf

        rotate_attr = light_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3)

    current_rot = rotate_attr.Get()
    if current_rot is None:
        current_rot = (0.0, 0.0, 0.0)

    # Apply randomization
    randomizer()
    new_rot = rotate_attr.Get()

    assert current_rot != new_rot, "Light orientation should have changed after randomization"
    # Check rotation changes are within delta range
    delta = torch.tensor([abs(new_rot[i] - current_rot[i]) for i in range(3)])
    assert torch.all(delta <= 4.9), "Rotation delta should be within specified range"

    log.info(f"Light orientation randomization (Type: {distribution}) test passed")


def light_seed(handler, distribution="uniform"):
    """Test that light randomization is reproducible with same seed."""
    from metasim.randomization.light_randomizer import LightIntensityRandomCfg

    # Create light randomizer
    cfg = LightRandomCfg(
        light_name="test_light",
        intensity=LightIntensityRandomCfg(intensity_range=(100.0, 1000.0), enabled=True, distribution=distribution),
    )

    # Test reproducibility
    randomizer = LightRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    light_prim, _, _ = get_light_prim_from_randomizer(randomizer)
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    # Apply randomization twice with same seed - should give same results
    randomizer()
    intensity_val1 = intensity_attr.Get()
    randomizer.set_seed(789)
    randomizer()
    intensity_val2 = intensity_attr.Get()

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
    test_func(handler, distribution=distribution)
