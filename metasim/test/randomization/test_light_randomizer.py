"""Test light randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.light_randomizer import LightRandomCfg, LightRandomizer
from metasim.test.randomization.conftest import get_shared_scenario


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


def _process_run_handler(scenario):
    """Process function for standalone mode - creates its own handler."""
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    distributions = ["uniform", "log_uniform", "gaussian"]
    for dist in distributions:
        for test_func in TEST_FUNCTIONS:
            test_func(handler, distribution=dist)
    handler.close()


def run_test(sim="isaacsim", num_envs=2):
    """Standalone test function for direct execution."""
    import multiprocessing as mp

    log.info(f"Running light randomizer test in standalone mode with {sim} and {num_envs}")

    if sim not in ["isaacsim"]:
        log.warning(f"Skipping: Only testing IsaacSim here, got {sim}")
        return

    scenario = get_shared_scenario(sim, num_envs)

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_process_run_handler, args=(scenario,))
    p.start()
    p.join(timeout=60)

    assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
    log.info("IsaacSim headless test finished successfully.")


@pytest.mark.usefixtures("shared_handler")
def test_light_randomizer_with_shared_handler(shared_handler):
    """Run light randomizer tests using the child-process handler via proxy."""
    log.info("Running light randomizer tests with shared handler (proxy)")

    proxy = shared_handler  # HandlerProxy

    distributions = ["uniform", "log_uniform", "gaussian"]

    # Run all light test functions with different distributions
    for dist in distributions:
        for test_func in TEST_FUNCTIONS:
            proxy.run_test(func=test_func, distribution=dist)

    log.info("All light randomizer tests completed with shared handler (proxy)")


if __name__ == "__main__":
    # Direct execution for quick testing - uses standalone mode
    import sys

    sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
    num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    run_test(sim, num_envs)
