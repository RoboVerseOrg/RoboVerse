"""Test camera randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)
import multiprocessing as mp

from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.randomization.camera_randomizer import (
    CameraImageRandomCfg,
    CameraIntrinsicsRandomCfg,
    CameraLookAtRandomCfg,
    CameraOrientationRandomCfg,
    CameraPositionRandomCfg,
    CameraRandomCfg,
    CameraRandomizer,
)
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.test.test_utils import get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg


def get_camera_xformable_from_randomizer(randomizer):
    from pxr import UsdGeom

    camera_prim = randomizer._get_camera_prim()
    camera = UsdGeom.Camera(camera_prim)
    xformable = UsdGeom.Xformable(camera_prim)
    if not camera:
        raise ValueError("Camera not found in the scene")
    if not xformable:
        raise ValueError("Camera Xformable not found in the scene")
    return camera, xformable


def camera_position_randomization(handler, distribution="uniform"):
    """Test camera position randomization with reproducible seed."""
    # Skip for simulators that don't support camera randomization well

    # Create camera randomizer with position delta
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        position=CameraPositionRandomCfg(
            delta_range=((-1, 1), (-1, 1), (-1, 1)),
            use_delta=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    _, xformable = get_camera_xformable_from_randomizer(randomizer)
    # Get current position
    current_pos, _, _ = randomizer._get_current_transform(xformable)
    # Apply randomization twice with same seed - should give same results
    randomizer()
    new_pos, _, _ = randomizer._get_current_transform(xformable)
    assert (not torch.allclose(torch.tensor(current_pos), torch.tensor(new_pos))) and torch.all(
        torch.abs(torch.tensor(current_pos) - torch.tensor(new_pos)) < 1
    ), "Camera position should have changed after randomization"
    # For IsaacSim, we need to step to see the effect
    log.info(f"Camera position randomization (Type: {distribution}) test passed")


def camera_orientation_randomization(handler, distribution="uniform"):
    """Test camera orientation randomization."""
    # Create camera randomizer with orientation delta
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        orientation=CameraOrientationRandomCfg(
            rotation_delta=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
            distribution=distribution,
            enabled=True,
        ),
    )
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    _, xformable = get_camera_xformable_from_randomizer(randomizer)
    # Get current orientation
    _, current_rot, _ = randomizer._get_current_transform(xformable)
    # Apply randomization
    randomizer()
    _, new_rot, _ = randomizer._get_current_transform(xformable)
    assert (not torch.allclose(torch.tensor(current_rot), torch.tensor(new_rot))) and torch.all(
        torch.abs(torch.tensor(current_rot) - torch.tensor(new_rot)) < 5
    ), "Camera orientation should have changed after randomization"
    log.info(f"Camera orientation randomization (Type: {distribution}) test passed")


def camera_look_at_randomization(handler, distribution="uniform"):
    """Test camera look-at target randomization."""
    # Create camera randomizer with look-at delta
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        look_at=CameraLookAtRandomCfg(
            look_at_delta=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
            use_delta=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    _, xformable = get_camera_xformable_from_randomizer(randomizer)
    # Get current transform
    current_pos, current_rot, _ = randomizer._get_current_transform(xformable)
    # Apply randomization
    randomizer()
    new_pos, new_rot, _ = randomizer._get_current_transform(xformable)
    # Either position or rotation should change due to look-at
    assert not torch.allclose(torch.tensor(current_pos), torch.tensor(new_pos)) or not torch.allclose(
        torch.tensor(current_rot), torch.tensor(new_rot)
    ), "Camera transform should have changed after look-at randomization"
    log.info("Camera look-at randomization test passed")


def camera_intrinsics_randomization(handler, distribution="uniform"):
    """Test camera intrinsics randomization."""
    # Create camera randomizer with intrinsics
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        intrinsics=CameraIntrinsicsRandomCfg(
            focal_length_range=(18.0, 35.0),
            horizontal_aperture_range=(15.0, 25.0),
            focus_distance_range=(0.5, 5.0),
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    camera, _ = get_camera_xformable_from_randomizer(randomizer)
    # Get current focal length
    current_focal = camera.GetFocalLengthAttr().Get()
    # Apply randomization
    randomizer()
    new_focal = camera.GetFocalLengthAttr().Get()
    assert current_focal != new_focal, "Camera intrinsics should have changed after randomization"
    assert 18.0 <= new_focal <= 35.0, "Focal length should be within specified range"

    cfg.intrinsics.clipping_range = ((0.1, 1.0), (20, 100.0))
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    randomizer()
    new_range = torch.tensor(camera.CreateClippingRangeAttr().Get())
    assert new_range[0] >= 0.1 and new_range[0] <= 1.0 and new_range[1] >= 20.0 and new_range[1] <= 100.0, (
        "Clipping range should be within specified range"
    )

    log.info("Camera intrinsics randomization test passed")


def camera_image_randomization(handler, distribution="uniform"):
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        image=CameraImageRandomCfg(
            width_range=(640, 1280),
            height_range=(480, 960),
            use_aspect_ratio=True,
            distribution=distribution,
            enabled=True,
        ),
    )
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    camera, _ = get_camera_xformable_from_randomizer(randomizer)

    current_focal_attr = camera.GetFocalLengthAttr().Get()
    randomizer()
    new_focal_attr = camera.GetFocalLengthAttr().Get()
    assert torch.abs(torch.tensor(current_focal_attr) - torch.tensor(new_focal_attr)) > 1e-3, (
        "Camera focal length should have changed after image resolution randomization"
    )

    cfg = CameraRandomCfg(
        camera_name="default_camera",
        image=CameraImageRandomCfg(
            aspect_ratio_range=(1, 1000),
            distribution=distribution,
            enabled=True,
        ),
    )
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    camera, _ = get_camera_xformable_from_randomizer(randomizer)
    current_aperture = camera.CreateHorizontalApertureAttr().Get()
    randomizer()
    new_aperture = camera.CreateHorizontalApertureAttr().Get()
    assert torch.abs(torch.tensor(current_aperture) - torch.tensor(new_aperture)) > 1e-3, (
        "Camera focal length should have changed after aspect ratio randomization"
    )

    log.info(f"Camera image randomization (Type: {distribution}) test passed")


def camera_seed_reproducibility(handler):
    """Test that camera randomization is reproducible with same seed."""
    # Create camera randomizer
    cfg = CameraRandomCfg(
        camera_name="default_camera",
        position=CameraPositionRandomCfg(
            delta_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
            use_delta=True,
            enabled=True,
        ),
    )

    # Test reproducibility
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Store RNG internal state by generating some values
    randomizer.set_seed(42)
    val1 = randomizer._rng.random()

    randomizer.set_seed(42)
    val2 = randomizer._rng.random()

    assert val1 == val2, "Same seed should produce same random values"
    log.info("Camera seed reproducibility test passed")


def run_handler_process(scenario):
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    camera_seed_reproducibility(handler)
    distributions = ["uniform", "log_uniform", "gaussian"]
    for dist in distributions:
        camera_position_randomization(handler, distribution=dist)
        camera_orientation_randomization(handler, distribution=dist)
        camera_look_at_randomization(handler, distribution=dist)
        camera_intrinsics_randomization(handler, distribution=dist)
        camera_image_randomization(handler, distribution=dist)
    handler.close()


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_main(sim, num_envs):
    log.info(f"Running camera randomizer test with {sim} and {num_envs}")

    if sim not in ["isaacsim"]:
        pytest.skip("Only testing IsaacSim here")

    scenario = ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.4, -0.6, 0.05],
            ),
        ],
        robots=[FrankaCfg()],
        cameras=[
            PinholeCameraCfg(
                name="default_camera",
                width=1024,
                height=1024,
                pos=(2.0, -2.0, 2.0),
                look_at=(0.0, 0.0, 0.05),
            )
        ],
    )

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=run_handler_process, args=(scenario,))
    p.start()
    p.join(timeout=60)

    assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
    log.info("IsaacSim headless test finished successfully.")


if __name__ == "__main__":
    # Direct execution for quick testing
    import sys

    sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
    num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    test_main(sim, num_envs)
