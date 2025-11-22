"""Test camera randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.randomization.camera_randomizer import CameraRandomCfg, CameraRandomizer


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


def camera_position(handler, distribution="uniform"):
    """Test camera position randomization with reproducible seed."""
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg

    # Skip for simulators that don't support camera randomization well

    # Create camera randomizer with position delta
    cfg = CameraRandomCfg(
        camera_name="test_camera",
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


def camera_orientation(handler, distribution="uniform"):
    """Test camera orientation randomization."""
    from metasim.randomization.camera_randomizer import CameraOrientationRandomCfg

    # Create camera randomizer with orientation delta
    cfg = CameraRandomCfg(
        camera_name="test_camera",
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


def camera_look_at(handler, distribution="uniform"):
    """Test camera look-at target randomization."""
    # Create camera randomizer with look-at delta
    from metasim.randomization.camera_randomizer import CameraLookAtRandomCfg

    cfg = CameraRandomCfg(
        camera_name="test_camera",
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


def camera_intrinsics(handler, distribution="uniform"):
    """Test camera intrinsics randomization."""
    from metasim.randomization.camera_randomizer import CameraIntrinsicsRandomCfg

    # Create camera randomizer with intrinsics
    cfg = CameraRandomCfg(
        camera_name="test_camera",
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


def camera_image(handler, distribution="uniform"):
    """Test camera image randomization."""
    from metasim.randomization.camera_randomizer import CameraImageRandomCfg

    cfg = CameraRandomCfg(
        camera_name="test_camera",
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
        camera_name="test_camera",
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


def camera_seed(handler, distribution="uniform"):
    """Test that camera randomization is reproducible with same seed."""
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg

    # Create camera randomizer
    cfg = CameraRandomCfg(
        camera_name="test_camera",
        position=CameraPositionRandomCfg(
            position_range=((-10, 10), (-10, 10), (-10, 10)),
            use_delta=False,
            distribution=distribution,
            enabled=True,
        ),
    )

    # Test reproducibility
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)
    _, xformable = get_camera_xformable_from_randomizer(randomizer)
    # Apply randomization twice with same seed - should give same results
    randomizer()
    pos_val1, _, _ = randomizer._get_current_transform(xformable)
    randomizer.set_seed(789)
    randomizer()
    pos_val2, _, _ = randomizer._get_current_transform(xformable)

    assert pos_val1 == pos_val2, "Same seed should produce same random values"
    log.info("Camera seed reproducibility test passed")


TEST_FUNCTIONS = [
    camera_seed,
    camera_position,
    camera_orientation,
    camera_look_at,
    camera_intrinsics,
    camera_image,
]


@pytest.mark.isaacsim
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_camera_randomizers(handler, test_func, distribution):
    """Run camera randomizer checks inside the shared handler process."""
    test_func(handler, distribution=distribution)
