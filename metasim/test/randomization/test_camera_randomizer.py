"""Test camera randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.randomization.camera_randomizer import CameraRandomCfg, CameraRandomizer


def camera_seed(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test that camera randomization is reproducible with same seed."""
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg

    # Create camera randomizer
    cfg = CameraRandomCfg(
        camera_name="test_camera",
        position=CameraPositionRandomCfg(
            position_range=[common_range for _ in range(3)],
            use_delta=False,
            distribution=distribution,
            enabled=True,
        ),
    )

    # Test reproducibility using USD transforms
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Apply randomization with seed 789
    randomizer()
    pos_val1, _ = _get_transform(handler)

    # Reset seed and apply again - should give same results
    randomizer.set_seed(789)
    randomizer()
    pos_val2, _ = _get_transform(handler)

    assert torch.allclose(torch.tensor(pos_val1), torch.tensor(pos_val2), atol=1e-4), (
        f"Same seed should produce same random values, got {pos_val1} and {pos_val2}"
    )
    log.info(f"Camera seed reproducibility (Type: {distribution}) test passed")


def camera_position(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test camera position randomization with reproducible seed."""
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg

    # Create camera randomizer with position delta
    cfg = CameraRandomCfg(
        camera_name="test_camera",
        position=CameraPositionRandomCfg(
            delta_range=[common_range for _ in range(3)],
            use_delta=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get camera prim directly to check actual USD changes
    current_pos, _ = _get_transform(handler)
    # Apply randomization
    randomizer()
    # Get new position from USD (which is what the randomizer actually updates)
    new_pos, _ = _get_transform(handler)

    # Check that position changed and is within delta range
    pos_diff = torch.abs(torch.tensor(current_pos) - torch.tensor(new_pos))
    assert not torch.allclose(torch.tensor(current_pos), torch.tensor(new_pos), atol=1e-4), (
        f"Camera position should have changed after randomization. Current: {current_pos}, New: {new_pos}"
    )
    ranges = torch.tensor(cfg.position.delta_range)
    assert torch.all(pos_diff <= (ranges[:, 1] - ranges[:, 0])), (
        f"Position delta should be within range [-10, 10], got diff: {pos_diff}"
    )

    log.info(f"Camera position randomization (Type: {distribution}) test passed")


def camera_orientation(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test camera orientation randomization."""
    from metasim.randomization.camera_randomizer import CameraOrientationRandomCfg

    # Create camera randomizer with orientation delta
    cfg = CameraRandomCfg(
        camera_name="test_camera",
        orientation=CameraOrientationRandomCfg(
            rotation_delta=[common_range for _ in range(3)],
            distribution=distribution,
            enabled=True,
        ),
    )
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get camera instance

    # Get current orientation from camera instance (quaternion)
    _, current_quat = _get_transform(handler)
    # Apply randomization
    randomizer()
    # Get new orientation
    _, new_quat = _get_transform(handler)

    # Check that orientation changed
    assert not torch.allclose(torch.tensor(current_quat), torch.tensor(new_quat), atol=1e-4), (
        f"Camera orientation should have changed after randomization. Current: {current_quat}, New: {new_quat}"
    )
    from metasim.utils.math import euler_xyz_from_quat

    c_r, c_p, c_y = euler_xyz_from_quat(current_quat)
    n_r, n_p, n_y = euler_xyz_from_quat(new_quat)
    # Check that each rotation axis change is within specified delta range
    ranges = torch.tensor(cfg.orientation.rotation_delta)
    assert torch.all(
        torch.abs(torch.stack([c_r - n_r, c_p - n_p, c_y - n_y], dim=-1)) <= (ranges[:, 1] - ranges[:, 0])
    ), (
        f"Orientation delta should be within range {cfg.orientation.rotation_delta}, got diffs: {[c_r - n_r, c_p - n_p, c_y - n_y]}"
    )
    log.info(f"Camera orientation randomization (Type: {distribution}) test passed")


def camera_look_at(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test camera look-at target randomization."""
    # Create camera randomizer with look-at delta
    from metasim.randomization.camera_randomizer import CameraLookAtRandomCfg

    cfg = CameraRandomCfg(
        camera_name="test_camera",
        look_at=CameraLookAtRandomCfg(
            look_at_delta=[common_range for _ in range(3)],
            use_delta=True,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get current transform from camera instance
    _, current_quat = _get_transform(handler)
    # Apply randomization
    randomizer()
    # Get new transform
    _, new_quat = _get_transform(handler)

    # Rotation should change due to look-at target change
    assert not torch.allclose(torch.tensor(current_quat), torch.tensor(new_quat), atol=1e-4), (
        f"Camera orientation should have changed after look-at randomization. Current: {current_quat}, New: {new_quat}"
    )

    log.info(f"Camera look-at randomization (Type: {distribution}) test passed")


def camera_intrinsics(handler, distribution="uniform", common_range=(1e-8, 10.0)):
    """Test camera intrinsics randomization."""
    import omni.usd
    from pxr import UsdGeom

    from metasim.randomization.camera_randomizer import CameraIntrinsicsRandomCfg

    # Create camera randomizer with intrinsics
    cfg = CameraRandomCfg(
        camera_name="test_camera",
        intrinsics=CameraIntrinsicsRandomCfg(
            focal_length_range=common_range,
            horizontal_aperture_range=common_range,
            focus_distance_range=common_range,
            distribution=distribution,
            enabled=True,
        ),
    )

    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get camera prim directly from USD
    camera_inst = handler.scene.sensors["test_camera"]
    camera_prim_path = camera_inst.cfg.prim_path.replace("env_.*", "env_0")
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    camera = UsdGeom.Camera(camera_prim)

    # Get current focal length
    current_focal_mm = camera.GetFocalLengthAttr().Get()

    # Apply randomization
    randomizer()

    # Get new focal length (need to re-get camera as USD might have updated)
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    camera = UsdGeom.Camera(camera_prim)
    new_focal_mm = camera.GetFocalLengthAttr().Get()

    assert current_focal_mm != new_focal_mm, "Camera intrinsics should have changed after randomization"
    # Note: focal_length_range is in cm, but USD stores in mm, so multiply by 10
    assert common_range[0] * 10.0 <= new_focal_mm <= common_range[1] * 10.0, (
        f"Focal length should be within specified range, got {new_focal_mm}"
    )

    # Test clipping range
    low_range = (common_range[0] * 0.1, common_range[1] * 0.1)
    cfg.intrinsics.clipping_range = (low_range, common_range)
    randomizer = CameraRandomizer(cfg, seed=790)
    randomizer.bind_handler(handler)
    randomizer()

    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    camera = UsdGeom.Camera(camera_prim)
    new_range_mm = camera.GetClippingRangeAttr().Get()
    # USD stores in cm, config is in meters, so multiply by 100
    assert (
        low_range[0] * 100.0 <= new_range_mm[0] <= low_range[1] * 100.0
        and common_range[0] * 100.0 <= new_range_mm[1] <= common_range[1] * 100.0
    ), f"Clipping range should be within specified range, got {new_range_mm}"

    log.info(f"Camera intrinsics randomization (Type: {distribution}) test passed")


TEST_FUNCTIONS = [
    camera_seed,
    camera_position,
    camera_orientation,
    camera_look_at,
    camera_intrinsics,
]


@pytest.mark.isaacsim
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
@pytest.mark.parametrize("test_func", TEST_FUNCTIONS, ids=[f.__name__ for f in TEST_FUNCTIONS])
def test_camera_randomizers(handler, test_func, distribution):
    """Run camera randomizer checks inside the shared handler process."""
    common_range = (1e-8, 20.0)
    test_func(handler, distribution=distribution, common_range=common_range)


def _get_transform(handler):
    """Extract position and rotation from xformable."""
    camera_inst = handler.scene.sensors["test_camera"]
    position, rotation = camera_inst._view.get_world_poses()

    return position.detach().cpu(), rotation.detach().cpu()


if __name__ == "__main__":
    pytest.main([__file__, "-k isaacsim"])
