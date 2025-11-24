"""Test camera randomizer functionality."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.randomization.camera_randomizer import CameraRandomCfg, CameraRandomizer


def get_camera_prim_from_handler(handler, camera_name="test_camera", env_idx=0):
    """Get camera prim from handler for testing.

    Args:
        handler: Handler instance
        camera_name: Name of the camera
        env_idx: Environment index

    Returns:
        tuple: (camera_prim, camera, xformable) from USD
    """
    from pxr import UsdGeom

    # Get camera instance from handler
    camera_inst = handler.scene.sensors[camera_name]
    camera_prim_path_pattern = camera_inst.cfg.prim_path

    # Get stage
    if not hasattr(handler, "stage"):
        import omni.usd

        handler.stage = omni.usd.get_context().get_stage()

    # Construct proper prim path for specific environment
    if "/env_0/" in camera_prim_path_pattern:
        env_prim_path = camera_prim_path_pattern.replace("/env_0/", f"/env_{env_idx}/")
    elif "env_.*" in camera_prim_path_pattern:
        env_prim_path = camera_prim_path_pattern.replace("env_.*", f"env_{env_idx}")
    else:
        env_prim_path = camera_prim_path_pattern

    prim = handler.stage.GetPrimAtPath(env_prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Camera prim not found at {env_prim_path}")

    camera = UsdGeom.Camera(prim)
    xformable = UsdGeom.Xformable(prim)

    if not camera:
        raise ValueError("Camera not found in the scene")
    if not xformable:
        raise ValueError("Camera Xformable not found in the scene")

    return prim, camera, xformable


def get_transform_from_xformable(xformable):
    """Extract position and rotation from xformable."""
    from pxr import UsdGeom

    xform_ops = xformable.GetOrderedXformOps()

    position = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]

    for op in xform_ops:
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            position = list(op.Get())
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotation = list(op.Get())

    return position, rotation


def camera_position(handler, distribution="uniform"):
    """Test camera position randomization with reproducible seed."""
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg

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

    # Get camera instance
    camera_inst = handler.scene.sensors["test_camera"]

    # Get camera prim directly to check actual USD changes
    _, _, xformable = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    current_pos, _ = get_transform_from_xformable(xformable)

    # Apply randomization
    randomizer()

    # Get new position from USD (which is what the randomizer actually updates)
    _, _, xformable = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    new_pos, _ = get_transform_from_xformable(xformable)

    # Check that position changed and is within delta range
    pos_diff = torch.abs(torch.tensor(current_pos) - torch.tensor(new_pos))
    assert not torch.allclose(torch.tensor(current_pos), torch.tensor(new_pos), atol=1e-4), (
        f"Camera position should have changed after randomization. Current: {current_pos}, New: {new_pos}"
    )
    assert torch.all(pos_diff <= 1.0), f"Position delta should be within range [-1, 1], got diff: {pos_diff}"

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

    # Get camera instance
    camera_inst = handler.scene.sensors["test_camera"]

    # Get current orientation from camera instance (quaternion)
    current_quat = camera_inst.data.quat_w_world[0].cpu().numpy()

    # Apply randomization
    randomizer()

    # Step simulation to apply changes
    handler.step()
    handler.render()

    # Get new orientation
    new_quat = camera_inst.data.quat_w_world[0].cpu().numpy()

    # Check that orientation changed
    assert not torch.allclose(torch.tensor(current_quat), torch.tensor(new_quat), atol=1e-4), (
        f"Camera orientation should have changed after randomization. Current: {current_quat}, New: {new_quat}"
    )

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

    # Get camera instance
    camera_inst = handler.scene.sensors["test_camera"]

    # Get current transform from camera instance
    current_quat = camera_inst.data.quat_w_world[0].cpu().numpy()

    # Apply randomization
    randomizer()

    # Step simulation to apply changes
    handler.step()
    handler.render()

    # Get new transform
    new_quat = camera_inst.data.quat_w_world[0].cpu().numpy()

    # Rotation should change due to look-at target change
    assert not torch.allclose(torch.tensor(current_quat), torch.tensor(new_quat), atol=1e-4), (
        f"Camera orientation should have changed after look-at randomization. Current: {current_quat}, New: {new_quat}"
    )

    log.info(f"Camera look-at randomization (Type: {distribution}) test passed")


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

    # Get camera prim directly from USD
    _, camera, _ = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)

    # Get current focal length
    current_focal = camera.GetFocalLengthAttr().Get()

    # Apply randomization
    randomizer()

    # Get new focal length (need to re-get camera as USD might have updated)
    _, camera, _ = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    new_focal = camera.GetFocalLengthAttr().Get()

    assert current_focal != new_focal, "Camera intrinsics should have changed after randomization"
    # Note: focal_length_range is in cm, but USD stores in mm, so multiply by 10
    assert 180.0 <= new_focal <= 350.0, f"Focal length should be within specified range (180-350mm), got {new_focal}"

    # Test clipping range
    cfg.intrinsics.clipping_range = ((0.1, 1.0), (20.0, 100.0))
    randomizer = CameraRandomizer(cfg, seed=790)
    randomizer.bind_handler(handler)
    randomizer()

    _, camera, _ = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    new_range = camera.GetClippingRangeAttr().Get()
    # USD stores in cm, config is in meters, so multiply by 100
    assert 10.0 <= new_range[0] <= 100.0 and 2000.0 <= new_range[1] <= 10000.0, (
        f"Clipping range should be within specified range, got {new_range}"
    )

    log.info(f"Camera intrinsics randomization (Type: {distribution}) test passed")


def camera_image(handler, distribution="uniform"):
    """Test camera image randomization.

    Note: CameraImageRandomCfg was removed from the API.
    This test now validates that horizontal aperture changes affect image properties.
    """
    from metasim.randomization.camera_randomizer import CameraIntrinsicsRandomCfg

    cfg = CameraRandomCfg(
        camera_name="test_camera",
        intrinsics=CameraIntrinsicsRandomCfg(
            horizontal_aperture_range=(15.0, 30.0),
            distribution=distribution,
            enabled=True,
        ),
    )
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Get camera prim from USD
    _, camera, _ = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)

    current_aperture = camera.GetHorizontalApertureAttr().Get()
    randomizer()

    # Re-get camera after randomization
    _, camera, _ = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    new_aperture = camera.GetHorizontalApertureAttr().Get()

    assert abs(current_aperture - new_aperture) > 1e-3, (
        "Camera horizontal aperture should have changed after randomization"
    )
    # aperture_range is in cm, USD stores in mm
    assert 150.0 <= new_aperture <= 300.0, (
        f"Horizontal aperture should be within specified range (150-300mm), got {new_aperture}"
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

    # Test reproducibility using USD transforms
    randomizer = CameraRandomizer(cfg, seed=789)
    randomizer.bind_handler(handler)

    # Apply randomization with seed 789
    randomizer()
    _, _, xformable1 = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    pos_val1, _ = get_transform_from_xformable(xformable1)

    # Reset seed and apply again - should give same results
    randomizer.set_seed(789)
    randomizer()
    _, _, xformable2 = get_camera_prim_from_handler(handler, "test_camera", env_idx=0)
    pos_val2, _ = get_transform_from_xformable(xformable2)

    assert torch.allclose(torch.tensor(pos_val1), torch.tensor(pos_val2), atol=1e-4), (
        f"Same seed should produce same random values, got {pos_val1} and {pos_val2}"
    )
    log.info(f"Camera seed reproducibility (Type: {distribution}) test passed")


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
