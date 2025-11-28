"""Tests for metasim.utils.camera_util module.

These are pure unit tests for camera utility functions.
All tests are marked @pytest.mark.general.
"""

import pytest
import torch

from metasim.utils.camera_util import get_cam_params


@pytest.mark.general
def test_get_cam_params_single_camera():
    """Test camera parameter computation for a single camera."""
    cam_pos = torch.tensor([[1.0, 0.0, 0.5]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    extrinsics, intrinsics = get_cam_params(
        cam_pos,
        cam_look_at,
        width=640,
        height=480,
        focal_length=24,
        horizontal_aperture=20.955,
    )

    # Check shapes
    assert extrinsics.shape == (1, 4, 4)
    assert intrinsics.shape == (1, 3, 3)

    # Check that extrinsics is a valid transformation matrix
    # Bottom row should be [0, 0, 0, 1]
    assert torch.allclose(extrinsics[0, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)

    # Check that rotation part is orthogonal (R^T R = I)
    R = extrinsics[0, :3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)

    # Check intrinsics structure (should be upper triangular)
    assert torch.allclose(intrinsics[0, 1, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(intrinsics[0, 2, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(intrinsics[0, 2, 1], torch.tensor(0.0), atol=1e-6)

    # Check that fx, fy are positive
    assert intrinsics[0, 0, 0] > 0
    assert intrinsics[0, 1, 1] > 0


@pytest.mark.general
def test_get_cam_params_batch():
    """Test camera parameter computation for multiple cameras."""
    num_envs = 4
    cam_pos = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [-1.0, 0.0, 0.5],
        [0.0, -1.0, 0.5],
    ])
    cam_look_at = torch.zeros(num_envs, 3)

    extrinsics, intrinsics = get_cam_params(cam_pos, cam_look_at)

    # Check shapes
    assert extrinsics.shape == (num_envs, 4, 4)
    assert intrinsics.shape == (num_envs, 3, 3)

    # Check all cameras have valid transformations
    for i in range(num_envs):
        R = extrinsics[i, :3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)


@pytest.mark.general
def test_get_cam_params_vertical_aperture():
    """Test camera parameters with custom vertical aperture."""
    cam_pos = torch.tensor([[1.0, 0.0, 0.5]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    width, height = 640, 480
    horizontal_aperture = 20.955
    vertical_aperture = 15.0

    extrinsics, intrinsics = get_cam_params(
        cam_pos,
        cam_look_at,
        width=width,
        height=height,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
    )

    # Check that fx and fy are different (non-square pixels)
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]

    # They should be different due to different apertures
    assert not torch.allclose(fx, fy, atol=1e-2)


@pytest.mark.general
def test_get_cam_params_camera_directions():
    """Test that camera coordinate frame is correct."""
    # Camera at origin looking down -x axis
    cam_pos = torch.tensor([[0.0, 0.0, 1.0]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    extrinsics, _ = get_cam_params(cam_pos, cam_look_at)

    # Extract rotation matrix
    R = extrinsics[0, :3, :3]

    # Camera front should point toward look_at
    cam_front = cam_look_at - cam_pos
    cam_front = cam_front / torch.norm(cam_front)

    # Third column of R should be camera front (in camera convention)
    R_front = R[:, 2]

    # They should be aligned (or anti-aligned depending on convention)
    dot = torch.dot(R_front, cam_front[0])
    assert torch.abs(dot) > 0.9  # Nearly parallel or anti-parallel


@pytest.mark.general
def test_get_cam_params_principal_point():
    """Test that principal point is at image center."""
    cam_pos = torch.tensor([[1.0, 0.0, 0.5]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    width, height = 640, 480

    _, intrinsics = get_cam_params(
        cam_pos,
        cam_look_at,
        width=width,
        height=height,
    )

    # Check principal point (cx, cy)
    cx = intrinsics[0, 0, 2]
    cy = intrinsics[0, 1, 2]

    assert torch.allclose(cx, torch.tensor(width * 0.5), atol=1e-4)
    assert torch.allclose(cy, torch.tensor(height * 0.5), atol=1e-4)


@pytest.mark.general
def test_get_cam_params_focal_length():
    """Test focal length computation."""
    cam_pos = torch.tensor([[1.0, 0.0, 0.5]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    width = 640
    focal_length = 24
    horizontal_aperture = 20.955

    _, intrinsics = get_cam_params(
        cam_pos,
        cam_look_at,
        width=width,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
    )

    # Check that fx matches expected formula
    expected_fx = width * focal_length / horizontal_aperture
    fx = intrinsics[0, 0, 0]

    assert torch.allclose(fx, torch.tensor(expected_fx), atol=1e-4)


@pytest.mark.general
def test_get_cam_params_orthogonality():
    """Test that camera coordinate axes are orthogonal."""
    cam_pos = torch.tensor([[2.0, 3.0, 4.0]])
    cam_look_at = torch.tensor([[0.0, 0.0, 0.0]])

    extrinsics, _ = get_cam_params(cam_pos, cam_look_at)

    R = extrinsics[0, :3, :3]

    # Extract the three axes
    right = R[:, 0]
    up = R[:, 1]
    front = R[:, 2]

    # Check orthogonality
    assert torch.allclose(torch.dot(right, up), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(torch.dot(right, front), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(torch.dot(up, front), torch.tensor(0.0), atol=1e-5)

    # Check unit length
    assert torch.allclose(torch.norm(right), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.norm(up), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.norm(front), torch.tensor(1.0), atol=1e-5)
