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


@pytest.mark.general
def test_quat_world_from_opengl_rotation_points_x_along_the_view_direction():
    """A camera looking down world +X with +Z up: OpenGL columns (right, up, backward) are -Y, +Z, -X; in the
    world convention (forward, left, up) that is the identity, so the quaternion is (1, 0, 0, 0)."""
    from metasim.utils.camera_util import camera_quat_world_from_opengl
    from metasim.utils.math import quat_apply, quat_from_matrix

    opengl = torch.tensor([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: right, up, backward
    quat = camera_quat_world_from_opengl(opengl)
    assert torch.allclose(quat, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    # a camera looking down world -Y: forward is what +X rotates to, up stays +Z
    forward, up = torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
    right = torch.cross(forward, up, dim=0)
    quat = camera_quat_world_from_opengl(torch.stack([right, up, -forward], dim=1))
    assert torch.allclose(quat_apply(quat[None], torch.tensor([[1.0, 0.0, 0.0]]))[0], forward, atol=1e-6)
    assert torch.allclose(quat_apply(quat[None], torch.tensor([[0.0, 0.0, 1.0]]))[0], up, atol=1e-6)
    from metasim.types import CameraState
    from metasim.utils.camera_util import camera_pose_fields
    from metasim.utils.math import convert_camera_frame_orientation_convention

    # the matrix form is the converter's rotation, so quat_opengl (the converter's inverse) undoes it
    via_converter = convert_camera_frame_orientation_convention(
        quat_from_matrix(torch.stack([right, up, -forward], dim=1)), origin="opengl", target="world"
    )
    assert torch.allclose(quat, via_converter, atol=1e-6) or torch.allclose(quat, -via_converter, atol=1e-6)
    back = CameraState(rgb=None, depth=None, quat_world=quat[None]).quat_opengl  # the inverse property
    assert torch.allclose(back, quat_from_matrix(torch.stack([right, up, -forward], dim=1))[None], atol=1e-6)
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3], c2w[:3, 3] = torch.stack([right, up, -forward], dim=1).double(), torch.tensor([1.0, 2.0, 3.0])
    fields = camera_pose_fields(torch.eye(3).double().numpy(), c2w.numpy())  # float64 matrices, as backends hold them
    assert fields["pos"].dtype == torch.float32 and torch.equal(fields["pos"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(fields["quat_world"][0], quat, atol=1e-6) and fields["intrinsics"].shape == (1, 3, 3)


@pytest.mark.general
def test_camera_pose_survives_the_nested_round_trip_and_the_demo_metadata():
    from types import SimpleNamespace

    from metasim.types import CameraState, ObjectState, TensorState
    from metasim.utils.save_util.save_util import _camera_metadata
    from metasim.utils.state import list_state_to_tensor, state_tensor_to_nested

    pos = torch.tensor([[1.0, -2.0, 0.5], [1.5, -2.0, 0.5]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    intrinsics = torch.tensor([[[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]] * 2)
    root = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6] * 2)
    state = TensorState(
        objects={
            "cube": ObjectState(root_state=root, body_names=None, body_state=None, joint_pos=None, joint_vel=None)
        },
        robots={},
        cameras={
            "cam": CameraState(
                rgb=torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
                depth=None,
                pos=pos,
                quat_world=quat,
                intrinsics=intrinsics,
            ),
            "bare": CameraState(rgb=torch.zeros(2, 4, 4, 3, dtype=torch.uint8), depth=None),
        },
        extras={},
    )
    handler = SimpleNamespace(get_joint_names=lambda n, sort=True: [], get_body_names=lambda n: [])
    nested = state_tensor_to_nested(handler, state)
    assert torch.equal(nested[1]["cameras"]["cam"]["pos"], pos[1]) and "pos" not in nested[1]["cameras"]["bare"]
    back = list_state_to_tensor(handler, nested)
    assert torch.equal(back.cameras["cam"].quat_world, quat) and torch.equal(back.cameras["cam"].intrinsics, intrinsics)
    assert back.cameras["bare"].pos is None
    meta = _camera_metadata(nested[0]["cameras"]["cam"])
    assert meta["cam_pos"] == [1.0, -2.0, 0.5] and meta["cam_intr"][0] == [100.0, 0.0, 32.0]
    assert meta["cam_look_at"] == pytest.approx([2.0, -2.0, 0.5]), "one metre ahead along +X, the forward axis"
    # cam_extr is the world-to-camera OpenCV matrix get_cam_params builds for the same look-at camera
    from metasim.utils.camera_util import camera_quat_world_from_opengl

    cam_pos, look_at = torch.tensor([[1.5, -1.5, 1.2]]), torch.tensor([[0.2, -0.2, 0.2]])
    expected, _ = get_cam_params(cam_pos, look_at)
    forward = look_at - cam_pos
    right = torch.cross(forward, torch.tensor([[0.0, 0.0, 1.0]]), dim=1)
    up = torch.cross(right, forward, dim=1)
    axes = torch.stack([right[0] / right.norm(), up[0] / up.norm(), -forward[0] / forward.norm()], dim=1)
    quat = camera_quat_world_from_opengl(axes)
    meta = _camera_metadata({"pos": cam_pos[0], "quat_world": quat, "intrinsics": intrinsics[0]})
    assert torch.allclose(torch.tensor(meta["cam_extr"]), expected[0], atol=1e-5)
    assert _camera_metadata(nested[0]["cameras"]["bare"]) == {
        "cam_pos": [],
        "cam_intr": [],
        "cam_look_at": [],
        "cam_extr": [],
    }
