"""Tests for metasim.utils.math module.

These are pure unit tests for mathematical functions that don't require a simulator.
All tests are marked @pytest.mark.general.
"""

import math

import pytest
import torch

from metasim.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    wrap_to_pi,
)


@pytest.mark.general
def test_normalize_unit_vector():
    """Test normalize with a standard vector."""
    x = torch.tensor([[3.0, 4.0, 0.0]])
    normalized = normalize(x)
    assert torch.allclose(normalized, torch.tensor([[0.6, 0.8, 0.0]]), atol=1e-6)
    assert torch.allclose(torch.norm(normalized, dim=-1), torch.tensor([1.0]), atol=1e-6)


@pytest.mark.general
def test_normalize_near_zero():
    """Test normalize with near-zero vector (division by zero protection)."""
    x = torch.tensor([[1e-12, 1e-12, 1e-12]])
    normalized = normalize(x, eps=1e-9)
    # Should not crash, result should be finite
    assert torch.all(torch.isfinite(normalized))


@pytest.mark.general
def test_normalize_batch():
    """Test normalize with batched vectors."""
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    normalized = normalize(x)
    norms = torch.norm(normalized, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


@pytest.mark.general
def test_wrap_to_pi_basic():
    """Test wrap_to_pi with basic angles."""
    # Test angles in range
    angles = torch.tensor([0.0, math.pi / 2, -math.pi / 2])
    wrapped = wrap_to_pi(angles)
    assert torch.allclose(wrapped, angles, atol=1e-6)


@pytest.mark.general
def test_wrap_to_pi_edge_cases():
    """Test wrap_to_pi with edge cases at ±π."""
    # Test boundary conditions
    angles = torch.tensor([math.pi, -math.pi, 3 * math.pi, -3 * math.pi])
    wrapped = wrap_to_pi(angles)

    # π should map to π, -π should map to -π (or close to it)
    # Odd multiples of π should map to ±π
    assert torch.allclose(wrapped[0], torch.tensor(math.pi), atol=1e-6)
    assert torch.allclose(wrapped[1], torch.tensor(-math.pi), atol=1e-6)
    assert torch.allclose(wrapped[2], torch.tensor(math.pi), atol=1e-6)
    assert torch.allclose(wrapped[3], torch.tensor(-math.pi), atol=1e-6)


@pytest.mark.general
def test_wrap_to_pi_large_angles():
    """Test wrap_to_pi with large angles."""
    angles = torch.tensor([10 * math.pi, -10 * math.pi])
    wrapped = wrap_to_pi(angles)
    # All results should be in [-π, π]
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped <= math.pi)


@pytest.mark.general
def test_quat_mul_identity():
    """Test quaternion multiplication with identity quaternion."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
    q2 = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # 90° rotation around x

    result = quat_mul(q1, q2)
    assert torch.allclose(result, q2, atol=1e-4)

    result = quat_mul(q2, q1)
    assert torch.allclose(result, q2, atol=1e-4)


@pytest.mark.general
def test_quat_mul_inverse():
    """Test that q * q_inv = identity."""
    q = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # 90° rotation around x
    q_inv = quat_inv(q)

    result = quat_mul(q, q_inv)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # Check if result is close to identity (accounting for sign ambiguity)
    assert torch.allclose(torch.abs(result), torch.abs(identity), atol=1e-4)


@pytest.mark.general
def test_quat_inv_basic():
    """Test quaternion inverse."""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
    q_inv = quat_inv(q)

    # Check w stays same, x,y,z flip sign
    assert torch.allclose(q_inv[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(q_inv[1], torch.tensor([0.7071, -0.7071, 0.0, 0.0]), atol=1e-4)


@pytest.mark.general
def test_quat_from_euler_xyz_zero():
    """Test Euler to quaternion conversion with zero angles."""
    roll = torch.tensor([0.0])
    pitch = torch.tensor([0.0])
    yaw = torch.tensor([0.0])

    quat = quat_from_euler_xyz(roll, pitch, yaw)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(quat, identity, atol=1e-6)


@pytest.mark.general
def test_quat_from_euler_xyz_90deg():
    """Test Euler to quaternion conversion with 90° rotations."""
    # 90° roll (x-axis)
    roll = torch.tensor([math.pi / 2])
    pitch = torch.tensor([0.0])
    yaw = torch.tensor([0.0])

    quat = quat_from_euler_xyz(roll, pitch, yaw)
    expected = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # w, x, y, z

    assert torch.allclose(quat, expected, atol=1e-4)


@pytest.mark.general
def test_euler_xyz_from_quat_roundtrip():
    """Test that Euler->Quat->Euler roundtrips correctly."""
    # Test with angles in safe range (avoiding gimbal lock)
    roll_in = torch.tensor([0.5, 0.0, -0.3])
    pitch_in = torch.tensor([0.3, 0.2, 0.0])
    yaw_in = torch.tensor([1.0, -0.5, 0.8])

    quat = quat_from_euler_xyz(roll_in, pitch_in, yaw_in)
    roll_out, pitch_out, yaw_out = euler_xyz_from_quat(quat)

    # Note: euler_xyz_from_quat returns angles in [0, 2π), so we need to wrap
    # the input angles to the same range for comparison
    roll_in_wrapped = roll_in % (2 * math.pi)
    pitch_in_wrapped = pitch_in % (2 * math.pi)
    yaw_in_wrapped = yaw_in % (2 * math.pi)

    assert torch.allclose(roll_out, roll_in_wrapped, atol=1e-3)
    assert torch.allclose(pitch_out, pitch_in_wrapped, atol=1e-3)
    assert torch.allclose(yaw_out, yaw_in_wrapped, atol=1e-3)


@pytest.mark.general
def test_matrix_from_quat_identity():
    """Test quaternion to matrix conversion with identity."""
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mat = matrix_from_quat(quat)

    identity_mat = torch.eye(3).unsqueeze(0)
    assert torch.allclose(mat, identity_mat, atol=1e-6)


@pytest.mark.general
def test_quat_from_matrix_roundtrip():
    """Test that Quat->Matrix->Quat roundtrips correctly."""
    quat_in = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # 90° around x
    mat = matrix_from_quat(quat_in)
    quat_out = quat_from_matrix(mat)

    # Quaternions have sign ambiguity: q and -q represent the same rotation
    assert torch.allclose(torch.abs(quat_out), torch.abs(quat_in), atol=1e-4)


@pytest.mark.general
def test_quat_apply_identity():
    """Test quat_apply with identity quaternion."""
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    vec = torch.tensor([[1.0, 2.0, 3.0]])

    result = quat_apply(quat, vec)
    assert torch.allclose(result, vec, atol=1e-6)


@pytest.mark.general
def test_quat_apply_90deg_rotation():
    """Test quat_apply with 90° rotation around z-axis."""
    # 90° rotation around z-axis: (cos(45°), 0, 0, sin(45°))
    quat = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])
    vec = torch.tensor([[1.0, 0.0, 0.0]])  # x-axis

    result = quat_apply(quat, vec)
    expected = torch.tensor([[0.0, 1.0, 0.0]])  # Should rotate to y-axis

    assert torch.allclose(result, expected, atol=1e-4)


@pytest.mark.general
def test_axis_angle_from_quat_identity():
    """Test axis-angle from identity quaternion."""
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    axis_angle = axis_angle_from_quat(quat)

    # Identity should give zero rotation
    assert torch.allclose(axis_angle, torch.zeros(1, 3), atol=1e-6)


@pytest.mark.general
def test_axis_angle_from_quat_90deg():
    """Test axis-angle from 90° rotation."""
    # 90° rotation around x-axis
    quat = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
    axis_angle = axis_angle_from_quat(quat)

    # Should give [π/2, 0, 0]
    expected = torch.tensor([[math.pi / 2, 0.0, 0.0]])
    assert torch.allclose(axis_angle, expected, atol=1e-3)


@pytest.mark.general
def test_quat_error_magnitude_identity():
    """Test quaternion error magnitude between identical quaternions."""
    q1 = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
    q2 = q1.clone()

    error = quat_error_magnitude(q1, q2)
    assert torch.allclose(error, torch.zeros(1), atol=1e-6)


@pytest.mark.general
def test_quat_error_magnitude_opposite():
    """Test quaternion error magnitude between opposite rotations."""
    # Identity quaternion
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # 180° rotation around x
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    error = quat_error_magnitude(q1, q2)
    # Error should be π (180°)
    assert torch.allclose(error, torch.tensor([math.pi]), atol=1e-2)


@pytest.mark.general
def test_combine_frame_transforms_translation_only():
    """Test combining transforms with translation only."""
    t01 = torch.tensor([[1.0, 0.0, 0.0]])
    q01 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
    t12 = torch.tensor([[0.0, 1.0, 0.0]])
    q12 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation

    t02, q02 = combine_frame_transforms(t01, q01, t12, q12)

    expected_t = torch.tensor([[1.0, 1.0, 0.0]])
    assert torch.allclose(t02, expected_t, atol=1e-6)
    assert torch.allclose(q02, q01, atol=1e-6)


@pytest.mark.general
def test_subtract_frame_transforms_inverse():
    """Test that subtract is inverse of combine."""
    t01 = torch.tensor([[1.0, 2.0, 3.0]])
    q01 = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
    t02 = torch.tensor([[2.0, 3.0, 4.0]])
    q02 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    # Get t12, q12 by subtracting
    t12, q12 = subtract_frame_transforms(t01, q01, t02, q02)

    # Combine back should give t02, q02
    t02_reconstructed, q02_reconstructed = combine_frame_transforms(t01, q01, t12, q12)

    assert torch.allclose(t02_reconstructed, t02, atol=1e-4)
    # Quaternion sign ambiguity
    assert torch.allclose(torch.abs(q02_reconstructed), torch.abs(q02), atol=1e-4)


@pytest.mark.general
def test_compute_pose_error_zero():
    """Test pose error between identical poses."""
    t01 = torch.tensor([[1.0, 2.0, 3.0]])
    q01 = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])

    pos_error, rot_error = compute_pose_error(t01, q01, t01, q01, rot_error_type="axis_angle")

    assert torch.allclose(pos_error, torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(rot_error, torch.zeros(1, 3), atol=1e-4)


@pytest.mark.general
def test_compute_pose_error_translation():
    """Test pose error with pure translation."""
    t01 = torch.tensor([[0.0, 0.0, 0.0]])
    t02 = torch.tensor([[1.0, 0.0, 0.0]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity

    pos_error, rot_error = compute_pose_error(t01, q, t02, q, rot_error_type="axis_angle")

    assert torch.allclose(pos_error, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(rot_error, torch.zeros(1, 3), atol=1e-6)


@pytest.mark.general
def test_quat_operations_batch():
    """Test that quaternion operations work correctly with batches."""
    # Create batch of 10 random unit quaternions
    batch_size = 10
    quats = torch.randn(batch_size, 4)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    # Test quat_inv
    quats_inv = quat_inv(quats)

    # q * q_inv should be identity
    result = quat_mul(quats, quats_inv)

    # First component should be close to ±1, others close to 0
    assert torch.allclose(torch.abs(result[:, 0]), torch.ones(batch_size), atol=1e-4)
    assert torch.allclose(result[:, 1:], torch.zeros(batch_size, 3), atol=1e-4)


@pytest.mark.general
def test_matrix_quat_conversion_batch():
    """Test matrix-quaternion conversions with batches."""
    batch_size = 5

    # Create batch of random rotations
    angles = torch.rand(batch_size) * 2 * math.pi
    axes = torch.randn(batch_size, 3)
    axes = axes / torch.norm(axes, dim=-1, keepdim=True)

    # Convert to quaternions using Euler angles
    roll = angles * axes[:, 0]
    pitch = angles * axes[:, 1]
    yaw = angles * axes[:, 2]

    quats = quat_from_euler_xyz(roll, pitch, yaw)

    # Convert to matrices and back
    matrices = matrix_from_quat(quats)
    quats_back = quat_from_matrix(matrices)

    # Should match (up to sign)
    assert torch.allclose(torch.abs(quats), torch.abs(quats_back), atol=1e-3)
