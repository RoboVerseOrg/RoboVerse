"""Tests for metasim.utils.gs_util module.

These are pure unit tests for gs (graphics/simulation) utility functions.
All tests are marked @pytest.mark.general.
"""

import math

import pytest

from metasim.utils.gs_util import quaternion_multiply


@pytest.mark.general
def test_quaternion_multiply_identity():
    """Test quaternion multiplication with identity quaternion."""
    # Identity quaternion in xyzw format
    identity = [0.0, 0.0, 0.0, 1.0]

    # Arbitrary quaternion (90° around x-axis)
    q = [0.7071, 0.0, 0.0, 0.7071]

    # Identity * q should equal q
    result1 = quaternion_multiply(identity, q)
    assert all(abs(a - b) < 1e-4 for a, b in zip(result1, q))

    # q * identity should equal q
    result2 = quaternion_multiply(q, identity)
    assert all(abs(a - b) < 1e-4 for a, b in zip(result2, q))


@pytest.mark.general
def test_quaternion_multiply_90deg_rotations():
    """Test quaternion multiplication of 90° rotations."""
    # 90° rotation around x-axis (xyzw format)
    qx = [0.7071, 0.0, 0.0, 0.7071]

    # 90° rotation around y-axis (xyzw format)
    qy = [0.0, 0.7071, 0.0, 0.7071]

    # Multiply them
    result = quaternion_multiply(qx, qy)

    # Result should be a valid unit quaternion
    magnitude = math.sqrt(sum(x**2 for x in result))
    assert abs(magnitude - 1.0) < 1e-4


@pytest.mark.general
def test_quaternion_multiply_inverse():
    """Test that q * q_conjugate is close to identity."""
    # Arbitrary quaternion
    q = [0.5, 0.5, 0.5, 0.5]

    # Conjugate (negate xyz, keep w)
    q_conj = [-0.5, -0.5, -0.5, 0.5]

    result = quaternion_multiply(q, q_conj)

    # Result should be close to identity or scaled identity
    # The w component should be dominant
    assert abs(result[3]) > 0.9  # w component should be large


@pytest.mark.general
def test_quaternion_multiply_commutativity():
    """Test that quaternion multiplication is non-commutative."""
    q1 = [0.7071, 0.0, 0.0, 0.7071]  # 90° around x
    q2 = [0.0, 0.7071, 0.0, 0.7071]  # 90° around y

    result1 = quaternion_multiply(q1, q2)
    result2 = quaternion_multiply(q2, q1)

    # Results should be different (quaternion multiplication is non-commutative)
    assert not all(abs(a - b) < 1e-4 for a, b in zip(result1, result2))


@pytest.mark.general
def test_quaternion_multiply_normalization():
    """Test that result remains normalized."""
    # Two normalized quaternions
    q1 = [0.5, 0.5, 0.5, 0.5]
    q2 = [0.0, 0.7071, 0.0, 0.7071]

    result = quaternion_multiply(q1, q2)

    # Result should also be normalized (unit quaternion)
    magnitude = math.sqrt(sum(x**2 for x in result))
    assert abs(magnitude - 1.0) < 1e-3


@pytest.mark.general
def test_quaternion_multiply_zero_rotation():
    """Test multiplication with zero rotation quaternions."""
    # Identity quaternion (zero rotation)
    identity = [0.0, 0.0, 0.0, 1.0]

    # Another identity
    result = quaternion_multiply(identity, identity)

    # Should remain identity
    assert all(abs(a - b) < 1e-6 for a, b in zip(result, identity))


@pytest.mark.general
def test_quaternion_multiply_180deg():
    """Test quaternion multiplication with 180° rotations."""
    # 180° rotation around x-axis
    q180_x = [1.0, 0.0, 0.0, 0.0]

    # 180° * 180° around same axis should give 360° = identity
    result = quaternion_multiply(q180_x, q180_x)

    # Should be close to identity (allowing for sign flip)
    identity = [0.0, 0.0, 0.0, 1.0]
    neg_identity = [0.0, 0.0, 0.0, -1.0]

    matches_identity = all(abs(a - b) < 1e-4 for a, b in zip(result, identity))
    matches_neg_identity = all(abs(a - b) < 1e-4 for a, b in zip(result, neg_identity))

    assert matches_identity or matches_neg_identity
