"""Basic test to verify test infrastructure is working."""

import pytest


def test_basic_setup():
    """Verify basic test setup is working."""
    assert True


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works."""
    assert len("hello") == 5
