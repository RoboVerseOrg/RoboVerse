"""Integration tests for gravity simulation."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sim("sapien3", "superdex")
def test_gravity(handler):
    """Test that gravity simulation is consistent."""
    state = handler.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 10.0]), atol=0.001, message="gravity initial")

    handler.simulate()

    state = handler.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 9.9950]), atol=0.001, message="gravity step 1")

    handler.simulate()

    state = handler.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    assert_close(pos, torch.Tensor([0, 0, 9.9800]), atol=0.001, message="gravity step 2")

    handler.simulate()

    state = handler.get_states(mode="dict")
    pos = state[0]["objects"]["cube"]["pos"]
    # SuperDex used to drift 1.8 mm from the analytic free fall here (implicit BACKWARD_EULER at a 1 ms
    # solver step, z=9.9569); at its default 5 ms solver step it is within this tolerance like the others.
    assert_close(pos, torch.Tensor([0, 0, 9.9551]), atol=0.001, message="gravity step 3")

    log.info(f"Gravity test passed for {handler.scenario.simulator}")
