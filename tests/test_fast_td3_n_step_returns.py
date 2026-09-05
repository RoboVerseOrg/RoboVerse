"""Regression tests for terminal-aware n-step returns in FastTD3."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("tensordict")

from roboverse_learn.rl.fast_td3.fttd3_module import SimpleReplayBuffer


def _buffer_with_ending(*, terminal: bool, truncation: bool) -> SimpleReplayBuffer:
    buffer = SimpleReplayBuffer(
        n_env=1,
        buffer_size=8,
        n_obs=1,
        n_act=1,
        n_critic_obs=1,
        n_steps=3,
        gamma=0.9,
        device="cpu",
    )
    # The second transition ends the episode.  The third reward is from the
    # next episode and must not leak into the sampled return.
    buffer.observations[0, :3, 0] = torch.tensor([0.0, 1.0, 2.0])
    buffer.next_observations[0, :3, 0] = torch.tensor([1.0, 2.0, 3.0])
    buffer.actions[0, :3, 0] = 0.0
    buffer.rewards[0, :3] = torch.tensor([1.0, 2.0, 100.0])
    buffer.dones[0, :3] = torch.tensor([0, int(terminal), 0])
    buffer.truncations[0, :3] = torch.tensor([0, int(truncation), 0])
    buffer.ptr = 3
    return buffer


@pytest.mark.general
@pytest.mark.parametrize(("terminal", "truncation"), [(True, False), (False, True)])
def test_n_step_return_keeps_end_reward_and_stops_after_it(terminal, truncation):
    """An n-step target includes the ending reward but excludes later data."""
    buffer = _buffer_with_ending(terminal=terminal, truncation=truncation)

    def zero_index(low, high, size, device=None, **kwargs):
        del low, high, kwargs
        return torch.zeros(size, dtype=torch.long, device=device)

    with patch("torch.randint", side_effect=zero_index):
        sample = buffer.sample(batch_size=1)

    # 1 + gamma * 2; the 100 reward must be excluded.
    assert sample["next", "rewards"].item() == pytest.approx(2.8)
    assert sample["next", "observations"].item() == pytest.approx(2.0)
    assert sample["next", "dones"].item() == int(terminal)
    assert sample["next", "truncations"].item() == int(truncation)
