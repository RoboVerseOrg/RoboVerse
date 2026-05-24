"""Unit tests for the IL<->RL weight bridge (roboverse_learn.fusion.bc_warmstart).

Pure-torch; no simulator. Verifies that copying a BC MLP's weights into an
rsl-rl-style actor reproduces the BC policy's outputs bit-for-bit, that
extraction round-trips, and that shape mismatches fail fast.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from roboverse_learn.fusion.bc_warmstart import (
    align_linear_layers,
    extract_actor_mlp_state_dict,
    load_bc_into_actor_critic,
)


def _mlp(in_dim, hidden, out_dim):
    dims = [in_dim, *hidden, out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class _FakeActorCritic(nn.Module):
    """Mimics rsl-rl: actor exposes its trunk under ``actor.mlp``."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.actor = nn.Module()
        self.actor.mlp = _mlp(in_dim, hidden, out_dim)
        self.critic = nn.Module()
        self.critic.mlp = _mlp(in_dim, hidden, 1)  # must stay untouched

    def act(self, x):
        return self.actor.mlp(x)


def test_bc_to_actor_reproduces_outputs():
    torch.manual_seed(0)
    in_dim, hidden, out_dim = 12, (64, 64), 4
    bc = _mlp(in_dim, hidden, out_dim)
    ac = _FakeActorCritic(in_dim, hidden, out_dim)

    x = torch.randn(8, in_dim)
    assert not torch.allclose(bc(x), ac.act(x)), "fresh nets should differ"

    critic_before = {k: v.clone() for k, v in ac.critic.state_dict().items()}
    n = load_bc_into_actor_critic(ac, bc)
    assert n == 3, f"expected 3 Linear layers copied, got {n}"

    # Actor now reproduces BC exactly.
    torch.testing.assert_close(ac.act(x), bc(x))
    # Critic untouched.
    for k, v in ac.critic.state_dict().items():
        torch.testing.assert_close(v, critic_before[k])


def test_extract_round_trips_into_fresh_mlp():
    torch.manual_seed(1)
    in_dim, hidden, out_dim = 9, (32,), 3
    ac = _FakeActorCritic(in_dim, hidden, out_dim)
    sd = extract_actor_mlp_state_dict(ac)

    fresh = nn.Module()
    fresh.mlp = _mlp(in_dim, hidden, out_dim)
    fresh.load_state_dict(sd)

    x = torch.randn(5, in_dim)
    torch.testing.assert_close(fresh.mlp(x), ac.act(x))


def test_shape_mismatch_fails_fast():
    bc = _mlp(12, (64, 64), 4)
    ac = _FakeActorCritic(12, (128, 128), 4)  # wrong hidden width
    try:
        align_linear_layers(bc, ac.actor)
    except ValueError as e:
        assert "Shape mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_layer_count_mismatch_fails_fast():
    bc = _mlp(12, (64,), 4)  # 2 layers
    ac = _FakeActorCritic(12, (64, 64), 4)  # 3 layers
    try:
        align_linear_layers(bc, ac.actor)
    except ValueError as e:
        assert "count mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on layer-count mismatch")


def test_non_strict_partial_copy():
    """Non-strict mode copies leading matching layers, stops at first mismatch."""
    torch.manual_seed(2)
    bc = _mlp(12, (64, 999), 4)  # diverges at layer 2
    ac = _FakeActorCritic(12, (64, 64), 4)
    n = load_bc_into_actor_critic(ac, bc, strict=False)
    assert n == 1, f"expected only the first layer copied, got {n}"
