"""Temporal ensembling in ``BaseEvalRunner`` follows the ACT rule it is adapted from.

The overlapping chunks that cover a step are averaged with ``w_i = exp(-m * i)`` where ``i = 0`` is
the *oldest* prediction (the earliest chunk counts most). The chunks are selected by position: the
queries made at steps ``step - chunk + 1 .. step``. Before this test the sign was inverted (newest
counted most) and the selection was a non-zero test on the action values, ANDed across envs, so a
legitimately zero action in one env silently dropped that chunk for every env.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from roboverse_learn.il.runners.base_eval_runner import BaseEvalRunner

CHUNK, DIM, ENVS, T = 3, 2, 2, 6


def _runner():
    r = object.__new__(BaseEvalRunner)
    r.num_envs, r.device, r.step, r.k = ENVS, "cpu", 0, 0.01
    r.policy_cfg = SimpleNamespace(action_config=SimpleNamespace(action_chunk_steps=CHUNK, action_dim=DIM))
    r.all_time_actions = torch.zeros(ENVS, T, T + CHUNK, DIM)
    return r


def _chunk(step):
    """Chunk predicted at ``step``: value ``10 * step + offset`` so every (query, target) pair is distinct."""
    return torch.stack([torch.full((ENVS, DIM), float(10 * step + o)) for o in range(CHUNK)])


def test_weights_follow_act_oldest_prediction_counts_most():
    r = _runner()
    outs = []
    for step in range(3):
        r.step = step
        outs.append(r.get_temporal_agg_action(_chunk(step)))
    # step 2 is covered by the chunks queried at steps 0, 1, 2: their entries for step 2 are 2, 11, 20
    w = np.exp(-0.01 * np.arange(3))
    w = w / w.sum()
    expected = float(np.dot(w, [2.0, 11.0, 20.0]))
    assert torch.allclose(outs[2], torch.full((ENVS, DIM), expected), atol=1e-5)
    assert outs[2][0, 0] < 11.0, "the oldest chunk (value 2) must weigh more than the newest (value 20)"
    # step 0 is covered by its own chunk only
    assert torch.allclose(outs[0], torch.zeros(ENVS, DIM))


def test_a_zero_action_in_one_env_does_not_drop_the_chunk_for_the_others():
    r = _runner()
    for step in range(3):
        r.step = step
        c = _chunk(step)
        if step == 1:
            c[:, 1, :] = 0.0  # env 1's whole chunk at step 1 is exactly zero (a legitimate action)
        out = r.get_temporal_agg_action(c)
    w = np.exp(-0.01 * np.arange(3))
    w = w / w.sum()
    assert torch.allclose(out[0], torch.full((DIM,), float(np.dot(w, [2.0, 11.0, 20.0]))), atol=1e-5)
    assert torch.allclose(out[1], torch.full((DIM,), float(np.dot(w, [2.0, 0.0, 20.0]))), atol=1e-5)
