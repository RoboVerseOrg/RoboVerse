"""Regression: FastTD3 must not update obs-normalization stats on replay batches.

``EmpiricalNormalization.forward`` folds the input into its running mean/var when
in training mode. FastTD3 already updates the stats once per rollout step (fresh
obs). The per-gradient-step normalisation of the *sampled replay batch* (and of
``next_observations``) was also running in training mode, double-counting stale
resampled obs and corrupting the input statistics with next-obs.

The fix brackets the replay normalisation with ``obs_normalizer.eval()`` /
``.train()``. These tests pin (1) that eval mode freezes the stats — the property
the fix relies on — and (2) that ``train.py`` brackets the replay normalise with
eval/train. The training script itself isn't runnable here (needs env + full
stack), so the loop wiring is checked at the source level.
"""

from __future__ import annotations

import pathlib

import pytest

_TRAIN = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "rl" / "fast_td3" / "train.py"


@pytest.mark.general
def test_empirical_normalization_eval_freezes_stats():
    pytest.importorskip("tensordict")  # fttd3_module imports tensordict
    import torch

    from roboverse_learn.rl.fast_td3.fttd3_module import EmpiricalNormalization

    norm = EmpiricalNormalization(shape=(4,), device="cpu")
    norm.train()
    x = torch.ones(8, 4)
    norm(x)
    count_after_rollout = int(norm.count)
    mean_after_rollout = norm._mean.clone()

    # Replay-style reuse in eval mode must NOT change the running stats.
    norm.eval()
    for _ in range(5):
        norm(x)
    assert int(norm.count) == count_after_rollout
    assert torch.equal(norm._mean, mean_after_rollout)

    # Back in train mode, a fresh batch updates again (rollout path).
    norm.train()
    norm(x)
    assert int(norm.count) > count_after_rollout


@pytest.mark.general
def test_replay_normalization_is_eval_bracketed_in_train_loop():
    src = _TRAIN.read_text()
    post = src[src.index('if global_step > cfg("learning_starts"):') :]
    assert "obs_normalizer.eval()" in post and "obs_normalizer.train()" in post, (
        "the replay-batch normalisation in the learning block must toggle "
        "obs_normalizer.eval()/.train() so the running stats are not updated on replay"
    )
    eval_at = post.index("obs_normalizer.eval()")
    norm_at = post.index('data["observations"] = normalize_obs(')
    train_at = post.index("obs_normalizer.train()", norm_at)
    assert eval_at < norm_at < train_at, (
        "the replay-batch normalize_obs calls must be bracketed by "
        "obs_normalizer.eval() ... obs_normalizer.train() so they don't update stats"
    )
