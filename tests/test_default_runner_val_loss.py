"""Regression: the IL val-loss reduction must accumulate floats, not 0-d tensors.

The validation loop did ``val_losses.append(loss)`` (0-d CUDA tensors) then
``torch.mean(torch.tensor(val_losses))`` — ``torch.tensor`` over a list of
existing tensors emits a copy-construct UserWarning and is fragile for
CUDA/grad tensors. The train loop already uses ``.item()`` + ``np.mean``; the fix
makes the val loop consistent. The runner isn't importable/runnable here (needs a
model + dataloader), so the loop wiring is pinned at the source level and the
reduction pattern's warning-freeness is demonstrated directly.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import torch

_RUNNER = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "il" / "runners" / "default_runner.py"


@pytest.mark.general
def test_val_loss_reduction_uses_item_and_npmean():
    src = _RUNNER.read_text()
    assert "val_losses.append(loss.item())" in src, "val losses must be accumulated as python floats"
    assert "torch.tensor(val_losses)" not in src, "torch.tensor over a list of 0-d tensors is fragile; use np.mean"
    assert "np.mean(val_losses)" in src, "val loss should be averaged with np.mean (consistent with train loop)"


@pytest.mark.general
def test_item_then_npmean_is_warning_free_and_correct():
    """The new pattern (item() + np.mean) averages correctly with no warnings.

    (The old ``torch.tensor(list_of_0d_tensors)`` form is fragile for CUDA/grad
    tensors and warns on some torch versions — the fix sidesteps it entirely.)
    """
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        result = np.mean([loss.item() for loss in losses])
    assert result == 2.0
