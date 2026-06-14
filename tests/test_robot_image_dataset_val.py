"""Regression: the IL validation dataset must not share IO buffers with train.

``RobotImageDataset.get_validation_dataset`` does ``copy.copy(self)`` (shallow),
so ``buffers`` / ``buffers_torch`` — the in-place destination of the
ndarray-index ``__getitem__`` fast path — were shared between the train and val
datasets. A val pass would scribble over a train batch (and vice-versa). The fix
re-allocates independent buffers for the val set. Backend-free: the dataset is
built via ``__new__`` with only the fields ``get_validation_dataset`` touches,
and ``SequenceSampler`` is stubbed so no replay buffer / zarr is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.general
def test_validation_dataset_has_independent_buffers(monkeypatch):
    import roboverse_learn.il.datasets.robot_image_dataset as mod

    # Stub the sampler so get_validation_dataset needs no real replay buffer.
    monkeypatch.setattr(mod, "SequenceSampler", lambda **kw: object())

    ds = mod.RobotImageDataset.__new__(mod.RobotImageDataset)
    ds.replay_buffer = object()
    ds.horizon = 3
    ds.pad_before = 0
    ds.pad_after = 0
    ds.train_mask = np.array([True, True, False])
    ds.buffers = {"state": np.zeros((2, 3, 4), dtype=np.float32)}
    ds.buffers_torch = {k: torch.from_numpy(v) for k, v in ds.buffers.items()}

    val = ds.get_validation_dataset()

    # Independent memory (not the same objects).
    assert val.buffers["state"] is not ds.buffers["state"]
    assert val.buffers_torch["state"] is not ds.buffers_torch["state"]

    # Writing into val's buffer must not corrupt the train buffer.
    val.buffers["state"][:] = 7.0
    assert (ds.buffers["state"] == 0.0).all(), "val buffer write leaked into the train buffer"
