"""Regression: IL image dataset must import + sample without numba (numpy >= 2.0).

The DP/ACT image dataset and its sequence sampler jit-compile a couple of hot
loops with numba. numba requires numpy <= 1.26 and hard-fails to import on numpy
>= 2.0, which is now common -- previously crashing the whole IL training entry at
import time (``ImportError: Numba needs NumPy 1.26 or less``). The fix makes numba
optional with a pure-numpy fallback of identical semantics. These lock that:
import works regardless of numba, and the fallback sampler fills sequences
correctly (including the pad-before / pad-after edge replication).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.general
def test_dataset_and_sampler_import_without_numba():
    """The modules import on any numpy; the _HAS_NUMBA flag reflects availability."""
    import roboverse_learn.il.datasets.robot_image_dataset as ds
    import roboverse_learn.il.utils.sampler as sampler

    assert hasattr(ds, "_HAS_NUMBA")
    assert callable(ds.batch_sample_sequence)
    assert callable(sampler.create_indices)


@pytest.mark.general
def test_batch_sample_sequence_fills_and_pads_correctly():
    """The (possibly numba-free) batched sampler replicates the standard semantics."""
    from roboverse_learn.il.datasets.robot_image_dataset import batch_sample_sequence

    # input_arr: 5 timesteps of a 2-d feature.
    input_arr = np.arange(10, dtype=np.float32).reshape(5, 2)
    sequence_length = 3
    # indices rows: (buffer_start, buffer_end, sample_start, sample_end)
    # row A: full middle window [1:4) -> sample [0:3)
    # row B: pad-before (sample_start=1) window [0:2) -> sample [1:3), front padded
    # row C: pad-after  (sample_end=2)  window [3:5) -> sample [0:2), back padded
    indices = np.array([[1, 4, 0, 3], [0, 2, 1, 3], [3, 5, 0, 2]], dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    data = np.zeros((3, sequence_length, 2), dtype=np.float32)

    batch_sample_sequence(data, input_arr, indices, idx, sequence_length)

    # row A: timesteps 1,2,3 straight through
    assert np.allclose(data[0], input_arr[1:4])
    # row B: window goes to slots 1,2; slot 0 replicates slot 1 (pad-before)
    assert np.allclose(data[1, 1:3], input_arr[0:2])
    assert np.allclose(data[1, 0], data[1, 1])
    # row C: window goes to slots 0,1; slot 2 replicates slot 1 (pad-after)
    assert np.allclose(data[2, 0:2], input_arr[3:5])
    assert np.allclose(data[2, 2], data[2, 1])


@pytest.mark.general
def test_create_indices_runs_without_numba():
    """create_indices produces valid windows under the fallback jit (identity)."""
    from roboverse_learn.il.utils.sampler import create_indices

    episode_ends = np.array([5, 10], dtype=np.int64)
    episode_mask = np.ones(2, dtype=bool)
    indices = create_indices(episode_ends, sequence_length=3, episode_mask=episode_mask, debug=False)
    assert indices.shape[1] == 4  # (buf_start, buf_end, sample_start, sample_end)
    assert len(indices) > 0
    # buffer windows never exceed the data length.
    assert indices[:, 1].max() <= episode_ends[-1]
