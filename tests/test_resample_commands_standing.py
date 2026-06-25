"""Regression: resample_commands actually zeroes the standing-env velocity commands.

``resample_commands`` picks a ``rel_standing_envs`` fraction of the just-resampled
envs and zeroes their velocity command so the robot learns to stand still. The
zeroing was written as ``cfg.value[env_ids][final_env_ids, :] = 0.0`` — but
``cfg.value[env_ids]`` is advanced indexing (a copy), so the write was discarded
and the standing command was never applied. With ``rel_standing_envs = 1.0`` every
resampled env must end up with a zero command; this is backend-free (a stub env +
commands cfg, no simulator).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.general
def test_standing_envs_command_is_zeroed():
    import roboverse_pack.callback_funcs.humanoid.step_funcs as sf

    # Velocity ranges kept away from 0 so the pre-fix (un-zeroed) command is
    # detectably non-zero.
    ranges = SimpleNamespace(
        lin_vel_x=(0.5, 1.0),
        lin_vel_y=(0.5, 1.0),
        ang_vel_yaw=(0.5, 1.0),
        heading=(0.5, 1.0),
    )
    cfg = SimpleNamespace(
        value=None,
        num_commands=3,
        ranges=ranges,
        heading_command=False,
        rel_standing_envs=1.0,  # every resampled env becomes a standing env
        resampling_time=0.02,
    )
    env = SimpleNamespace(
        commands_manager=cfg,
        num_envs=8,
        device="cpu",
        step_dt=0.02,  # resampling_time / step_dt == 1 -> all envs resampled this step
        _episode_steps=torch.zeros(8, dtype=torch.long),
    )

    sf.resample_commands(env)

    # All 8 envs were resampled and are standing -> every command must be zero.
    assert cfg.value is not None
    assert cfg.value.shape == (8, 3)
    assert torch.count_nonzero(cfg.value) == 0, (
        "standing-env velocity commands were not zeroed — the advanced-index write "
        "is a no-op; index by absolute env id instead."
    )
