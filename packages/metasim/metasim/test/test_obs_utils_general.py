"""``ObsSaver`` never drops a frame silently: a video that lost frames no longer lines up with the steps."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from metasim.utils.obs_utils import ObsSaver

pytestmark = pytest.mark.general


def _state(with_rgb: bool):
    cam = SimpleNamespace(rgb=torch.zeros(2, 4, 4, 3, dtype=torch.uint8) if with_rgb else None)
    return SimpleNamespace(cameras={"cam": cam} if with_rgb else {})


def test_state_without_camera_rgb_disables_the_saver_once_with_a_warning(tmp_path):
    saver = ObsSaver(video_path=str(tmp_path / "v.mp4"))
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        saver.add(_state(with_rgb=False))
        saver.add(_state(with_rgb=False))
    finally:
        logger.remove(sink)
    assert len(warnings) == 1 and "no camera RGB" in warnings[0]
    assert saver.images == []
    saver.save()
    assert not (tmp_path / "v.mp4").exists()


def test_frames_with_rgb_are_kept_in_order(tmp_path):
    saver = ObsSaver(video_path=str(tmp_path / "v.mp4"))
    for _ in range(3):
        saver.add(_state(with_rgb=True))
    assert len(saver.images) == 3 and saver.images[0].shape[-1] == 3


def test_a_real_error_propagates(tmp_path):
    saver = ObsSaver(video_path=str(tmp_path / "v.mp4"))
    bad = SimpleNamespace(cameras={"cam": SimpleNamespace(rgb=torch.zeros(3))})  # not an image batch
    with pytest.raises(Exception):  # noqa: B017 - whatever torch raises, it must not be swallowed
        saver.add(bad)
