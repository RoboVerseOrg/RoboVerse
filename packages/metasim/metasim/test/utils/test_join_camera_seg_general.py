"""``join_tensor_states`` must carry camera segmentation through the join.

The camera join concatenated rgb/depth/pos/quat_world/intrinsics but silently
dropped ``instance_id_seg`` / ``instance_seg`` and their ``*_id2label`` maps.
So a backend that populates segmentation (e.g. isaacsim) lost it the moment
the state passed through a parallel join (num_envs>1) — the same class of
silent-drop as the earlier rgb/depth-only fix.
"""

from __future__ import annotations

import pytest
import torch

from metasim.types import CameraState, TensorState
from metasim.utils.state import join_tensor_states


def _cam(fill: float) -> CameraState:
    """A single-env CameraState with rgb + both segmentation channels."""
    return CameraState(
        rgb=torch.full((1, 4, 4, 3), fill),
        depth=None,
        instance_id_seg=torch.full((1, 4, 4), fill),
        instance_id_seg_id2label={0: "background", 1: "cube"},
        instance_seg=torch.full((1, 4, 4), fill),
        instance_seg_id2label={0: "background", 1: "robot"},
    )


@pytest.mark.general
def test_join_carries_camera_segmentation_and_id2label():
    s0 = TensorState(objects={}, robots={}, cameras={"cam": _cam(0.0)})
    s1 = TensorState(objects={}, robots={}, cameras={"cam": _cam(1.0)})

    joined = join_tensor_states([s0, s1]).cameras["cam"]

    # segmentation tensors concatenated across the two single-env states
    assert joined.instance_id_seg is not None, "instance_id_seg dropped by join"
    assert joined.instance_seg is not None, "instance_seg dropped by join"
    assert joined.instance_id_seg.shape == (2, 4, 4)
    assert joined.instance_seg.shape == (2, 4, 4)
    assert torch.equal(joined.instance_id_seg[0], torch.zeros(4, 4))
    assert torch.equal(joined.instance_id_seg[1], torch.ones(4, 4))

    # id2label maps (per-env identical) preserved
    assert joined.instance_id_seg_id2label == {0: "background", 1: "cube"}
    assert joined.instance_seg_id2label == {0: "background", 1: "robot"}

    # rgb still concatenated (unchanged behaviour)
    assert joined.rgb.shape == (2, 4, 4, 3)
