# Copyright (c) ManiSkill2-real2sim contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ManiSkill2-real2sim (https://github.com/simpler-env/ManiSkill2_real2sim), the sim
# stack behind SimplerEnv (https://github.com/simpler-env/SimplerEnv).
# Changes: the visual-matching (greenscreen) compositing of
#   `mani_skill2_real2sim/envs/custom_scenes/base_env.py::get_obs` is reproduced as a standalone
#   pure-numpy/cv2 function with no dependency on the upstream package; the pixel math is
#   unchanged (kept bit-for-bit faithful for parity).
# Full license: roboverse_pack/tasks/simpler_env/_native/control/LICENSE
"""Visual-matching (greenscreen) overlay — vendored from SimplerEnv, pixel-exact.

SimplerEnv's "visual matching" eval composites a real-world inpainted photo behind the
simulated robot+object: background pixels are replaced by the overlay image while the
foreground (robot links, target objects, other articulation links) stays sim-rendered.
The mask comes from the actor-level segmentation channel.

This reproduces ``mani_skill2_real2sim/envs/custom_scenes/base_env.py::get_obs`` (the
greenscreen block, lines ~349-399 @ ef7a4d4) exactly, as a standalone pure-numpy/cv2
function with NO dependency on the upstream package — so the native port can do
visual-matching without ``mani_skill2_real2sim``. Deterministic: given identical render +
segmentation it is bit-for-bit identical to upstream.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_overlay_image(rgb_overlay_path: str) -> np.ndarray:
    """Load a visual-matching overlay exactly as SimplerEnv does: BGR->RGB, /255 float."""
    return cv2.cvtColor(cv2.imread(rgb_overlay_path), cv2.COLOR_BGR2RGB) / 255  # (H,W,3) float64


def apply_greenscreen_overlay(
    color: np.ndarray,
    actor_seg: np.ndarray,
    overlay_img: np.ndarray,
    *,
    foreground_actor_ids: np.ndarray,
    mode: str = "background",
) -> np.ndarray:
    """Composite ``overlay_img`` behind the foreground, matching SimplerEnv's get_obs.

    Args:
        color: rendered RGB, shape (H,W,3) or (H,W,4); only [...,:3] is modified. Same dtype
            semantics as upstream (it writes float blend back into the Color array in place).
        actor_seg: actor-level segmentation, shape (H,W) (== ``Segmentation[...,1]``).
        overlay_img: overlay image in SimplerEnv convention (RGB, /255 float), any size; it is
            cv2.resize'd to the color image's (W,H).
        foreground_actor_ids: ids kept sim-rendered (robot link ids ++ target object actor ids
            ++ other articulation link ids), as produced by SimplerEnv get_obs.
        mode: "background" (overlay background, keep foreground) — the SimplerEnv default.

    Returns:
        The color array with [...,:3] composited (modified in place and returned).
    """
    if "background" not in mode and "debug" not in mode:
        raise NotImplementedError(mode)
    mask = np.ones_like(actor_seg, dtype=np.float32)
    if ("object" not in mode) or ("debug" in mode):
        mask[np.isin(actor_seg, foreground_actor_ids)] = 0.0
    else:
        # "overlay everything except robot" path needs only robot ids; caller passes them as fg
        mask[np.isin(actor_seg, foreground_actor_ids)] = 0.0
    mask = mask[..., np.newaxis]
    resized = cv2.resize(overlay_img, (color.shape[1], color.shape[0]))
    if "debug" not in mode:
        color[..., :3] = color[..., :3] * (1 - mask) + resized * mask
    else:
        color[..., :3] = color[..., :3] * 0.5 + resized * 0.5
    return color


def foreground_actor_ids(robot_link_ids, target_object_actor_ids, other_link_ids) -> np.ndarray:
    """Concatenate the three id groups exactly as SimplerEnv's get_obs does for the mask."""
    return np.concatenate([
        np.asarray(robot_link_ids, dtype=np.int32),
        np.asarray(target_object_actor_ids, dtype=np.int32),
        np.asarray(other_link_ids, dtype=np.int32),
    ])
