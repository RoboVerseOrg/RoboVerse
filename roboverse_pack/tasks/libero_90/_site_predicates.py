# Copyright (c) 2023 Lifelong Robot Learning
# SPDX-License-Identifier: MIT
#
# Adapted from LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO), commit 8f1084e:
# SiteObject.in_box / SiteObject.under (libero/libero/envs/objects/site_object.py), ObjectState.check_ontop
# (libero/libero/envs/object_states/base_object_states.py) and the Open / Close / TurnOn / TurnOff thresholds
# of WoodenCabinet, Microwave and FlatStove (libero/libero/envs/objects/articulated_objects.py).
# Changes: batched over envs in torch; the site's pose and size come from metasim.utils.site_util.site_pose
# (the state plus the object's MJCF) instead of MuJoCo's data; the contact clauses of ``On`` (the object must
# touch the site's parent, or the other object) are not checked.
# Full license: roboverse_pack/tasks/libero_90/LICENSE.LIBERO
"""LIBERO's goal predicates on a site region, for the LIBERO-90 config ports.

A BDDL goal names a region by site: ``(In chefmate_8_frypan_1 wooden_two_layer_shelf_1_top_region)``,
``(On moka_pot_1 flat_stove_1_cook_region)``. These are LIBERO's tests for those two relations, with its
constants and its quirks kept, so a checker succeeds on the states LIBERO counted as success.
"""

from __future__ import annotations

import torch

from metasim.utils.site_util import SitePose

__all__ = [
    "CABINET_CLOSED_ABOVE",
    "CABINET_OPEN_BELOW",
    "MICROWAVE_CLOSED_ABOVE",
    "MICROWAVE_OPEN_BELOW",
    "STOVE_OFF_BELOW",
    "STOVE_ON_FROM",
    "object_on",
    "site_contains",
    "site_supports",
]

#: ``WoodenCabinet`` / ``WhiteCabinet``: a drawer joint is open below this position...
CABINET_OPEN_BELOW = -0.14
#: ... and closed above this one.
CABINET_CLOSED_ABOVE = 0.0
#: ``Microwave``: the door hinge is open below this angle...
MICROWAVE_OPEN_BELOW = -1.3
#: ... and closed above this one.
MICROWAVE_CLOSED_ABOVE = -0.005
#: ``FlatStove``: the knob joint turns the stove on from this position...
STOVE_ON_FROM = 0.5
#: ... and off below this one.
STOVE_OFF_BELOW = 0.0


def _size(site: SitePose, like: torch.Tensor) -> torch.Tensor:
    if site.size is None or len(site.size) != 3:
        raise ValueError(f"a region predicate needs a box site with a 3-number size, got size={site.size}")
    return torch.tensor(site.size, dtype=like.dtype, device=like.device)


def site_contains(site: SitePose, point: torch.Tensor) -> torch.Tensor:
    """``In(obj, region)``: ``point`` (N, 3) lies in the site's box, as LIBERO's ``SiteObject.in_box``.

    LIBERO bounds the box by ``|R @ size|`` around the site's position (axis-aligned) and lowers the floor
    by 1 cm.
    """
    half = (site.rot @ _size(site, point)).abs()  # (N, 3)
    lower = site.pos - half
    lower[:, 2] -= 0.01
    upper = site.pos + half
    return ((point > lower) & (point < upper)).all(dim=-1)


def site_supports(site: SitePose, point: torch.Tensor, *, height: float = 0.10) -> torch.Tensor:
    """``On(obj, region)``: ``point`` (N, 3) rests on the site, as LIBERO's ``SiteObject.under``.

    The offset is taken into the site frame with ``R @ (p - t)`` (LIBERO's expression; for the stove and
    cabinet sites R is a rotation about z or the identity), and must lie within the box's xy extent and
    between its top face (less 5 mm) and ``height`` above it.
    """
    size = _size(site, point)
    delta = (site.rot @ (point - site.pos).unsqueeze(-1)).squeeze(-1)
    in_height = (delta[:, 2] > size[2] - 0.005) & (delta[:, 2] < size[2] + height)
    return in_height & (delta[:, :2].abs() < size[:2]).all(dim=-1)


def object_on(point: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """``On(obj, other_obj)``: ``point`` (N, 3) is above ``base`` (N, 3) within 3 cm in xy.

    LIBERO's ``ObjectState.check_ontop``, without its contact clause.
    """
    return (base[:, 2] <= point[:, 2]) & ((point[:, :2] - base[:, :2]).norm(dim=-1) < 0.03)
