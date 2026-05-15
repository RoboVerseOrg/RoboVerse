"""Placeholder for shared pick_place helper functions.

``approach_grasp_ceramic_teapot.py`` and ``approach_grasp_knife.py``
each carry a ``from .functions import *`` line that anticipates a
shared utility module living here. The two task bodies define their
own reward / control logic locally, so the import currently brings
in no symbols — keeping the file empty resolves the import without
introducing wildcards that other modules can't see.

If shared helpers do get extracted later (e.g. ``approach_reward``,
``joint2_lift_command``), add them here and they will become
available to those task files automatically.
"""

from __future__ import annotations

__all__: list[str] = []
