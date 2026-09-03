# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: port of mjlab's `envs/mdp/actions/actions.py` `JointPositionAction` config surface to RoboVerse
#   dataclasses; the `clip(scale * raw + offset, ranges)` processing semantics follow mjlab, but the action is
#   applied in each RoboVerse task's `_apply_action` instead of by an mjlab action manager.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""Action term cfgs — mjlab JointPositionAction declarative-API port.

Mjlab declares action cfgs inside ``cfg.actions``:

    actions = {"joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,
        use_default_offset=True,
    )}

This module provides matching ``@dataclass`` cfg containers so an mjlab
config can be copy-pasted; the actual action processing in RoboVerse
still happens inside each task's ``_apply_action(processed_action)``
override (which reads these cfgs from ``env.cfg.actions`` if present).

Action processing semantics mirror mjlab/envs/mdp/actions/actions.py::BaseAction:

    processed = clip(scale * raw + offset, ranges)

where ``scale``, ``offset``, and ``ranges`` can be scalar, dict
(name → value), or full tensor.

Two scenarios this enables:

1. **Config-import parity** — a user pulls in mjlab's cfg dict, swaps the
   import to ``roboverse_pack.tasks.mjlab.mdp.actions``, and the task
   accepts the same param names.
2. **Hot-swap action scale / offset** — task ``_apply_action`` can do
   ``scale = env.cfg.actions["joint_pos"].resolved_scale(...)`` instead
   of hardcoding constants.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class JointPositionActionCfg:
    """Mjlab ``JointPositionActionCfg`` shape, RoboVerse-side.

    Fields mirror mjlab/envs/mdp/actions/actions.py::JointPositionActionCfg.
    The ``use_default_offset=True`` semantic (read default joint position
    from the entity's defaults) is honored in :func:`resolve_offset_t`.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] = ()
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    clip: dict[str, tuple[float, float]] | None = None
    use_default_offset: bool = False
    preserve_order: bool = False


def _resolve_named_values(
    spec: float | dict[str, float],
    names: Iterable[str],
    default: float = 0.0,
) -> list[float]:
    """Resolve mjlab-style ``{name-or-regex: value}`` to a per-name list."""
    names_list = list(names)
    if isinstance(spec, (int, float)):
        return [float(spec)] * len(names_list)
    if not spec:
        return [default] * len(names_list)
    import re

    out: list[float] = []
    for n in names_list:
        v: float | None = None
        for pat, val in spec.items():
            if pat == n or re.fullmatch(pat, n):
                v = float(val)
                break
        out.append(default if v is None else v)
    return out


def resolve_scale_t(cfg: JointPositionActionCfg, joint_names: Iterable[str], device) -> torch.Tensor:
    """Per-joint scale tensor for vectorized action processing."""
    vals = _resolve_named_values(cfg.scale, joint_names, default=1.0)
    return torch.tensor(vals, device=device, dtype=torch.float32)


def resolve_offset_t(
    cfg: JointPositionActionCfg,
    joint_names: Iterable[str],
    defaults: dict[str, float],
    device,
) -> torch.Tensor:
    """Per-joint offset tensor.

    Mjlab semantic: when ``use_default_offset=True``, offset comes from the
    entity's default-joint-position map (``defaults``). Otherwise the
    explicit ``cfg.offset`` is used (scalar or named).
    """
    if cfg.use_default_offset:
        names_list = list(joint_names)
        vals = [float(defaults.get(n, 0.0)) for n in names_list]
        return torch.tensor(vals, device=device, dtype=torch.float32)
    vals = _resolve_named_values(cfg.offset, joint_names, default=0.0)
    return torch.tensor(vals, device=device, dtype=torch.float32)


def resolve_clip_t(
    cfg: JointPositionActionCfg,
    joint_names: Iterable[str],
    device,
) -> torch.Tensor | None:
    """Per-joint (lo, hi) clip range tensor, or None if unbounded."""
    if cfg.clip is None or not cfg.clip:
        return None
    names_list = list(joint_names)
    import re

    out = []
    for n in names_list:
        match = None
        for pat, lo_hi in cfg.clip.items():
            if pat == n or re.fullmatch(pat, n):
                match = lo_hi
                break
        if match is None:
            out.append((-float("inf"), float("inf")))
        else:
            out.append((float(match[0]), float(match[1])))
    return torch.tensor(out, device=device, dtype=torch.float32)


class JointPositionActionManager:
    """Mjlab ``JointPositionAction`` port — joint-position control.

    Constructed by the task with a ``JointPositionActionCfg`` plus the
    list of actuator joint names + their default positions; processes
    every raw policy action into ``target = scale * raw + offset``
    (optionally clipped), subtracts ``env._encoder_bias`` if present
    (mjlab parity — ``JointPositionAction.apply_actions`` does the same),
    and writes the processed targets via the supplied callback.

    Usage in a task::

        self.action_manager = JointPositionActionManager(
            self,
            JointPositionActionCfg(
                entity_name="go1", actuator_names=(".*",),
                scale=0.5, use_default_offset=True,
            ),
            joint_names=_GO1_JOINT_NAMES,
            defaults=robot_default_joint_positions,
        )

        def _apply_action(self, raw):
            target = self.action_manager.process(raw)
            self.handler.physics.data.ctrl[:self.num_actions] = (
                target[0].cpu().numpy()
            )

    Hot-swap stiffness at runtime via ``action_manager.cfg.scale = ...``;
    the next ``process()`` re-resolves the per-joint scale tensor.
    """

    def __init__(
        self,
        env,
        cfg: JointPositionActionCfg,
        joint_names: Iterable[str],
        defaults: dict[str, float],
    ):
        self.env = env
        self.cfg = cfg
        self._joint_names = tuple(joint_names)
        self._defaults = dict(defaults)
        self._device = env.device
        self._scale: torch.Tensor | None = None
        self._offset: torch.Tensor | None = None
        self._clip: torch.Tensor | None = None
        self._rebuild_tensors()

    def _rebuild_tensors(self) -> None:
        self._scale = resolve_scale_t(self.cfg, self._joint_names, self._device)
        self._offset = resolve_offset_t(self.cfg, self._joint_names, self._defaults, self._device)
        self._clip = resolve_clip_t(self.cfg, self._joint_names, self._device)

    def process(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply mjlab BaseAction processing: ``scale * raw + offset``, clip, subtract encoder_bias.

        The encoder_bias is subtracted only if present on env. Returns the
        same shape as ``raw_action``.
        """
        if self._scale is None or self._offset is None:
            self._rebuild_tensors()
        scaled = self._scale.unsqueeze(0) * raw_action + self._offset.unsqueeze(0)
        if self._clip is not None:
            scaled = torch.clamp(scaled, self._clip[:, 0], self._clip[:, 1])
        # mjlab parity: JointPositionAction.apply_actions subtracts encoder_bias.
        bias = getattr(self.env, "_encoder_bias", None)
        if bias is not None and bias.shape == scaled.shape:
            scaled = scaled - bias
        return scaled

    # --- runtime mutation -------------------------------------------------

    def set_scale(self, scale: float | dict[str, float]) -> None:
        """Set the action scale at runtime and rebuild the scale tensor."""
        self.cfg.scale = scale
        self._scale = resolve_scale_t(self.cfg, self._joint_names, self._device)

    def set_offset(self, offset: float | dict[str, float]) -> None:
        """Set the action offset at runtime and rebuild the offset tensor."""
        self.cfg.offset = offset
        self.cfg.use_default_offset = False
        self._offset = resolve_offset_t(self.cfg, self._joint_names, self._defaults, self._device)
