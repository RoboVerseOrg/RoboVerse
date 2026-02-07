from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from loguru import logger as log

from .contract import CompatTermError, TermContext, WarnOnce
from .utils import resolve_pattern_values


class BaseCompatActionTerm(Protocol):
    action_dim: int

    def apply(self, *, actions: torch.Tensor, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> None: ...


def _cfg_type_name(cfg: Any) -> str:
    return getattr(getattr(cfg, "__class__", None), "__name__", type(cfg).__name__)


@dataclass
class CompatNoOpActionTerm:
    """Consume action slice but do not write targets (best-effort fallback)."""

    name: str
    action_dim: int
    reason: str
    warn_once: WarnOnce

    def apply(self, *, actions: torch.Tensor, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> None:
        _ = actions
        _ = targets_by_asset
        self.warn_once.warning(
            f"action.{self.name}/noop",
            "action.{}: unsupported; consuming actions as no-op ({}).",
            self.name,
            self.reason,
        )


class _CompatJointActionTermBase:
    """Shared implementation for joint-space action terms."""

    def __init__(self, *, name: str, cfg: Any, env: Any) -> None:
        self.name = name
        self.cfg = cfg
        self._env = env

        asset_name = getattr(cfg, "asset_name", None)
        joint_names = getattr(cfg, "joint_names", None)
        if asset_name is None or joint_names is None:
            raise CompatTermError(
                ctx=TermContext(kind="action", name=name),
                backend=getattr(getattr(env, "scenario", None), "simulator", None),
                message="missing required fields `asset_name` or `joint_names`.",
            )

        self.asset_name = asset_name
        self.asset = env.scene[asset_name]
        self.joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=True)
        self.action_dim = len(self.joint_ids)

        self.scale = getattr(cfg, "scale", 1.0)
        self.clip = getattr(cfg, "clip", None)
        self._scale_buf: torch.Tensor | None = None

    def _asset_targets(self, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        resolved = getattr(self._env.scene, "resolve_asset_name", lambda x: x)(self.asset_name)
        if resolved not in targets_by_asset:
            targets_by_asset[resolved] = {}
        return targets_by_asset[resolved]

    def _ensure_target_channel(
        self,
        *,
        targets_by_asset: dict[str, dict[str, torch.Tensor]],
        channel: str,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        asset_targets = self._asset_targets(targets_by_asset)
        if channel not in asset_targets:
            if default is None:
                default = torch.zeros(
                    (self._env.num_envs, len(self.asset.joint_names)),
                    device=self._env.device,
                    dtype=torch.float32,
                )
            asset_targets[channel] = default
        return asset_targets[channel]

    def _ensure_scale(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._scale_buf is not None and self._scale_buf.device == device and self._scale_buf.dtype == dtype:
            return self._scale_buf

        scale = self.scale
        if isinstance(scale, (float, int)):
            self._scale_buf = torch.tensor(float(scale), device=device, dtype=dtype)
            return self._scale_buf
        if isinstance(scale, dict):
            resolved = resolve_pattern_values(scale, self.asset.joint_names, default=1.0)
            full = torch.tensor([float(resolved[n]) for n in self.asset.joint_names], device=device, dtype=dtype)
            ids = torch.as_tensor(self.joint_ids, device=device, dtype=torch.long)
            self._scale_buf = full[ids]
            return self._scale_buf

        self._scale_buf = torch.as_tensor(scale, device=device, dtype=dtype)
        # If a per-joint scale vector is provided for the full articulation, slice it.
        if self._scale_buf.ndim == 1 and self._scale_buf.numel() == len(self.asset.joint_names):
            ids = torch.as_tensor(self.joint_ids, device=device, dtype=torch.long)
            self._scale_buf = self._scale_buf[ids]
        return self._scale_buf

    def _normalize_actions_for_target(self, *, actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = actions.to(device=target.device, dtype=target.dtype)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if a.shape[1] != self.action_dim:
            # Shape mismatches are handled at manager level; be defensive here.
            a = (
                a[:, : self.action_dim]
                if a.shape[1] > self.action_dim
                else torch.cat(
                    [a, torch.zeros((a.shape[0], self.action_dim - a.shape[1]), device=a.device, dtype=a.dtype)],
                    dim=1,
                )
            )
        return a

    def _apply_scaled(self, *, target: torch.Tensor, actions: torch.Tensor) -> None:
        a = self._normalize_actions_for_target(actions=actions, target=target)
        if isinstance(self.scale, (float, int)):
            scaled = a * float(self.scale)
        else:
            scaled = a * self._ensure_scale(device=a.device, dtype=a.dtype)
        target[:, self.joint_ids] = target[:, self.joint_ids] + scaled

        if self.clip is not None:
            try:
                low, high = self.clip
                target[:, self.joint_ids].clamp_(float(low), float(high))
            except (TypeError, ValueError):
                log.warning("action.{}: invalid clip value {!r}; skipping clamp.", self.name, self.clip)


class CompatJointPositionActionTerm(_CompatJointActionTermBase):
    """Compat implementation of IsaacLab joint-position action terms.

    Provides the `_offset` tensor attribute used by some IsaacLab-style randomization
    events when `use_default_offset=True`.
    """

    def __init__(self, *, name: str, cfg: Any, env: Any) -> None:
        super().__init__(name=name, cfg=cfg, env=env)
        self.use_default_offset = bool(getattr(cfg, "use_default_offset", False))
        # Lazy-initialized once `asset.data` exists.
        self._offset_buf: torch.Tensor | None = None

    def _ensure_offset(self) -> torch.Tensor:
        if self._offset_buf is not None:
            return self._offset_buf
        if not self.use_default_offset:
            self._offset_buf = torch.zeros((self._env.num_envs, len(self.asset.joint_names)), device=self._env.device)
            return self._offset_buf

        # IMPORTANT: keep a reference to the articulation buffer so event code that writes to
        # `env.action_manager.get_term(...)._offset[...] = ...` stays in sync.
        self._offset_buf = self.asset.data.default_joint_pos
        return self._offset_buf

    @property
    def _offset(self) -> torch.Tensor:
        return self._ensure_offset()

    def apply(self, *, actions: torch.Tensor, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> None:
        default = None
        if self.use_default_offset:
            default = self._ensure_offset().clone().to(device=self._env.device, dtype=torch.float32)
        target = self._ensure_target_channel(
            targets_by_asset=targets_by_asset, channel="dof_pos_target", default=default
        )
        self._apply_scaled(target=target, actions=actions)


class CompatJointVelocityActionTerm(_CompatJointActionTermBase):
    """Compat implementation of IsaacLab joint-velocity action terms."""

    def apply(self, *, actions: torch.Tensor, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> None:
        target = self._ensure_target_channel(targets_by_asset=targets_by_asset, channel="dof_vel_target")
        self._apply_scaled(target=target, actions=actions)


class CompatJointEffortActionTerm(_CompatJointActionTermBase):
    """Compat implementation of IsaacLab joint-effort action terms."""

    def apply(self, *, actions: torch.Tensor, targets_by_asset: dict[str, dict[str, torch.Tensor]]) -> None:
        target = self._ensure_target_channel(targets_by_asset=targets_by_asset, channel="dof_effort_target")
        self._apply_scaled(target=target, actions=actions)


class ActionTermRegistry:
    """Build compat action term instances from IsaacLab-style action cfg objects."""

    def __init__(self, *, env: Any, strict: bool, warn_once: WarnOnce) -> None:
        self._env = env
        self._strict = strict
        self._warn_once = warn_once

    def build(self, *, name: str, cfg: Any) -> BaseCompatActionTerm:
        ctx = TermContext(kind="action", name=name)
        backend = getattr(getattr(self._env, "scenario", None), "simulator", None)

        def _unsupported_noop_or_raise(*, term_cls: type, message: str) -> BaseCompatActionTerm:
            if self._strict:
                raise CompatTermError(ctx=ctx, backend=backend, message=message)
            dim = term_cls(name=name, cfg=cfg, env=self._env).action_dim
            return CompatNoOpActionTerm(
                name=name, action_dim=dim, reason=f"backend={backend}", warn_once=self._warn_once
            )

        type_name = _cfg_type_name(cfg)

        # Explicit mappings for common IsaacLab action cfgs.
        if type_name == "JointPositionActionCfg" or "Position" in type_name:
            return CompatJointPositionActionTerm(name=name, cfg=cfg, env=self._env)

        if type_name == "JointVelocityActionCfg" or "Velocity" in type_name:
            # Only some backends read velocity targets today.
            if backend not in {"mujoco", "newton"}:
                return _unsupported_noop_or_raise(
                    term_cls=CompatJointVelocityActionTerm,
                    message="joint-velocity actions are not supported on this backend.",
                )
            return CompatJointVelocityActionTerm(name=name, cfg=cfg, env=self._env)

        if type_name == "JointEffortActionCfg" or "Effort" in type_name or "Torque" in type_name:
            if backend not in {"newton"}:
                return _unsupported_noop_or_raise(
                    term_cls=CompatJointEffortActionTerm,
                    message="joint-effort actions are not supported on this backend.",
                )
            return CompatJointEffortActionTerm(name=name, cfg=cfg, env=self._env)

        # Backward-compatible duck-typing: cfg has the minimal fields; assume position action.
        if hasattr(cfg, "asset_name") and hasattr(cfg, "joint_names"):
            self._warn_once.warning(
                f"{ctx.label()}/unknown_cfg",
                "{}: unknown action cfg type {}. Treating as JointPositionActionCfg.",
                ctx.label(),
                type(cfg),
            )
            return CompatJointPositionActionTerm(name=name, cfg=cfg, env=self._env)

        # Unknown action term: cannot infer action_dim => cannot preserve action space.
        if self._strict:
            raise CompatTermError(ctx=ctx, backend=backend, message=f"unsupported action term cfg type: {type(cfg)}")
        log.error("Unsupported action term '{}' ({}); cannot infer action dimension.", name, type(cfg))
        raise CompatTermError(ctx=ctx, backend=backend, message=f"unsupported action term cfg type: {type(cfg)}")
