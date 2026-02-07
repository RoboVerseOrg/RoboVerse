from __future__ import annotations

"""Term execution contract for IsaacLab manager-based tasks running on MetaSim.

This module centralizes:
- strict vs best-effort behavior (raise vs warn+skip),
- kwargs filtering (IsaacLab-style `func(env, **params)` signatures),
- output normalization to match common IsaacLab conventions:
  - observations: (num_envs, D) (or (num_envs,) -> (num_envs, 1)),
  - rewards/terminations: (num_envs,) (or scalar / (num_envs, 1) -> (num_envs,)),
  - all outputs live on `env.device`.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from loguru import logger as log

from .utils import filter_kwargs_for_callable

try:  # optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TermContext:
    """Human-readable identity for a manager term."""

    kind: str
    name: str
    group: str | None = None

    def label(self) -> str:
        if self.group is None:
            return f"{self.kind}.{self.name}"
        return f"{self.kind}[{self.group}].{self.name}"


class CompatTermError(RuntimeError):
    """Raised when strict term execution fails or a term violates the contract."""

    def __init__(self, *, ctx: TermContext, backend: str | None, message: str):
        self.ctx = ctx
        self.backend = backend
        self.detail = message
        backend_str = backend or "unknown-backend"
        super().__init__(f"{ctx.label()} ({backend_str}): {message}")


class WarnOnce:
    """Deduplicate warnings by key (per-env instance)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def warning(self, key: str, message: str, *args: Any) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        log.warning(message, *args)


def safe_call(
    func: Callable[..., Any], *, host_env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce, **kwargs: Any
):
    """Call a term function with IsaacLab-style kwargs filtering and strict/best-effort error handling."""
    call_kwargs = filter_kwargs_for_callable(func, kwargs)
    try:
        return func(**call_kwargs)
    except Exception as exc:
        if strict:
            raise CompatTermError(
                ctx=ctx,
                backend=getattr(getattr(host_env, "scenario", None), "simulator", None),
                message=f"term call failed: {exc}",
            ) from exc
        warn_once.warning(f"{ctx.label()}/call_failed", "{}: term call failed ({}). Skipping.", ctx.label(), exc)
        return None


def to_tensor(value: Any, *, env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce) -> torch.Tensor | None:
    """Best-effort conversion of term outputs into torch.Tensor on env.device."""
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        tensor = value
    elif np is not None and isinstance(value, np.ndarray):  # pragma: no cover
        tensor = torch.from_numpy(value)
    elif isinstance(value, (bool, int, float)):
        tensor = torch.tensor(value)
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:
            if strict:
                raise CompatTermError(
                    ctx=ctx,
                    backend=getattr(getattr(env, "scenario", None), "simulator", None),
                    message=f"expected tensor-like output, got {type(value)}",
                ) from exc
            warn_once.warning(
                f"{ctx.label()}/non_tensor",
                "{}: expected tensor-like output, got {}. Skipping.",
                ctx.label(),
                type(value),
            )
            return None

    device = getattr(env, "device", None)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def ensure_float32(
    t: torch.Tensor, *, env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce
) -> torch.Tensor | None:
    if t.is_floating_point():
        return t
    # IsaacLab reward terms often return integer counts (e.g., contact counts). IsaacLab's
    # reward manager implicitly casts these to float during accumulation. Mirror that behavior
    # and avoid spamming warnings for common, well-typed integer rewards.
    if ctx.kind == "reward" and not strict:
        return t.to(dtype=torch.float32)
    if strict:
        raise CompatTermError(
            ctx=ctx,
            backend=getattr(getattr(env, "scenario", None), "simulator", None),
            message=f"expected floating-point tensor, got dtype={t.dtype}",
        )
    warn_once.warning(
        f"{ctx.label()}/cast_float",
        "{}: expected floating-point tensor, got dtype={}. Casting to float32.",
        ctx.label(),
        t.dtype,
    )
    return t.to(dtype=torch.float32)


def normalize_batch_2d(
    t: torch.Tensor, *, env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce
) -> torch.Tensor | None:
    """Normalize observation-like outputs to (num_envs, D)."""
    num_envs = int(getattr(env, "num_envs", 1))
    if t.ndim == 0:
        return t.reshape(1, 1).repeat(num_envs, 1)
    if t.ndim == 1 and t.shape[0] == num_envs:
        return t.unsqueeze(-1)
    if t.ndim == 2 and t.shape[0] == num_envs:
        return t

    if strict:
        raise CompatTermError(
            ctx=ctx,
            backend=getattr(getattr(env, "scenario", None), "simulator", None),
            message=f"expected shape (num_envs, D) or (num_envs,), got shape={tuple(t.shape)}",
        )
    warn_once.warning(
        f"{ctx.label()}/bad_shape",
        "{}: expected shape (num_envs, D) or (num_envs,), got shape={}. Skipping.",
        ctx.label(),
        tuple(t.shape),
    )
    return None


def normalize_batch_1d(
    t: torch.Tensor, *, env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce
) -> torch.Tensor | None:
    """Normalize reward/termination-like outputs to (num_envs,)."""
    num_envs = int(getattr(env, "num_envs", 1))
    if t.ndim == 0:
        return t.reshape(1).repeat(num_envs)
    if t.ndim == 1 and t.shape[0] == num_envs:
        return t
    if t.ndim == 2 and t.shape[0] == num_envs and t.shape[1] == 1:
        return t[:, 0]

    if strict:
        raise CompatTermError(
            ctx=ctx,
            backend=getattr(getattr(env, "scenario", None), "simulator", None),
            message=f"expected shape (num_envs,) or (num_envs,1), got shape={tuple(t.shape)}",
        )
    warn_once.warning(
        f"{ctx.label()}/bad_shape",
        "{}: expected shape (num_envs,) or (num_envs,1), got shape={}. Skipping.",
        ctx.label(),
        tuple(t.shape),
    )
    return None


def normalize_bool(
    t: torch.Tensor, *, env: Any, ctx: TermContext, strict: bool, warn_once: WarnOnce
) -> torch.Tensor | None:
    """Normalize termination outputs to bool tensor shaped (num_envs,)."""
    t1 = normalize_batch_1d(t, env=env, ctx=ctx, strict=strict, warn_once=warn_once)
    if t1 is None:
        return None
    if t1.dtype == torch.bool:
        return t1
    if strict and t1.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8):
        raise CompatTermError(
            ctx=ctx,
            backend=getattr(getattr(env, "scenario", None), "simulator", None),
            message=f"expected bool-like tensor, got dtype={t1.dtype}",
        )
    return t1.to(dtype=torch.bool)
