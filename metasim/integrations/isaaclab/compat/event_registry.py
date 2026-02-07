from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .capabilities import CAPABILITIES
from .contract import CompatTermError, TermContext, WarnOnce


def _callable_path(func: Callable[..., Any]) -> str:
    mod = getattr(func, "__module__", "") or ""
    name = getattr(func, "__name__", repr(func))
    return f"{mod}.{name}"


class EventTermRegistry:
    """Backend-gated wrappers for IsaacLab-style event functions."""

    def __init__(self, *, env: Any, strict: bool, warn_once: WarnOnce) -> None:
        self._env = env
        self._strict = strict
        self._warn_once = warn_once

    @dataclass(frozen=True)
    class SupportSpec:
        supported_backends: set[str] | None
        reason: str
        capability: str | None = None

    # Declared per-event support map (callable-path -> support spec).
    _SUPPORT_MAP: dict[str, SupportSpec] = {
        # BeyondMimic: requires PhysX view access (`root_physx_view`) which MetaSim compat does not expose.
        "roboverse_pack.tasks.beyondmimic.isaaclab.mdp.events.randomize_rigid_body_com": SupportSpec(
            supported_backends=None,
            reason="requires IsaacSim PhysX views (`root_physx_view`) which are not available in MetaSim compat.",
            capability=CAPABILITIES.ISAACSIM_PHYSX_VIEWS,
        ),
    }

    def _is_supported(self, func: Callable[..., Any]) -> tuple[bool, str]:
        backend = getattr(getattr(self._env, "scenario", None), "simulator", None)
        path = _callable_path(func)
        spec = self._SUPPORT_MAP.get(path)
        if spec is None:
            return True, ""
        if spec.capability is not None:
            support = CAPABILITIES.check(capability=spec.capability, backend=backend)
            if support.supported:
                return True, ""
            return False, spec.reason or support.reason

        if spec.supported_backends is None:
            return True, ""
        if backend in spec.supported_backends:
            return True, ""
        return False, spec.reason

    def wrap(self, *, name: str, term_cfg: Any) -> Callable[..., Any] | None:
        func = getattr(term_cfg, "func", None)
        if not callable(func):
            return None

        supported, reason = self._is_supported(func)
        if supported:
            return func

        backend = getattr(getattr(self._env, "scenario", None), "simulator", None)
        ctx = TermContext(kind="event", name=name)
        if self._strict:
            raise CompatTermError(
                ctx=ctx, backend=backend, message=f"unsupported on this backend: {reason} ({_callable_path(func)})"
            )

        key = f"event.{name}/unsupported"

        def _noop(*_args: Any, **_kwargs: Any):
            self._warn_once.warning(
                key,
                "{}: unsupported on backend '{}' ({}). No-op.",
                ctx.label(),
                backend,
                reason or _callable_path(func),
            )
            return None

        return _noop
