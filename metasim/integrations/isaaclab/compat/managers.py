from __future__ import annotations

import copy
import inspect
from typing import Any

import torch

from .action_registry import ActionTermRegistry
from .contract import (
    CompatTermError,
    TermContext,
    WarnOnce,
    ensure_float32,
    normalize_batch_1d,
    normalize_batch_2d,
    normalize_bool,
    safe_call,
    to_tensor,
)
from .event_registry import EventTermRegistry
from .utils import is_class_term_cfg, is_term_cfg, iter_public_fields, resolve_scene_entity_cfgs


class CompatManagerError(CompatTermError):
    """Backwards-compatible name for compat failures."""


def _normalize_env_ids(env_ids: torch.Tensor | list[int] | slice | None) -> torch.Tensor | list[int] | slice:
    return slice(None) if env_ids is None else env_ids


def _instantiate_class_terms(
    *,
    cfg: Any,
    env: Any,
    strict: bool,
    warn_once: WarnOnce,
    kind: str,
) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    for name, term_cfg in iter_public_fields(cfg):
        if term_cfg is None or not is_class_term_cfg(term_cfg):
            continue
        term_cls = term_cfg.class_type
        try:
            terms[name] = term_cls(term_cfg, env)
        except Exception as exc:
            ctx = TermContext(kind=kind, name=name)
            if strict:
                raise CompatTermError(
                    ctx=ctx,
                    backend=getattr(getattr(env, "scenario", None), "simulator", None),
                    message=f"failed to instantiate {term_cls}: {exc}",
                ) from exc
            warn_once.warning(
                f"{ctx.label()}/init_failed",
                "{}: failed to instantiate {} ({}). Skipping.",
                ctx.label(),
                term_cls,
                exc,
            )
    return terms


def _call_term_cfg_resets(
    *,
    terms: dict[str, Any],
    env: Any,
    strict: bool,
    warn_once: WarnOnce,
    kind: str,
    env_ids: torch.Tensor | list[int] | slice | None = None,
) -> None:
    env_ids_ = _normalize_env_ids(env_ids)
    for name, term_cfg in terms.items():
        reset_fn = getattr(getattr(term_cfg, "func", None), "reset", None)
        if callable(reset_fn):
            ctx = TermContext(kind=kind, group="reset", name=name)
            safe_call(
                reset_fn,
                host_env=env,
                ctx=ctx,
                strict=strict,
                warn_once=warn_once,
                env_ids=env_ids_,
            )


class CompatCommandManager:
    """Command manager compatible with IsaacLab term configs/classes."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._terms = _instantiate_class_terms(
            cfg=cfg, env=env, strict=strict, warn_once=self._warn_once, kind="command"
        )

    def get_term(self, name: str):
        return self._terms[name]

    def get_command(self, name: str) -> torch.Tensor:
        term = self.get_term(name)
        cmd = getattr(term, "command", None)
        if cmd is None:
            raise CompatTermError(
                ctx=TermContext(kind="command", name=name),
                backend=getattr(getattr(self._env, "scenario", None), "simulator", None),
                message="term has no `.command` property.",
            )
        return cmd

    def reset(self, env_ids: torch.Tensor | list[int]):
        metrics: dict[str, torch.Tensor] = {}
        for name, term in self._terms.items():
            reset_fn = getattr(term, "reset", None)
            if callable(reset_fn):
                ctx = TermContext(kind="command", group="reset", name=name)
                out = safe_call(
                    reset_fn,
                    host_env=self._env,
                    ctx=ctx,
                    strict=self._strict,
                    warn_once=self._warn_once,
                    env_ids=env_ids,
                )
                if isinstance(out, dict):
                    for k, v in out.items():
                        metrics[f"{name}/{k}"] = v
        return metrics

    def compute(self):
        for name, term in self._terms.items():
            compute_fn = getattr(term, "compute", None)
            if callable(compute_fn):
                ctx = TermContext(kind="command", group="compute", name=name)
                safe_call(
                    compute_fn,
                    host_env=self._env,
                    ctx=ctx,
                    strict=self._strict,
                    warn_once=self._warn_once,
                    dt=self._env.step_dt,
                )


class CompatActionManager:
    """Action manager: processes high-level actions into handler target payloads."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._terms: list[tuple[str, Any]] = []
        self._term_instances: dict[str, Any] = {}

        self.action: torch.Tensor | None = None
        self.prev_action: torch.Tensor | None = None

        for name, term_cfg in iter_public_fields(cfg):
            if term_cfg is None:
                continue
            self._terms.append((name, term_cfg))

        registry = ActionTermRegistry(env=env, strict=strict, warn_once=self._warn_once)
        for name, term_cfg in self._terms:
            term = registry.build(name=name, cfg=term_cfg)
            self._term_instances[name] = term

        self.total_action_dim = int(sum(int(getattr(t, "action_dim", 0)) for t in self._term_instances.values()))

    @property
    def terms(self) -> dict[str, Any]:
        """Public term collection in cfg order (IsaacLab-style)."""
        return self._term_instances

    def get_term(self, name: str):
        return self._term_instances[name]

    def _normalize_action_batch(self, actions: torch.Tensor) -> torch.Tensor:
        num_envs = int(self._env.num_envs)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if actions.shape[0] != num_envs:
            if not self._strict and actions.shape[0] == 1 and num_envs > 1:
                self._warn_once.warning(
                    "action/broadcast_batch",
                    "action: got batch=1 but env.num_envs={}; broadcasting actions across envs.",
                    num_envs,
                )
                actions = actions.repeat(num_envs, 1)
            else:
                raise CompatTermError(
                    ctx=TermContext(kind="action", name="__all__"),
                    backend=getattr(getattr(self._env, "scenario", None), "simulator", None),
                    message=f"expected actions batch size {num_envs}, got {actions.shape[0]}",
                )

        if actions.shape[1] != self.total_action_dim:
            if self._strict:
                raise CompatTermError(
                    ctx=TermContext(kind="action", name="__all__"),
                    backend=getattr(getattr(self._env, "scenario", None), "simulator", None),
                    message=f"action dim mismatch: got {actions.shape[1]} expected {self.total_action_dim}",
                )
            self._warn_once.warning(
                "action/shape_mismatch",
                "action: dim mismatch got {} expected {}. Truncating/padding with zeros.",
                actions.shape[1],
                self.total_action_dim,
            )
            if actions.shape[1] < self.total_action_dim:
                pad = torch.zeros(
                    (actions.shape[0], self.total_action_dim - actions.shape[1]),
                    device=actions.device,
                    dtype=actions.dtype,
                )
                actions = torch.cat([actions, pad], dim=1)
            else:
                actions = actions[:, : self.total_action_dim]

        return actions

    def process(self, actions: torch.Tensor) -> list[dict[str, Any]]:
        actions = torch.as_tensor(actions, device=self._env.device)
        actions = self._normalize_action_batch(actions)

        if self.action is None:
            self.action = torch.zeros_like(actions)
            self.prev_action = torch.zeros_like(actions)
        else:
            self.prev_action = self.action
            self.action = actions

        if not self._term_instances or self.total_action_dim == 0:
            return [{} for _ in range(int(self._env.num_envs))]

        targets_by_asset: dict[str, dict[str, torch.Tensor]] = {}
        cursor = 0
        for _name, term in self._term_instances.items():
            dim = int(getattr(term, "action_dim", 0))
            a_slice = actions[:, cursor : cursor + dim]
            cursor += dim
            term.apply(actions=a_slice, targets_by_asset=targets_by_asset)

        # Many MetaSim handlers assume `dof_pos_target` exists. If no position targets were produced,
        # fall back to holding the current joint positions (best-effort no-op).
        for robot in getattr(self._env.scenario, "robots", []):
            robot_name = getattr(robot, "name", None)
            if not robot_name:
                continue
            if robot_name not in targets_by_asset:
                targets_by_asset[robot_name] = {}
            if "dof_pos_target" in targets_by_asset[robot_name]:
                continue
            try:
                art = self._env.scene[robot_name]
                targets_by_asset[robot_name]["dof_pos_target"] = art.data.joint_pos.clone().to(dtype=torch.float32)
            except Exception:
                try:
                    art = self._env.scene[robot_name]
                    num_joints = len(getattr(art, "joint_names", []))
                except Exception:
                    num_joints = 0
                targets_by_asset[robot_name]["dof_pos_target"] = torch.zeros(
                    (int(self._env.num_envs), int(num_joints)), device=self._env.device, dtype=torch.float32
                )

        # Convert tensor targets into MetaSim dict-list `Action` payloads.
        out: list[dict[str, Any]] = [{} for _ in range(int(self._env.num_envs))]
        for asset_name, channel_map in targets_by_asset.items():
            try:
                asset = self._env.scene[asset_name]
                joint_names = list(getattr(asset, "joint_names", []))
            except Exception:
                joint_names = []

            for env_id in range(int(self._env.num_envs)):
                env_payload = out[env_id].setdefault(asset_name, {})
                for channel, tensor in channel_map.items():
                    row = tensor[env_id].detach().to(device="cpu", dtype=torch.float32).tolist()
                    env_payload[channel] = {jn: float(val) for jn, val in zip(joint_names, row)}

        return out


class CompatObservationManager:
    """Observation manager: executes cfg term functions and concatenates per group."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._terms_by_group: dict[str, dict[str, Any]] = {}

        for group_name, group_cfg in iter_public_fields(self._cfg):
            if group_cfg is None:
                continue
            terms: dict[str, Any] = {}
            for term_name, term_cfg in iter_public_fields(group_cfg):
                if not is_term_cfg(term_cfg):
                    continue
                terms[term_name] = term_cfg
            self._terms_by_group[group_name] = terms

    @property
    def terms(self) -> dict[str, dict[str, Any]]:
        """Public term registry grouped by observation group (cfg order)."""
        return self._terms_by_group

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Names of active observation terms in each group (cfg order)."""
        return {group: list(terms.keys()) for group, terms in self._terms_by_group.items()}

    def get_term(self, name: str, *, group: str | None = None) -> Any:
        """Return the term cfg for a given observation term.

        Args:
            name: Term name.
            group: Optional observation group name. Required when the term name is not unique across groups.
        """
        if group is not None:
            return self._terms_by_group[group][name]

        matches: list[Any] = []
        for terms in self._terms_by_group.values():
            if name in terms:
                matches.append(terms[name])

        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise KeyError(name)
        raise ValueError(f"observation term name '{name}' is ambiguous across groups; pass group=...")

    def _apply_noise(self, obs: torch.Tensor, noise_cfg: Any) -> torch.Tensor:
        n_min = getattr(noise_cfg, "n_min", None)
        n_max = getattr(noise_cfg, "n_max", None)
        if n_min is None or n_max is None:
            return obs
        return obs + (torch.rand_like(obs) * (float(n_max) - float(n_min)) + float(n_min))

    def compute(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        for group_name, group_cfg in iter_public_fields(self._cfg):
            if group_cfg is None:
                continue

            enable_corruption = bool(getattr(group_cfg, "enable_corruption", False))
            concatenate_terms = bool(getattr(group_cfg, "concatenate_terms", True))

            results: list[torch.Tensor] = []
            for term_name, term_cfg in iter_public_fields(group_cfg):
                if not is_term_cfg(term_cfg):
                    continue

                ctx = TermContext(kind="obs", group=group_name, name=term_name)
                func = term_cfg.func
                params = getattr(term_cfg, "params", None) or {}
                value = safe_call(
                    func,
                    host_env=self._env,
                    ctx=ctx,
                    strict=self._strict,
                    warn_once=self._warn_once,
                    env=self._env,
                    **params,
                )
                t = to_tensor(value, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
                if t is None:
                    continue
                t = ensure_float32(t, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
                if t is None:
                    continue
                t = normalize_batch_2d(t, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
                if t is None:
                    continue
                if enable_corruption and getattr(term_cfg, "noise", None) is not None:
                    t = self._apply_noise(t, term_cfg.noise)
                results.append(t)

            if concatenate_terms:
                if results:
                    try:
                        out[group_name] = torch.cat(results, dim=-1)
                    except Exception as exc:
                        if self._strict:
                            raise CompatTermError(
                                ctx=TermContext(kind="obs", group=group_name, name="__concat__"),
                                backend=getattr(getattr(self._env, "scenario", None), "simulator", None),
                                message=f"failed to concatenate observation terms: {exc}",
                            ) from exc
                        self._warn_once.warning(
                            f"obs[{group_name}]/concat_failed",
                            "obs[{}]: failed to concatenate terms ({}). Returning empty obs group.",
                            group_name,
                            exc,
                        )
                        out[group_name] = torch.zeros((self._env.num_envs, 0), device=self._env.device)
                else:
                    out[group_name] = torch.zeros((self._env.num_envs, 0), device=self._env.device)
            else:
                out[group_name] = torch.stack(results, dim=0) if results else torch.zeros((0,), device=self._env.device)

        return out


class CompatRewardManager:
    """Reward manager: evaluates cfg reward terms."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._terms: dict[str, Any] = {}
        self._episode_sums: dict[str, torch.Tensor] = {}

        for name, term_cfg in iter_public_fields(self._cfg):
            if not is_term_cfg(term_cfg):
                continue
            self._terms[name] = term_cfg
            self._episode_sums[name] = torch.zeros(int(env.num_envs), dtype=torch.float32, device=env.device)

    @property
    def terms(self) -> dict[str, Any]:
        """Public term registry in cfg order."""
        return self._terms

    @property
    def active_terms(self) -> list[str]:
        """Names of active reward terms (cfg order)."""
        return list(self._terms.keys())

    def get_term(self, name: str) -> Any:
        """Return the term cfg for a given reward term."""
        return self._terms[name]

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> dict[str, torch.Tensor]:
        """Return per-term episodic reward stats and clear sums for selected envs (IsaacLab-style)."""
        env_ids = _normalize_env_ids(env_ids)

        extras: dict[str, torch.Tensor] = {}
        denom = float(getattr(self._env, "max_episode_length_s", 1.0) or 1.0)
        denom = max(denom, 1e-8)
        for name, buf in self._episode_sums.items():
            extras[f"Episode_Reward/{name}"] = torch.mean(buf[env_ids]) / denom
            buf[env_ids] = 0.0

        # Reset class-based terms if they expose a reset method.
        _call_term_cfg_resets(
            terms=self._terms,
            env=self._env,
            strict=self._strict,
            warn_once=self._warn_once,
            kind="reward",
            env_ids=env_ids,
        )

        return extras

    def compute(self) -> torch.Tensor:
        rew = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self._env.device)
        for name, term_cfg in self._terms.items():
            ctx = TermContext(kind="reward", name=name)
            func = term_cfg.func
            params = getattr(term_cfg, "params", None) or {}
            weight = float(getattr(term_cfg, "weight", 1.0))
            value = safe_call(
                func,
                host_env=self._env,
                ctx=ctx,
                strict=self._strict,
                warn_once=self._warn_once,
                env=self._env,
                **params,
            )
            t = to_tensor(value, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
            if t is None:
                continue
            t = ensure_float32(t, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
            if t is None:
                continue
            t = normalize_batch_1d(t, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
            if t is None:
                continue
            value = t.to(rew.device) * weight * float(getattr(self._env, "step_dt", 1.0))
            rew = rew + value
            # Update episodic sum (IsaacLab semantics).
            if name not in self._episode_sums or self._episode_sums[name].shape != (int(self._env.num_envs),):
                self._episode_sums[name] = torch.zeros(
                    int(self._env.num_envs), dtype=torch.float32, device=self._env.device
                )
            self._episode_sums[name].add_(value.detach())
        return rew


class CompatTerminationManager:
    """Termination manager: evaluates cfg termination terms."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()

        self._terms: dict[str, Any] = {}
        for name, term_cfg in iter_public_fields(self._cfg):
            if not is_term_cfg(term_cfg):
                continue
            self._terms[name] = term_cfg

        self._term_names = list(self._terms.keys())
        self._term_name_to_idx = {name: i for i, name in enumerate(self._term_names)}

        self._term_dones = torch.zeros((env.num_envs, len(self._term_names)), device=env.device, dtype=torch.bool)
        self._last_episode_dones = torch.zeros_like(self._term_dones)

        # IsaacLab-compatible split buffers
        self.terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.time_outs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        return self.terminated | self.time_outs

    def get_term(self, name: str) -> torch.Tensor:
        """Return per-env value for a termination term at the current step."""
        return self._term_dones[:, self._term_name_to_idx[name]]

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> dict[str, torch.Tensor]:
        """Return episodic termination stats (IsaacLab-style).

        Notes:
            IsaacLab logs last-episode termination term stats globally (env_ids are ignored for stats).
        """
        _ = env_ids
        extras: dict[str, torch.Tensor] = {}
        if self._term_names:
            stats = self._last_episode_dones.float().mean(dim=0)
            for i, name in enumerate(self._term_names):
                extras[f"Episode_Termination/{name}"] = stats[i]

        # Reset class-based terms if they expose a reset method.
        _call_term_cfg_resets(
            terms=self._terms,
            env=self._env,
            strict=self._strict,
            warn_once=self._warn_once,
            kind="done",
            env_ids=env_ids,
        )

        return extras

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.terminated[:] = False
        self.time_outs[:] = False
        if self._term_dones.shape != (self._env.num_envs, len(self._term_names)):
            self._term_dones = torch.zeros(
                (self._env.num_envs, len(self._term_names)), device=self._env.device, dtype=torch.bool
            )
            self._last_episode_dones = torch.zeros_like(self._term_dones)

        for i, (name, term_cfg) in enumerate(self._terms.items()):
            ctx = TermContext(kind="done", name=name)
            func = term_cfg.func
            params = getattr(term_cfg, "params", None) or {}
            is_timeout = bool(getattr(term_cfg, "time_out", False))
            value = safe_call(
                func,
                host_env=self._env,
                ctx=ctx,
                strict=self._strict,
                warn_once=self._warn_once,
                env=self._env,
                **params,
            )
            t = to_tensor(value, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
            if t is None:
                continue
            t = normalize_bool(t, env=self._env, ctx=ctx, strict=self._strict, warn_once=self._warn_once)
            if t is None:
                continue
            self._term_dones[:, i] = t
            if is_timeout:
                self.time_outs |= t
            else:
                self.terminated |= t

        # Update last-episode dones for envs where any term fired this step.
        rows = self._term_dones.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            self._last_episode_dones[rows] = self._term_dones[rows]

        return self.terminated, self.time_outs


class _AliasScene:
    """Scene facade that maps IsaacLab task asset names to handler scene names."""

    def __init__(self, *, scene: Any, resolve_name: Any) -> None:
        self._scene = scene
        self._resolve_name = resolve_name

    @property
    def num_envs(self) -> int:
        return int(getattr(self._scene, "num_envs", 1))

    def __getitem__(self, key: str) -> Any:
        resolved = self._resolve_name(key) if callable(self._resolve_name) else key
        return self._scene[resolved]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        # Delegate everything else (env_origins, sensors, etc.)
        return getattr(self._scene, name)


class _IsaacLabEventEnvProxy:
    """Env proxy for executing IsaacLab-native event terms on the handler's IsaacLab scene."""

    def __init__(self, *, env: Any, scene: Any) -> None:
        self._env = env
        self.scene = scene
        self.sim = getattr(getattr(env, "handler", None), "sim", None)
        self.device = getattr(env, "device", None)
        self.num_envs = int(getattr(env, "num_envs", 1))

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        return getattr(self._env, name)


class CompatEventManager:
    """Event manager: startup/interval events with explicit backend gating."""

    def __init__(self, cfg: Any | None, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._startup_applied = False

        # Each term stores: (name, func, params, call_env)
        self._startup_terms: list[tuple[str, Any, dict[str, Any], Any]] = []
        # Each term stores: (name, func, interval_range, params, call_env)
        self._interval_terms: list[tuple[str, Any, tuple[float, float], dict[str, Any], Any]] = []
        self._interval_timers: dict[str, torch.Tensor] = {}

        if self._cfg is None:
            return

        # Optional IsaacLab-backed execution for event terms on the IsaacSim handler.
        self._isaaclab_env: Any | None = None
        backend = getattr(getattr(env, "scenario", None), "simulator", None)
        handler_scene = getattr(getattr(env, "handler", None), "scene", None)
        if backend == "isaacsim" and handler_scene is not None:
            resolve_name = getattr(getattr(env, "scene", None), "resolve_asset_name", lambda x: x)
            alias_scene = _AliasScene(scene=handler_scene, resolve_name=resolve_name)
            self._isaaclab_env = _IsaacLabEventEnvProxy(env=env, scene=alias_scene)

        def _callable_path(func: Any) -> str:
            mod = getattr(func, "__module__", "") or ""
            name = getattr(func, "__name__", type(func).__name__)
            return f"{mod}.{name}"

        def _should_use_isaaclab_env(func: Any) -> bool:
            if self._isaaclab_env is None:
                return False
            # Class-based terms generally require IsaacLab asset types / PhysX views.
            if inspect.isclass(func):
                return True
            mod = getattr(func, "__module__", "") or ""
            # IsaacLab MDP events are native and expect IsaacLab assets.
            if mod.startswith("isaaclab.envs.mdp.events"):
                return True
            # Terms explicitly gated on PhysX views should run on the handler scene when available.
            spec = EventTermRegistry._SUPPORT_MAP.get(_callable_path(func))
            return spec is not None and spec.capability is not None

        registry = EventTermRegistry(env=env, strict=strict, warn_once=self._warn_once)
        for name, term_cfg in iter_public_fields(self._cfg):
            mode = getattr(term_cfg, "mode", None)
            wrapped = registry.wrap(name=name, term_cfg=term_cfg)
            if wrapped is None:
                continue

            call_env = self._isaaclab_env if _should_use_isaaclab_env(wrapped) else self._env
            term_cfg_for_call = term_cfg

            # When using the handler's IsaacLab scene, re-resolve SceneEntityCfg indices using that scene
            # (avoids body/joint index mismatches due to MetaSim's sorted-name convention).
            if call_env is self._isaaclab_env:
                try:
                    term_cfg_for_call = copy.deepcopy(term_cfg)
                    # Preserve any registry wrapping (unsupported no-op, etc.).
                    term_cfg_for_call.func = wrapped
                    resolve_scene_entity_cfgs(term_cfg_for_call, scene=call_env.scene)
                except Exception:
                    term_cfg_for_call = term_cfg

            func = getattr(term_cfg_for_call, "func", wrapped)
            params = getattr(term_cfg_for_call, "params", None) or {}

            # IsaacLab event terms can be callable classes (ManagerTermBase-derived). Instantiate once.
            if inspect.isclass(func):
                try:
                    func = func(term_cfg_for_call, call_env)
                except Exception as exc:
                    ctx = TermContext(kind="event", name=name)
                    if self._strict:
                        raise CompatTermError(
                            ctx=ctx,
                            backend=getattr(getattr(env, "scenario", None), "simulator", None),
                            message=f"failed to instantiate {getattr(func, '__name__', func)}: {exc}",
                        ) from exc
                    self._warn_once.warning(
                        f"{ctx.label()}/init_failed",
                        "{}: failed to instantiate {} ({}). Skipping.",
                        ctx.label(),
                        getattr(func, "__name__", func),
                        exc,
                    )
                    continue

            if mode == "startup":
                self._startup_terms.append((name, func, params, call_env))
            elif mode == "interval":
                interval_range = getattr(term_cfg, "interval_range_s", None)
                if interval_range is None:
                    continue
                low, high = float(interval_range[0]), float(interval_range[1])
                self._interval_terms.append((name, func, (low, high), params, call_env))

    def _merge_metrics(self, *, prefix: str, metrics: dict[str, Any]) -> None:
        if not metrics:
            return
        dest = self._env.extras.setdefault("metrics", {})
        if not isinstance(dest, dict):
            return
        for k, v in metrics.items():
            dest[f"{prefix}/{k}"] = v

    def _invoke_event_term(
        self,
        *,
        name: str,
        func: Any,
        params: dict[str, Any],
        call_env: Any,
        group: str,
        env_ids: torch.Tensor | list[int] | None,
    ) -> None:
        ctx = TermContext(kind="event", group=group, name=name)
        out = safe_call(
            func,
            host_env=self._env,
            ctx=ctx,
            strict=self._strict,
            warn_once=self._warn_once,
            env=call_env,
            env_ids=env_ids,
            **params,
        )
        if isinstance(out, dict):
            self._merge_metrics(prefix=f"event/{name}", metrics=out)

    def apply_startup(self):
        if self._cfg is None or self._startup_applied:
            return
        for name, func, params, call_env in self._startup_terms:
            self._invoke_event_term(
                name=name,
                func=func,
                params=params,
                call_env=call_env,
                group="startup",
                env_ids=None,
            )
        self._startup_applied = True

    def step(self):
        if self._cfg is None:
            return

        for name, func, (low, high), params, call_env in self._interval_terms:
            if name not in self._interval_timers:
                self._interval_timers[name] = torch.empty(self._env.num_envs, device=self._env.device).uniform_(
                    low, high
                )

            self._interval_timers[name] -= float(self._env.step_dt)
            env_ids = torch.where(self._interval_timers[name] <= 0.0)[0]
            if env_ids.numel() == 0:
                continue

            self._interval_timers[name][env_ids] = torch.empty_like(self._interval_timers[name][env_ids]).uniform_(
                low, high
            )

            self._invoke_event_term(
                name=name,
                func=func,
                params=params,
                call_env=call_env,
                group="interval",
                env_ids=env_ids,
            )


class CompatCurriculumManager:
    """Curriculum manager: best-effort IsaacLab-like API surface."""

    def __init__(self, cfg: Any, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()

        self._terms: dict[str, Any] = {}
        for name, term_cfg in iter_public_fields(self._cfg):
            if not is_term_cfg(term_cfg):
                continue
            self._terms[name] = term_cfg

        # Stores per-term curriculum state returned by the term.
        self._curriculum_state: dict[str, Any] = {name: None for name in self._terms}

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def compute(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        env_ids = _normalize_env_ids(env_ids)
        for name, term_cfg in self._terms.items():
            ctx = TermContext(kind="curriculum", name=name)
            func = term_cfg.func
            params = getattr(term_cfg, "params", None) or {}
            state = safe_call(
                func,
                host_env=self._env,
                ctx=ctx,
                strict=self._strict,
                warn_once=self._warn_once,
                env=self._env,
                env_ids=env_ids,
                **params,
            )
            self._curriculum_state[name] = state

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> dict[str, Any]:
        """Return curriculum state for logging and reset class-based terms (best-effort)."""
        extras: dict[str, Any] = {}
        for name, state in self._curriculum_state.items():
            if state is None:
                continue
            if isinstance(state, dict):
                for k, v in state.items():
                    extras[f"Curriculum/{name}/{k}"] = v
            else:
                extras[f"Curriculum/{name}"] = state

        _call_term_cfg_resets(
            terms=self._terms,
            env=self._env,
            strict=self._strict,
            warn_once=self._warn_once,
            kind="curriculum",
            env_ids=env_ids,
        )
        return extras


class CompatRecorderManager:
    """Recorder manager: best-effort IsaacLab-like hook surface (no-op by default)."""

    def __init__(self, cfg: Any | None, *, env: Any, strict: bool = False, warn_once: WarnOnce | None = None) -> None:
        self._cfg = cfg
        self._env = env
        self._strict = strict
        self._warn_once = warn_once or WarnOnce()
        self._terms: dict[str, Any] = {}

        if not self._cfg:
            return

        self._terms = _instantiate_class_terms(
            cfg=self._cfg,
            env=env,
            strict=self._strict,
            warn_once=self._warn_once,
            kind="recorder",
        )

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> dict[str, Any]:
        _ = env_ids
        return {}

    def _call_hook(self, hook_name: str, *args: Any) -> None:
        for term in self._terms.values():
            fn = getattr(term, hook_name, None)
            if callable(fn):
                fn(*args)

    def record_pre_reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        self._call_hook("record_pre_reset", env_ids)

    def record_post_reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        self._call_hook("record_post_reset", env_ids)

    def record_pre_step(self) -> None:
        self._call_hook("record_pre_step")

    def record_post_step(self) -> None:
        self._call_hook("record_post_step")

    def record_post_physics_decimation_step(self) -> None:
        self._call_hook("record_post_physics_decimation_step")
