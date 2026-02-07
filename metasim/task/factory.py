from __future__ import annotations

import inspect
from typing import Any

from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import get_task_class


def _filter_kwargs_for_callable(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs containing only parameters accepted by fn."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    accepted = {}
    for name, value in kwargs.items():
        param = params.get(name)
        if param is not None and param.kind != inspect.Parameter.VAR_POSITIONAL:
            accepted[name] = value
        elif accepts_var_keyword:
            # Preserve runtime kwargs for `**kwargs` constructors, but avoid injecting
            # the generic training `args` kwarg unless it is explicitly declared.
            if name != "args":
                accepted[name] = value
    return accepted


def make_task_env(
    task_name: str,
    *,
    scenario: ScenarioCfg,
    args: Any,
    device: Any,
    **kwargs: Any,
):
    """Instantiate a task environment (MetaSim-native or IsaacLab-style).

    - MetaSim-native tasks are typically subclasses of `BaseTaskEnv`.
    - IsaacLab-style tasks are often plain Gymnasium envs registered via `register_task`.

    This factory exists so training/evaluation scripts can have a single codepath and we
    can centrally manage Isaac Sim app lifecycle sharing.
    """
    task_cls = get_task_class(task_name)

    base_kwargs: dict[str, Any] = dict(kwargs)
    base_kwargs.update(
        scenario=scenario,
        args=args,
        device=device,
    )

    if isinstance(task_cls, type) and issubclass(task_cls, BaseTaskEnv):
        init_kwargs = _filter_kwargs_for_callable(task_cls.__init__, base_kwargs)
        return task_cls(**init_kwargs)

    # Treat non-BaseTaskEnv tasks as IsaacLab-style tasks by default.
    try:
        from metasim.integrations.isaaclab.wrapper import make_isaaclab_task_env

        make_kwargs = dict(kwargs)
        return make_isaaclab_task_env(task_cls, scenario=scenario, args=args, device=device, **make_kwargs)
    except Exception as exc:
        log.warning(
            "Failed to instantiate task '%s' via IsaacLab wrapper (%s). Falling back to direct construction.",
            task_name,
            exc,
        )
        init_kwargs = _filter_kwargs_for_callable(task_cls.__init__, base_kwargs)
        return task_cls(**init_kwargs)
