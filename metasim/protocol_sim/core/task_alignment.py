from __future__ import annotations

import copy
import importlib
import re
from dataclasses import dataclass
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class


@dataclass(frozen=True)
class TaskAlignmentSpec:
    """Minimal task metadata needed to align a protocol server with a task env."""

    task_cls: type
    scenario: ScenarioCfg
    env_cfg: Any | None


def load_task_alignment_spec(task_ref: str) -> TaskAlignmentSpec:
    """Load a task class + deep-copied scenario/env_cfg for alignment.

    Args:
        task_ref: Either a task registry name (e.g. "unitree_rl.walk_g1_dof29") or
            an import path in the form "some.module:TaskClass".
    """
    task_ref = task_ref.strip()
    if not task_ref:
        raise ValueError("task_ref must be a non-empty string.")

    if ":" in task_ref:
        mod_name, cls_name = task_ref.split(":", 1)
        mod = importlib.import_module(mod_name)
        task_cls = getattr(mod, cls_name)
    else:
        task_cls = get_task_class(task_ref)

    scenario = copy.deepcopy(task_cls.scenario)
    scenario.__post_init__()

    env_cfg = None
    env_cfg_cls = getattr(task_cls, "env_cfg_cls", None)
    if env_cfg_cls is not None:
        env_cfg = env_cfg_cls()

    return TaskAlignmentSpec(task_cls=task_cls, scenario=scenario, env_cfg=env_cfg)


def _overlay_regex_map(base: dict[str, float], overrides: dict[str, float], names: list[str]) -> dict[str, float]:
    """Overlay regex -> value pairs onto a name->value map (fullmatch)."""
    out = {n: float(base.get(n, 0.0)) for n in names}
    for key, val in overrides.items():
        pat = re.compile(key)
        for name in names:
            if pat.fullmatch(name):
                out[name] = float(val)
    return out


def apply_task_initial_state(
    *,
    handler,
    robot_name: str,
    env_cfg: Any,
    pos_fallback: tuple[float, float, float] | None = None,
) -> None:
    """Best-effort: set root/joint state to the task's initial_states for one robot.

    This mirrors what RoboVerse task envs do during reset, but is intentionally minimal:
    - single-env only
    - sets root pos (and resets root vel/ang_vel)
    - sets joint positions (and zeros joint velocities)
    """
    init = getattr(getattr(env_cfg, "initial_states", None), "robots", {}).get(robot_name)
    if not init:
        return

    sorted_joint_names = handler.get_joint_names(robot_name, sort=True)

    # Resolve the RobotCfg for defaults.
    robot_cfg = None
    for r in getattr(handler, "robots", []):
        if getattr(r, "name", None) == robot_name:
            robot_cfg = r
            break
    if robot_cfg is None:
        raise ValueError(f"Robot '{robot_name}' not found in handler.robots.")

    overrides = init.get("default_joint_pos", {}) or {}
    joint_pos_by_name = _overlay_regex_map(
        getattr(robot_cfg, "default_joint_positions", {}) or {},
        overrides,
        sorted_joint_names,
    )
    joint_pos = [joint_pos_by_name[jn] for jn in sorted_joint_names]

    pos = init.get("pos", None)
    if pos is None:
        if pos_fallback is not None:
            pos = list(pos_fallback)
    if pos is None:
        return

    ts = handler.get_states()
    rs = ts.robots[robot_name]
    device = rs.root_state.device
    dtype = rs.root_state.dtype

    # Root state: pos + identity quat, zero velocities.
    rs.root_state[0, 0:3] = torch.as_tensor(pos, device=device, dtype=dtype)
    rs.root_state[0, 3:7] = torch.as_tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    rs.root_state[0, 7:13] = 0.0

    # Joint state.
    rs.joint_pos[0] = torch.as_tensor(joint_pos, device=rs.joint_pos.device, dtype=rs.joint_pos.dtype)
    rs.joint_vel[0] = 0.0

    handler.set_states(ts)
