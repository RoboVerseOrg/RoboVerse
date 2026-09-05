"""Trajectory recording shared by the FastTD3 evaluators.

One conversion per step: ``recorded_step`` moves the handler's tensor state to the host once and yields,
per env, the v2 state entry (``metasim.utils.demo_util.demo_util_v2.state_nested_to_v2`` over
``state_tensor_to_nested``: ``pos`` / ``rot`` / ``dof_pos`` per entity, joints in the handler's order,
plain Python) and the joint targets the handler reports for each robot. An env the RL env auto-reset on
this step is taken from ``info["terminal_states"]``, the state its episode ended in, not the reset pose
the handler now shows; an ``info`` without that contract is refused, because the two cannot be told apart
afterwards. ``commanded_action`` is the ``actions[t]`` entry: the reported ``dof_pos_target``, which every
position-controlled backend derives from the targets it was given. The evaluators used to write the
*achieved* ``joint_pos`` under ``dof_pos_target``, so a replay drove the robot to where it had already been.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from metasim.types import TensorState
from metasim.utils.demo_util.demo_util_v2 import state_nested_to_v2
from metasim.utils.state import select_envs, state_tensor_to_nested, state_to_device


@dataclass
class StepRecord:
    """One env's record of one step."""

    state: dict
    """the v2 state entry (``{entity: {"pos", "rot", "dof_pos"?}}``, plain Python)"""
    targets: dict[str, dict[str, float] | None]
    """per robot, the joint targets the handler reports for this step, or None when it reports none"""


def recorded_step(handler, states, info: dict | None = None, env_ids: list[int] | None = None) -> list[StepRecord]:
    """Records of this step for ``env_ids`` (every env when None), in that order, with the terminal state
    for envs that were auto-reset.

    Only the requested envs are converted (host copies and per-joint dicts are the cost).

    Raises:
        ValueError: ``info`` comes from a step that did not record its auto-reset
            (``RLTaskEnv._note_auto_reset``), so reset poses could be mistaken for terminal states.
    """
    states = _physics_only(states)
    ids = list(range(_num_envs(states))) if env_ids is None else [int(i) for i in env_ids]
    if info is None:
        return _records(handler, select_envs(states, ids) if env_ids is not None else states)
    if "auto_reset_env_ids" not in info:
        raise ValueError(
            "the env's step info carries no 'auto_reset_env_ids' / 'terminal_states': a task that runs its own "
            "reset loop must call RLTaskEnv._note_auto_reset before resetting, or the recorder cannot tell an "
            "env's terminal state from its reset pose"
        )
    reset_ids = [int(i) for i in info["auto_reset_env_ids"]]
    if reset_ids and info.get("terminal_states") is None:
        raise ValueError(
            "envs were auto-reset this step but info['terminal_states'] is None: set "
            "env.record_terminal_states = True before stepping so the pre-reset state is kept"
        )
    # a reset env is converted from the terminal copy only; the others from the current state
    terminal_row = {env_id: row for row, env_id in enumerate(reset_ids)}
    current_ids = [i for i in ids if i not in terminal_row]
    current = (
        dict(zip(current_ids, _records(handler, select_envs(states, current_ids)), strict=True)) if current_ids else {}
    )
    wanted_rows = [terminal_row[i] for i in ids if i in terminal_row]
    terminal = (
        dict(
            zip(
                wanted_rows,
                _records(handler, select_envs(_physics_only(info["terminal_states"]), wanted_rows)),
                strict=True,
            )
        )
        if wanted_rows
        else {}
    )
    return [terminal[terminal_row[i]] if i in terminal_row else current[i] for i in ids]


def _physics_only(states: TensorState) -> TensorState:
    """Objects and robots only: cameras and extras are not part of a record and are not copied."""
    return TensorState(objects=states.objects, robots=states.robots, cameras={}, extras={})


def restart_done_envs(env, done_ids: list[int], info: dict, next_obs, *, init_states: bool) -> dict[int, dict]:
    """After the done envs of a step were recorded: every one of them starts its next episode.

    ``RLTaskEnv.step`` auto-reset the envs it found done; a task that marks an env done after that base step
    (the pick-place release check) leaves it un-reset, so it is reset here and its row of ``next_obs`` is
    replaced. Returns each done env's v2 ``init_state`` when ``init_states`` is set (an empty dict otherwise).
    """
    stragglers = [i for i in done_ids if i not in set(info.get("auto_reset_env_ids", ()))]
    if stragglers:
        reset_obs, _ = env.reset(env_ids=stragglers)
        next_obs[stragglers] = reset_obs[stragglers]
    if not init_states or not done_ids:
        return {}
    return dict(zip(done_ids, initial_states(env.handler, done_ids), strict=True))


def initial_states(handler, env_ids: list[int] | None = None) -> list[dict]:
    """The v2 ``init_state`` of every env (or of ``env_ids``) from the handler's current (just reset) state."""
    return [record.state for record in recorded_step(handler, handler.get_states(mode="tensor"), None, env_ids)]


def _num_envs(states) -> int:
    entity = next(iter(list(states.robots.values()) + list(states.objects.values())))
    return entity.root_state.shape[0]


def commanded_action(record: StepRecord, *, robot_name: str) -> dict:
    """The v2 ``actions[t]`` entry: the joint target the handler applied for ``robot_name`` this step.

    Raises:
        ValueError: the handler reports no ``joint_pos_target`` for the robot (no action applied yet, an
            effort-driven robot, or a backend that cannot derive it);
            achieved positions are never written in its place.
    """
    if robot_name not in record.targets:
        raise ValueError(f"{robot_name!r} is not in the recorded state {sorted(record.targets)}")
    target = record.targets[robot_name]
    if target is None:
        raise ValueError(
            f"{robot_name!r}: the handler reports no joint_pos_target (no action applied yet, an effort-driven "
            "robot, or a backend that does not derive it from the applied action), so the commanded action cannot be recorded and achieved positions are not written in its place"
        )
    return {"dof_pos_target": dict(target)}


def _records(handler, states) -> list[StepRecord]:
    nested = state_tensor_to_nested(handler, state_to_device(states, "cpu"))
    out = []
    for env_state in nested:
        targets = {
            robot: None if robot_state.get("dof_pos_target") is None else dict(robot_state["dof_pos_target"])
            for robot, robot_state in env_state.get("robots", {}).items()
        }
        out.append(StepRecord(state=_plain(state_nested_to_v2(env_state)), targets=targets))
    return out


def _plain(value):
    """Tensors and arrays as lists, containers recursed, scalars untouched: what a v2 file stores.

    ``roboverse_learn.il.utils.tensor_util.to_list`` is not this: it assumes every leaf is a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()  # a numpy scalar becomes a Python one
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
