"""Bring a freshly reset simulation to a settled, validated initial state.

Shared by the imitation-learning runner, the ACT / VLA evaluators and the demo collector so the
settle rule is defined once. Import-safe: only torch and loguru.

``ensure_clean_state(handler, expected_state)`` steps the physics until two consecutive steps leave
every *object's* joint positions (to 1e-5) and position (to 1e-4) unchanged, after at least
``min_steps`` and at most ``max_steps`` steps. Robots are deliberately not part of the settle rule:
an arm with no target held sags under gravity on every step and would never count as settled; the
policy takes over the robot, the objects are what a reset must leave at rest. When ``expected_state``
is given (the nested dict a demo file stores, ``{"objects": {name: {"dof_pos": {joint: value},
...}}, "robots": {...}}``) the settled *object* joint positions of env ``env_id`` (articulated objects:
drawers, doors) are compared with it to 5e-3 rad and a mismatch is logged as a warning: the reset
did not take, and the episode would start from the wrong state. A scene that is still moving after
``max_steps`` is warned about too; the return value is True only when the objects settled and the
expected state (if any) matched. Callers that reset one env pass its ``env_id``.
"""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.utils.state import _dof_tensor_to_dict  # the demo writer's joint-name contract, reused verbatim

JOINT_ATOL = 1e-5
POSITION_ATOL = 1e-4
EXPECTED_ATOL = 5e-3


def settle_recipients(num_envs: int, *, env_id: int | None, finished=None, terminal=None, recording=()) -> list[int]:
    """The envs whose in-flight demo must receive the settle steps of a reset of ``env_id``.

    Settling one env steps every env of the batch, so every *other* env that keeps stepping after
    this iteration absorbs that physics and its demo must record it. Excluded: the env being reset,
    envs already ``finished`` (indexable by env), envs in ``terminal`` (their demo closes in this
    iteration; trailing frames would corrupt a closed demo), and envs not in ``recording``.
    """
    return [
        other
        for other in range(num_envs)
        if other != env_id
        and not (finished is not None and finished[other])
        and not (terminal is not None and other in terminal)
        and other in recording
    ]


def _entities(state) -> dict:
    """``name -> entity state`` for the objects of a ``TensorState`` (robots are not settled or validated)."""
    return dict(getattr(state, "objects", {}) or {})


def _is_settled(current, previous) -> bool:
    prev = _entities(previous)
    for name, entity in _entities(current).items():
        before = prev.get(name)
        if before is None:
            continue
        q, q0 = getattr(entity, "joint_pos", None), getattr(before, "joint_pos", None)
        if q is not None and q0 is not None and not torch.allclose(q, q0, atol=JOINT_ATOL):
            return False
        r, r0 = getattr(entity, "root_state", None), getattr(before, "root_state", None)
        if r is not None and r0 is not None and not torch.allclose(r[:, :3], r0[:, :3], atol=POSITION_ATOL):
            return False
    return True


def ensure_clean_state(
    handler,
    expected_state=None,
    *,
    env_id: int = 0,
    max_steps: int = 10,
    min_steps: int = 2,
    on_step=None,
) -> bool:
    """Step until the objects settle; True when they did and (if given) env ``env_id`` matches ``expected_state``.

    ``simulate()`` steps every env of a batched handler, so settling one env also advances the others.
    ``on_step(state)`` is called with the full ``TensorState`` after each step so a recorder can keep
    those steps in the other envs' episodes instead of silently losing them.
    """
    prev_state = None
    stable_count = 0
    current_state = None
    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states(mode="tensor")
        if on_step is not None:
            on_step(current_state)
        if step >= min_steps and prev_state is not None:
            if _is_settled(current_state, prev_state):
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                stable_count = 0
        prev_state = current_state
    settled = stable_count >= 2
    if not settled:
        log.warning(f"Scene did not settle within {max_steps} steps after reset: objects are still moving.")
    if expected_state is None or current_state is None:
        return settled
    mismatches = _expected_mismatches(handler, current_state, expected_state, env_id)
    if mismatches:
        log.warning(
            f"State validation failed after settling (env {env_id}): the reset may not have taken full effect. "
            + "; ".join(
                f"{name}.{joint}: got {got:.4f}, expected {exp:.4f}" for name, joint, got, exp in mismatches[:5]
            )
        )
        return False
    return settled


def _expected_mismatches(
    handler, current_state, expected_state: dict, env_id: int
) -> list[tuple[str, str, float, float]]:
    """Object joints of env ``env_id`` that differ from the demo dict's ``dof_pos`` by more than ``EXPECTED_ATOL``."""
    current_entities = _entities(current_state)
    mismatches = []
    for name, expected_entity in (expected_state.get("objects", {}) or {}).items():
        expected = expected_entity.get("dof_pos") if isinstance(expected_entity, dict) else None
        current = current_entities.get(name)
        if (
            not isinstance(expected, dict)
            or not expected
            or current is None
            or getattr(current, "joint_pos", None) is None
        ):
            continue
        current_dof = _dof_tensor_to_dict(current.joint_pos[env_id], handler.get_joint_names(name))
        for joint, exp in expected.items():
            got = current_dof.get(joint)
            if got is not None and abs(float(got) - float(exp)) > EXPECTED_ATOL:
                mismatches.append((name, joint, float(got), float(exp)))
    return mismatches
