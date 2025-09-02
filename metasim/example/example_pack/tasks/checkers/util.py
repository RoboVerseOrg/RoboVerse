from __future__ import annotations

from typing import Any

import torch

from metasim.utils.state import state_tensor_to_nested


def _get_root_state(states: Any, obj_name: str) -> torch.Tensor:
    if hasattr(states, "objects") and obj_name in states.objects:
        return states.objects[obj_name].root_state
    if hasattr(states, "robots") and obj_name in states.robots:
        return states.robots[obj_name].root_state
    raise ValueError(f"Object {obj_name} not found in states")


def get_pos(handler: Any, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
    """Return positions for `obj_name` over `env_ids`.

    If `env_ids` is None, infer the number of environments from the first axis of
    the object's `root_state` and use all environments.
    """
    states = handler.get_states(mode="tensor")
    root_state = _get_root_state(states, obj_name)
    if env_ids is None:
        env_ids = list(range(root_state.shape[0]))
    return root_state[env_ids, :3]


def get_rot(handler: Any, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
    """Return quaternions for `obj_name` over `env_ids`.

    If `env_ids` is None, infer the number of environments from the first axis of
    the object's `root_state` and use all environments.
    """
    states = handler.get_states(mode="tensor")
    root_state = _get_root_state(states, obj_name)
    if env_ids is None:
        env_ids = list(range(root_state.shape[0]))
    return root_state[env_ids, 3:7]


def get_dof_pos(handler: Any, obj_name: str, joint_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
    """Return the DOF position tensor for a given joint across `env_ids`.

    Falls back to inferring `num_envs` from any available object/robot `root_state`
    when `env_ids` is None.
    """
    if env_ids is None:
        # Derive num_envs by reading any available root_state
        states_tensor = handler.get_states(mode="tensor")
        # Try to infer number of environments from any object/robot
        try:
            some_state = next(iter(states_tensor.objects.values()))
        except StopIteration:
            some_state = next(iter(states_tensor.robots.values()))
        num_envs = some_state.root_state.shape[0]
        env_ids = list(range(num_envs))

    states_tensor = handler.get_states(env_ids=env_ids, mode="tensor")
    nested_states = state_tensor_to_nested(handler, states_tensor)
    values = [({**es["objects"], **es["robots"]}[obj_name]["dof_pos"][joint_name]) for es in nested_states]
    device = getattr(handler, "device", None)
    if device is not None:
        return torch.tensor(values, dtype=torch.float32, device=device)
    return torch.tensor(values, dtype=torch.float32)
