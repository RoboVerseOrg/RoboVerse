from __future__ import annotations

from typing import Any

import torch


def _get_root_state(states: Any, obj_name: str) -> torch.Tensor:
    if hasattr(states, "objects") and obj_name in states.objects:
        return states.objects[obj_name].root_state
    if hasattr(states, "robots") and obj_name in states.robots:
        return states.robots[obj_name].root_state
    raise ValueError(f"Object {obj_name} not found in states")


def _normalize_env_slice(tensor: torch.Tensor, env_ids: list[int] | torch.Tensor | None, handler: Any) -> torch.Tensor:
    if env_ids is None:
        return tensor

    env_len = len(env_ids)
    if tensor.shape[0] == env_len:
        return tensor

    if isinstance(env_ids, torch.Tensor):
        idx = env_ids.to(device=tensor.device, dtype=torch.long)
    else:
        idx = torch.tensor(env_ids, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, idx)


def get_pos(handler: Any, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
    """Return positions for `obj_name` over `env_ids`.

    If `env_ids` is None, infer the number of environments from the first axis of
    the object's `root_state` and use all environments.
    """
    states = handler.get_states(env_ids=env_ids, mode="tensor")
    root_state = _get_root_state(states, obj_name)
    norm_root = _normalize_env_slice(root_state, env_ids, handler)
    return norm_root[:, :3]


def get_rot(handler: Any, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
    """Return quaternions for `obj_name` over `env_ids`.

    If `env_ids` is None, infer the number of environments from the first axis of
    the object's `root_state` and use all environments.
    """
    states = handler.get_states(env_ids=env_ids, mode="tensor")
    root_state = _get_root_state(states, obj_name)
    norm_root = _normalize_env_slice(root_state, env_ids, handler)
    return norm_root[:, 3:7]


def get_dof_pos(
    handler: Any,
    obj_name: str,
    joint_name: str,
    env_ids: list[int] | None = None,
) -> torch.FloatTensor:
    """Return joint_name positions of obj_name for each env in env_ids."""
    # Infer env_ids if none given
    if env_ids is None:
        ts = handler.get_states(mode="tensor")
        sample = next(iter(ts.objects.values()), None) or next(iter(ts.robots.values()))
        env_ids = list(range(sample.root_state.shape[0]))

    ts = handler.get_states(env_ids=env_ids, mode="tensor")

    # Locate joint tensor
    if obj_name in ts.objects:
        joint_tensor = ts.objects[obj_name].joint_pos
    elif obj_name in ts.robots:
        joint_tensor = ts.robots[obj_name].joint_pos
    else:
        raise KeyError(f"{obj_name} not found")

    j_idx = handler._get_joint_names(obj_name).index(joint_name)
    norm_joint = _normalize_env_slice(joint_tensor, env_ids, handler)
    dof_pos = norm_joint[:, j_idx]

    target_dev = getattr(handler, "device", joint_tensor.device)
    return dof_pos.to(dtype=torch.float32, device=target_dev)
