from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

import torch


def get_indices_from_substring(
    candidates_list: list[str] | tuple[str] | str,
    data_base: list[str],
    fullmatch: bool = True,
) -> torch.Tensor:
    """Get indices of items matching the candidates patterns.

    Args:
        candidates_list: Single pattern or list of patterns (supports regex if use_regex=True)
        data_base: List of names to search in
        fullmatch: If True, require full regex match; otherwise allow substring search.

    Returns:
        Sorted tensor of matching indices

    Examples:
        >>> get_indices_from_substring(".*ankle.*", ["left_ankle", "right_ankle", "knee"])
        tensor([0, 1])
        >>> get_indices_from_substring([".*ankle.*", ".*knee.*"], ["left_ankle", "knee"])
        tensor([0, 1])
    """
    found_indices = []
    if isinstance(candidates_list, str):
        candidates_list = (candidates_list,)
    assert isinstance(candidates_list, (list, tuple)), "candidates_list must be a list, tuple or string."

    for candidate in candidates_list:
        # Compile regex pattern for efficiency
        try:
            pattern = re.compile(candidate)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{candidate}': {e}") from e

        for i, name in enumerate(data_base):
            if fullmatch and pattern.fullmatch(name):
                found_indices.append(i)
            elif not fullmatch and pattern.search(name):
                found_indices.append(i)

    # Remove duplicates and sort
    found_indices = sorted(set(found_indices))
    return torch.tensor(found_indices, dtype=torch.int32, requires_grad=False)


def pattern_match(sub_names: dict[str, any], all_names: list[str]) -> dict[str, any]:
    """Pattern match the sub_names to all_names using regex."""
    matched_names = {_key: 0.0 for _key in all_names}
    for sub_key, sub_val in sub_names.items():
        pattern = re.compile(sub_key)
        for name in all_names:
            if pattern.fullmatch(name):
                matched_names[name] = sub_val
    return matched_names


def get_reward_fn(target: str, reward_functions: list[Callable] | str) -> Callable:
    """Resolve a reward function by name from a list or module path."""
    if isinstance(reward_functions, (list, tuple)):
        fn = next((f for f in reward_functions if f.__name__ == target), None)
    elif isinstance(reward_functions, str):
        reward_module = __import__(reward_functions, fromlist=[target])
        fn = getattr(reward_module, target, None)
    else:
        raise ValueError("reward_functions should be a list of functions or a string module path")
    if fn is None:
        raise KeyError(f"No reward function named '{target}'")
    return fn


def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
    """Construct arguments to `Vec` according to axis index."""
    zs = torch.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = torch.where(zs == 1.0, value, zs)
    params[0] = x_value
    return params.tolist()


@lru_cache(maxsize=128)
def hash_names(names: str | tuple[str]) -> str:
    """Hash a string or tuple of strings into a stable key."""
    if isinstance(names, str):
        names = (names,)
    assert isinstance(names, tuple) and all(isinstance(_, str) for _ in names), (
        "body_names must be a string or a list of strings."
    )
    hash_key = "_".join(sorted(names))
    return hash_key
