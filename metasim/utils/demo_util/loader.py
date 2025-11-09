"""Sub-module containing utilities for loading and saving trajectories.

Adds a robust unpickler that can remap legacy/internal module paths that sometimes
appear in pickled artifacts created under different NumPy versions (e.g.,
"numpy._core" -> "numpy.core"). This avoids environment/version-specific
ModuleNotFoundError when loading v2 .pkl trajectories.
"""

import gzip
import json
import pickle
from typing import Any

import yaml


class _RenameModuleUnpickler(pickle.Unpickler):
    """Unpickler that remaps known legacy module paths to current ones.

    Currently handles:
    - numpy._core -> numpy.core
    """

    # Prefix remaps: if module starts with any key, replace that prefix once
    _MODULE_PREFIX_MAP = {
        "numpy._core": "numpy.core",
    }

    def find_class(self, module: str, name: str) -> Any:
        # Apply exact and prefix-based remaps for legacy module paths
        new_module = module
        # Exact match first
        if module in self._MODULE_PREFIX_MAP:
            new_module = self._MODULE_PREFIX_MAP[module]
        else:
            # Prefix match (e.g., numpy._core.something -> numpy.core.something)
            for old_prefix, new_prefix in self._MODULE_PREFIX_MAP.items():
                if module.startswith(old_prefix + "."):
                    new_module = module.replace(old_prefix, new_prefix, 1)
                    break
        return super().find_class(new_module, name)


def _safe_pickle_load(fp) -> Any:
    """Attempt pickle.load, falling back to a module-remapping unpickler.

    This handles cross-environment pickles that reference modules like
    "numpy._core" which may not exist in the current NumPy wheel.
    """
    try:
        return pickle.load(fp)
    except ModuleNotFoundError as e:
        # Rewind and retry with the remapping unpickler for known cases
        try:
            fp.seek(0)
        except Exception:
            # Some file-like objects may not support seek; ignore and reopen at caller.
            raise e
        return _RenameModuleUnpickler(fp).load()


def load_traj_file(path: str):
    """Load a trajectory from a file.

    Args:
        path: The path to the trajectory file.

    Returns:
        The trajectory.
    """
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            try:
                return _safe_pickle_load(f)
            except ModuleNotFoundError:
                # Reopen and force remapping unpickler in case seek failed above
                with open(path, "rb") as f2:
                    return _RenameModuleUnpickler(f2).load()
    elif path.endswith(".pkl.gz"):
        # gzip files support seek, but to be safe, reopen on fallback
        with gzip.open(path, "rb") as f:
            try:
                return _safe_pickle_load(f)
            except ModuleNotFoundError:
                with gzip.open(path, "rb") as f2:
                    return _RenameModuleUnpickler(f2).load()
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def save_traj_file(data: dict, path: str):
    """Save a trajectory to a file.

    Args:
        data: The trajectory to save.
        path: The path to save the trajectory.
    """
    if path.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif path.endswith(".pkl.gz"):
        with gzip.open(path, "wb", compresslevel=1) as f:
            pickle.dump(data, f)
    elif path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(data, f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "w") as f:
            yaml.dump(data, f)
    else:
        raise ValueError(f"Unsupported file extension: {path}")
