"""Runner registry for RSL-RL (compat).

Upstream `rsl_rl.runners.__init__` imports both `OnPolicyRunner` and `DistillationRunner`.
Some environments in this repo (notably IsaacGym on Python 3.8) cannot import the distillation
algorithm module due to incompatible type annotations, which makes importing *any* runner fail.

We keep `OnPolicyRunner` available everywhere and only expose `DistillationRunner` on Python >= 3.10
(and only if the upstream implementation can be imported).
"""

from __future__ import annotations

import sys
from importlib import util as importlib_util
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


def _maybe_add_upstream_runners_path() -> None:
    # Discover upstream `rsl_rl` base directory via a stable sibling submodule, then add
    # its `runners` folder to this package's search path so `from .on_policy_runner import ...`
    # resolves to the upstream implementation.
    spec = importlib_util.find_spec("rsl_rl.env")
    if spec is None:
        return

    locations = list(spec.submodule_search_locations or [])
    if not locations:
        return

    base_dir = Path(locations[0]).parent
    candidate = base_dir / "runners"
    candidate_str = str(candidate)
    if candidate.is_dir() and candidate_str not in __path__:
        __path__.append(candidate_str)


_maybe_add_upstream_runners_path()


from .on_policy_runner import OnPolicyRunner

__all__ = ["OnPolicyRunner"]

if sys.version_info >= (3, 10):
    try:
        from .distillation_runner import DistillationRunner
    except Exception:
        DistillationRunner = None  # type: ignore[assignment]
    else:
        __all__.append("DistillationRunner")
