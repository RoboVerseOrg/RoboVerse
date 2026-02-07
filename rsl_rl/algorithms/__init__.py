"""Algorithm registry for RSL-RL (compat).

The upstream `rsl_rl` package imports `Distillation` in `rsl_rl.algorithms.__init__`.
Some installed versions of that module use PEP604 (`A | B`) annotations without
`from __future__ import annotations`, which breaks on Python < 3.10 (e.g. IsaacGym).

We keep `PPO` available everywhere and only expose `Distillation` on Python >= 3.10.
"""

from __future__ import annotations

import sys
from importlib import util as importlib_util
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


def _maybe_add_upstream_algorithms_path() -> None:
    """Add upstream `rsl_rl/algorithms` directory to this package's search path.

    We intentionally shadow `rsl_rl.algorithms.__init__` to avoid importing modules that are not
    Python 3.8 compatible in some environments (e.g., IsaacGym). However, we still want to load
    upstream algorithm implementations (PPO, etc.).

    The upstream `rsl_rl` is often installed as an editable package (PEP 660) which may not place
    its source directory on `sys.path`. We therefore discover the upstream package location via a
    known submodule (`rsl_rl.env`) and derive the sibling `algorithms` directory.
    """
    spec = importlib_util.find_spec("rsl_rl.env")
    if spec is None:
        return

    locations = list(spec.submodule_search_locations or [])
    if not locations:
        return

    base_dir = Path(locations[0]).parent
    candidate = base_dir / "algorithms"
    candidate_str = str(candidate)
    if candidate.is_dir() and candidate_str not in __path__:
        __path__.append(candidate_str)


_maybe_add_upstream_algorithms_path()

from .ppo import PPO

__all__ = ["PPO"]

if sys.version_info >= (3, 10):
    try:
        from .distillation import Distillation
    except Exception:
        Distillation = None  # type: ignore[assignment]
    else:
        __all__.append("Distillation")
