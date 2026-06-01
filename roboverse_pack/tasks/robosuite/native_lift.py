"""Back-compat shim: NativeLiftEnv now lives in the generic ``native`` module
(which hosts all robosuite-free task ports). Kept so existing imports work.
"""

from __future__ import annotations

from .native import NativeLiftEnv  # noqa: F401
