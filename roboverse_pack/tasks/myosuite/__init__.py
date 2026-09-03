"""MyoSuite passthrough — musculoskeletal biological-actuator MuJoCo benchmark.

Wraps myosuite's 204 envs under a MyoSuite/<env_id> namespace. Idempotent.
"""

from __future__ import annotations

from loguru import logger as log

from ._passthrough import register_myosuite_passthrough

try:
    register_myosuite_passthrough()
except ImportError as exc:  # optional benchmark package absent: expected, keep quiet
    log.debug(f"[myosuite] passthrough registration skipped: {exc}")
except Exception as exc:  # anything else is a real bug and must not be hidden
    log.warning(f"[myosuite] passthrough registration failed: {exc!r}")
