"""MyoSuite passthrough — musculoskeletal biological-actuator MuJoCo benchmark.

Wraps myosuite's 204 envs under a MyoSuite/<env_id> namespace. Idempotent.
"""

from __future__ import annotations

from ._passthrough import register_myosuite_passthrough

try:
    register_myosuite_passthrough()
except Exception:
    pass
