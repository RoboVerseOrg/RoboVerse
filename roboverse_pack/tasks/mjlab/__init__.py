"""mjlab task wrappers.

Re-exports task classes so importing `roboverse_pack.tasks.mjlab` triggers
the `@register_task` side-effects and makes the tasks visible to
MetaSim's CLI / registry queries.

Parity numbers vs raw mujoco (verified by scripts/parity_obs_reward_cartpole.py
and scripts/test_mjlab_v2_backward_compat.py):
- mjlab.cartpole_balance / .cartpole_swingup: bitwise identical
- mjlab.velocity_flat_g1: machine-epsilon (~1e-13)
- mjlab.velocity_flat_go1: 0.16 residual (contact-exclude semantics)
- mjlab.lift_cube_yam: 1.6 residual (contact-exclude semantics)
"""

from __future__ import annotations

from loguru import logger as log

from ._passthrough import register_mjlab_passthrough_tasks
from .cartpole import MjlabCartpoleBalance, MjlabCartpoleSwingup
from .cartpole_train import MjlabCartpoleBalanceTrain, MjlabCartpoleSwingupTrain

# Auto-register all mjlab tasks under MjlabPassthrough/<task_id>
try:
    register_mjlab_passthrough_tasks()
except ImportError as exc:  # mjlab not installed: expected, keep quiet
    log.debug(f"[mjlab] passthrough registration skipped: {exc}")
except Exception as exc:  # a registry bug must not be hidden
    log.warning(f"[mjlab] passthrough registration failed: {exc!r}")
# v2 manager-based ports — register ``mjlab.*_v2`` task IDs via @register_task.
from . import cartpole_v2 as _cartpole_v2  # noqa: F401
from . import lift_cube_yam_v2 as _lift_cube_yam_v2  # noqa: F401
from . import velocity_g1_v2 as _velocity_g1_v2  # noqa: F401
from . import velocity_go1_v2 as _velocity_go1_v2  # noqa: F401
from .floating_base import (
    MjlabTrackingFlatG1,
    MjlabTrackingFlatG1NoStateEst,
    MjlabVelocityFlatG1,
    MjlabVelocityFlatGo1,
    MjlabVelocityRoughG1,
    MjlabVelocityRoughGo1,
)
from .lift_cube import (
    MjlabLiftCubeYam,
    MjlabLiftCubeYamDepth,
    MjlabLiftCubeYamRgb,
    MjlabMultiCubeSegYam,
)

__all__ = [
    "MjlabCartpoleBalance",
    "MjlabCartpoleBalanceTrain",
    "MjlabCartpoleSwingup",
    "MjlabCartpoleSwingupTrain",
    "MjlabLiftCubeYam",
    "MjlabLiftCubeYamDepth",
    "MjlabLiftCubeYamRgb",
    "MjlabMultiCubeSegYam",
    "MjlabTrackingFlatG1",
    "MjlabTrackingFlatG1NoStateEst",
    "MjlabVelocityFlatG1",
    "MjlabVelocityFlatGo1",
    "MjlabVelocityRoughG1",
    "MjlabVelocityRoughGo1",
]
