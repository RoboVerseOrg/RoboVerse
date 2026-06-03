"""mjlab task wrappers.

Re-exports task classes so importing `roboverse_pack.tasks.mjlab` triggers
the `@register_task` side-effects and makes the tasks visible to
MetaSim's CLI / registry queries.

The deep physics ports are the ``mjlab.*_v2`` tasks (manager-based, running on
MetaSim's own MuJoCo/Newton handlers): obs + reward reproduce mjlab native at
machine-epsilon — see scripts/test_mjlab_v2_backward_compat.py and the
parity_* harnesses. ``cartpole_train`` holds the standalone 1:1 cartpole
training envs; ``_passthrough`` bridges any upstream mjlab task through
``gym.make`` without a physics port.

(The earlier ``mjlab.*`` non-``_v2`` tasks — BaseTaskEnv scaffolds in
cartpole.py/floating_base.py/lift_cube.py — were superseded by the ``_v2``
ports, which fixed their residuals (go1 0.16, lift_cube 1.6), and have been
removed.)
"""

from __future__ import annotations

from ._passthrough import register_mjlab_passthrough_tasks
from .cartpole_train import MjlabCartpoleBalanceTrain, MjlabCartpoleSwingupTrain

# Auto-register all mjlab tasks under MjlabPassthrough/<task_id>
try:
    register_mjlab_passthrough_tasks()
except Exception:
    pass  # mjlab not installed or registry issue
# v2 manager-based ports — register ``mjlab.*_v2`` task IDs via @register_task.
from . import cartpole_v2 as _cartpole_v2  # noqa: F401
from . import lift_cube_yam_v2 as _lift_cube_yam_v2  # noqa: F401
from . import velocity_g1_v2 as _velocity_g1_v2  # noqa: F401
from . import velocity_go1_v2 as _velocity_go1_v2  # noqa: F401

__all__ = [
    "MjlabCartpoleBalanceTrain",
    "MjlabCartpoleSwingupTrain",
]
