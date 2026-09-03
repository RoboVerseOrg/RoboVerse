"""mjlab cartpole tasks (balance + swingup).

mjlab's `cartpole.xml` is a *complete* scene — its own floor, rails,
cameras, plus the cart and pole — so we load it via `SceneCfg` rather
than as a `RobotCfg`-style wrap. That keeps the cart at world z=1
(authored pose) and avoids MetaSim's auto-ground plane interpenetrating
the cart's collision box. With those two settings the physics step is
bitwise identical to raw mujoco over 200 control steps.

The class-level `scenario` carries the right defaults; CLI users get
`mjlab.cartpole_balance` / `mjlab.cartpole_swingup` for free via the
`@register_task` side-effect at import time.
"""

from __future__ import annotations

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

from ._locator import lazy_scenario, mjlab_asset

_CARTPOLE_XML = "tasks/cartpole/cartpole.xml"


def _cartpole_scenario() -> ScenarioCfg:
    return ScenarioCfg(
        scene=SceneCfg(mjcf_path=mjlab_asset(_CARTPOLE_XML)),
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        decimation=5,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


@register_task("mjlab.cartpole_balance")
class MjlabCartpoleBalance(BaseTaskEnv):
    """Balance the pole upright (hinge_1 starts near 0)."""

    scenario = lazy_scenario(lambda: _cartpole_scenario())
    max_episode_steps = 5000  # mjlab episode_length_s = 50s / dt 0.01


@register_task("mjlab.cartpole_swingup")
class MjlabCartpoleSwingup(BaseTaskEnv):
    """Swing the pole up from hanging-down (hinge_1 starts at pi)."""

    scenario = lazy_scenario(lambda: _cartpole_scenario())
    max_episode_steps = 5000

    def _get_initial_states(self):
        # Swingup just resets the hinge angle; nothing else to override.
        return [
            {
                "objects": {},
                "robots": {},
                "scene": {"hinge_1": 3.141592653589793},
            }
        ]
