"""Register all 25 SimplerEnv tasks (MetaSim-API BaseTaskEnv classes) under the gym ``SimplerEnv/``
namespace, and under MetaSim's task registry via ``@register_task``.

``gym.make("SimplerEnv/<task>")`` builds the MetaSim-native task (ScenarioCfg + sapien2 handler,
zero upstream dependency) wrapped in a thin gymnasium adapter.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger as log

from .coke_task import (
    PickCokeCan,
    PickHorizontalCokeCan,
    PickStandingCokeCan,
    PickVerticalCokeCan,
)
from .drawer_task import (
    CloseBottomDrawer,
    CloseDrawer,
    CloseMiddleDrawer,
    CloseTopDrawer,
    OpenBottomDrawer,
    OpenDrawer,
    OpenMiddleDrawer,
    OpenTopDrawer,
)
from .grasp_objects_task import MoveNearV0, MoveNearV1, SimplerPickObjectTask
from .place_task import (
    PlaceAppleInClosedTopDrawer,
    PlaceInClosedBottomDrawer,
    PlaceInClosedDrawer,
    PlaceInClosedMiddleDrawer,
    PlaceInClosedTopDrawer,
)
from .put_on_task import PutCarrotOnPlate, PutEggplantInBasket, PutSpoonOnTowel, StackGreenOnYellow

# task name -> (BaseTaskEnv class, kwargs)
TASK_MAP: dict[str, tuple[type, dict]] = {
    "google_robot_pick_coke_can": (PickCokeCan, {}),
    "google_robot_pick_horizontal_coke_can": (PickHorizontalCokeCan, {}),
    "google_robot_pick_vertical_coke_can": (PickVerticalCokeCan, {}),
    "google_robot_pick_standing_coke_can": (PickStandingCokeCan, {}),
    "google_robot_pick_object": (SimplerPickObjectTask, {}),
    "google_robot_move_near_v0": (MoveNearV0, {}),
    "google_robot_move_near_v1": (MoveNearV1, {}),
    "google_robot_move_near": (MoveNearV1, {}),
    "google_robot_open_drawer": (OpenDrawer, {}),
    "google_robot_open_top_drawer": (OpenTopDrawer, {}),
    "google_robot_open_middle_drawer": (OpenMiddleDrawer, {}),
    "google_robot_open_bottom_drawer": (OpenBottomDrawer, {}),
    "google_robot_close_drawer": (CloseDrawer, {}),
    "google_robot_close_top_drawer": (CloseTopDrawer, {}),
    "google_robot_close_middle_drawer": (CloseMiddleDrawer, {}),
    "google_robot_close_bottom_drawer": (CloseBottomDrawer, {}),
    "google_robot_place_in_closed_drawer": (PlaceInClosedDrawer, {}),
    "google_robot_place_in_closed_top_drawer": (PlaceInClosedTopDrawer, {}),
    "google_robot_place_in_closed_middle_drawer": (PlaceInClosedMiddleDrawer, {}),
    "google_robot_place_in_closed_bottom_drawer": (PlaceInClosedBottomDrawer, {}),
    "google_robot_place_apple_in_closed_top_drawer": (PlaceAppleInClosedTopDrawer, {}),
    "widowx_spoon_on_towel": (PutSpoonOnTowel, {}),
    "widowx_carrot_on_plate": (PutCarrotOnPlate, {}),
    "widowx_stack_cube": (StackGreenOnYellow, {}),
    "widowx_put_eggplant_in_basket": (PutEggplantInBasket, {}),
}
ENVIRONMENTS = list(TASK_MAP.keys())


def _make(task_name: str, **kwargs: Any):
    return MetaSimSimplerGymEnv(task_name, **kwargs)


class MetaSimSimplerGymEnv(gym.Env):
    """gymnasium adapter over a MetaSim-native SimplerEnv BaseTaskEnv (lazy build; one per process)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_name: str, **kwargs: Any) -> None:
        super().__init__()
        if task_name not in TASK_MAP:
            raise KeyError(f"unknown SimplerEnv task {task_name!r}")
        self.task_name = task_name
        self._cls, self._kw = TASK_MAP[task_name]
        self._task = None
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = None  # dict obs; env checker disabled at registration

    def _ensure(self):
        if self._task is None:
            self._task = self._cls(**self._kw)
        return self._task

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        t = self._ensure()
        return t.reset(seed=0 if seed is None else seed, options=options)

    def step(self, action):
        return self._task.step(action)

    def render(self):
        return self._task.get_obs()["image"]

    def close(self):
        self._task = None

    def get_language_instruction(self, *a, **k):
        return self._ensure().get_language_instruction()

    def advance_to_next_subtask(self, *a, **k):
        t = self._ensure()
        if hasattr(t, "cur_subtask_id"):
            t.cur_subtask_id = 1

    def is_final_subtask(self, *a, **k):
        t = self._ensure()
        return getattr(t, "cur_subtask_id", 1) == 1


def register_metasim_simpler_tasks(prefix: str = "SimplerEnv/") -> list[str]:
    """Register all 25 MetaSim-native SimplerEnv tasks under ``<prefix><task>`` (gym) and in
    MetaSim's task registry as ``simpler.<task>`` (discoverable/usable via metasim, not only gym)."""
    from gymnasium.envs.registration import register, registry

    try:
        from metasim.task.registry import register_task

        for name, (cls, _kw) in TASK_MAP.items():
            register_task(f"simpler.{name}")(cls)
    except Exception as e:  # pragma: no cover
        log.debug(f"[simpler_env_metasim] metasim register_task skipped: {e}")

    registered = []
    for name in ENVIRONMENTS:
        ns_id = f"{prefix}{name}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        register(
            id=ns_id,
            entry_point="roboverse_pack.tasks.simpler_env._metasim.registry:_make",
            kwargs={"task_name": name},
            max_episode_steps=200,
            disable_env_checker=True,
            order_enforce=False,
        )
        registered.append(ns_id)
    log.info(f"[simpler_env_metasim] registered {len(registered)} MetaSim-native SimplerEnv tasks under {prefix!r}")
    return registered
