"""Native SimplerEnv gym registration — `gym.make("SimplerEnv/<task>")` with ZERO upstream dep.

Registers all 25 SimplerEnv tasks under the ``SimplerEnv/`` gym namespace, each backed by the
vendored ``Native*Env`` (pure SAPIEN, reads only ``roboverse_data`` assets). No
``mani_skill2_real2sim`` / ``simpler_env`` import anywhere in this path — so the cloned packages
can be deleted and these ids still resolve and run. Observation/reward/success/instruction follow
the same contract the eval loops use.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger as log

# task name -> (family, kwargs for the native env constructor)
TASK_SPECS: dict[str, tuple[str, dict]] = {
    "google_robot_pick_coke_can": ("coke", {"default_orientation": None, "instruction": "pick coke can"}),
    "google_robot_pick_horizontal_coke_can": (
        "coke",
        {"default_orientation": "lr_switch", "instruction": "pick coke can"},
    ),
    "google_robot_pick_vertical_coke_can": (
        "coke",
        {"default_orientation": "laid_vertically", "instruction": "pick coke can"},
    ),
    "google_robot_pick_standing_coke_can": ("coke", {"default_orientation": "upright", "instruction": "pick coke can"}),
    "google_robot_pick_object": ("pickobject", {}),
    "google_robot_move_near_v0": ("movenear", {"variant": "v0"}),
    "google_robot_move_near_v1": ("movenear", {"variant": "v1"}),
    "google_robot_move_near": ("movenear", {"variant": "v1"}),
    "google_robot_open_drawer": ("drawer", {"is_close": False, "drawer_ids": ["top", "middle", "bottom"]}),
    "google_robot_open_top_drawer": ("drawer", {"is_close": False, "drawer_ids": ["top"]}),
    "google_robot_open_middle_drawer": ("drawer", {"is_close": False, "drawer_ids": ["middle"]}),
    "google_robot_open_bottom_drawer": ("drawer", {"is_close": False, "drawer_ids": ["bottom"]}),
    "google_robot_close_drawer": ("drawer", {"is_close": True, "drawer_ids": ["top", "middle", "bottom"]}),
    "google_robot_close_top_drawer": ("drawer", {"is_close": True, "drawer_ids": ["top"]}),
    "google_robot_close_middle_drawer": ("drawer", {"is_close": True, "drawer_ids": ["middle"]}),
    "google_robot_close_bottom_drawer": ("drawer", {"is_close": True, "drawer_ids": ["bottom"]}),
    "google_robot_place_in_closed_drawer": (
        "place",
        {"drawer_ids": ["top", "middle", "bottom"], "fixed_model_id": None},
    ),
    "google_robot_place_in_closed_top_drawer": ("place", {"drawer_ids": ["top"], "fixed_model_id": None}),
    "google_robot_place_in_closed_middle_drawer": ("place", {"drawer_ids": ["middle"], "fixed_model_id": None}),
    "google_robot_place_in_closed_bottom_drawer": ("place", {"drawer_ids": ["bottom"], "fixed_model_id": None}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "place",
        {"drawer_ids": ["top"], "fixed_model_id": "baked_apple_v2"},
    ),
    "widowx_spoon_on_towel": ("puton", {"task": "spoon"}),
    "widowx_carrot_on_plate": ("puton", {"task": "carrot"}),
    "widowx_stack_cube": ("puton", {"task": "stack"}),
    "widowx_put_eggplant_in_basket": ("puton", {"task": "eggplant"}),
}

ENVIRONMENTS = list(TASK_SPECS.keys())


def _build_native(task_name: str):
    """Construct the vendored native env for a task, resolving assets from roboverse_data."""
    from ._assets import paths

    fam, kw = TASK_SPECS[task_name]
    p = paths()
    if fam == "coke":
        from .env import NativeCokeCanEnv

        return NativeCokeCanEnv(
            urdf_path=p["urdf_google"],
            arena_glb=p["coke_scene"],
            coke_collision=p["coke_collision"],
            coke_visual=p["coke_visual"],
            overlay_png=p["coke_overlay"],
            **kw,
        )
    if fam == "pickobject":
        from .pick_object import NativePickObjectEnv

        return NativePickObjectEnv(
            urdf_path=p["urdf_google"],
            arena_glb=p["coke_scene"],
            models_dir=p["models_dir"],
            model_db_dir=p["model_db_dir"],
            overlay_png=p["coke_overlay"],
            **kw,
        )
    if fam == "movenear":
        from .move_near import NativeMoveNearEnv

        return NativeMoveNearEnv(
            urdf_path=p["urdf_google"],
            arena_glb=p["coke_scene"],
            models_dir=p["models_dir"],
            model_db_dir=p["model_db_dir"],
            overlay_png=f"{p['overlay_dir']}/google_move_near_real_eval_1.png",
            **kw,
        )
    if fam == "drawer":
        from .drawer import NativeDrawerEnv

        return NativeDrawerEnv(urdf_path=p["urdf_google"], cabinet_urdf=p["cabinet_urdf"], **kw)
    if fam == "place":
        from .place import NativePlaceInDrawerEnv

        return NativePlaceInDrawerEnv(
            urdf_path=p["urdf_google"],
            cabinet_urdf=p["cabinet_urdf"],
            models_dir=p["models_dir"],
            model_db_dir=p["model_db_dir"],
            **kw,
        )
    if fam == "puton":
        from .put_on import NativePutOnEnv

        return NativePutOnEnv(
            urdf_path=p["urdf_widowx"],
            scenes_dir=p["scenes_dir"],
            models_dir=p["models_dir"],
            model_db_dir=p["model_db_dir"],
            **kw,
        )
    raise ValueError(fam)


def _make_native_simpler_env(task_name: str, **kwargs: Any):
    return NativeSimplerGymEnv(task_name, **kwargs)


class NativeSimplerGymEnv(gym.Env):
    """gymnasium adapter over the vendored native SimplerEnv ports (zero upstream dep).

    Lazily builds the native env on first ``reset`` so registration/``gym.make`` stay cheap and the
    heavy SAPIEN scene is only created when used (one env per process — SAPIEN renderer is global).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_name: str, **kwargs: Any) -> None:
        super().__init__()
        if task_name not in TASK_SPECS:
            raise KeyError(f"unknown SimplerEnv task {task_name!r}")
        self.task_name = task_name
        self._kwargs = kwargs
        self._env = None
        # 7-D EE-delta + gripper action; the vendored controller clips/scales internally.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = None  # dict obs; env checker disabled at registration

    def _ensure(self):
        if self._env is None:
            self._env = _build_native(self.task_name)
            self._env.build()
        return self._env

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        env = self._ensure()
        return env.reset(seed=0 if seed is None else seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.get_obs()["image"]

    def close(self):
        self._env = None

    # SimplerEnv eval-loop helpers
    def get_language_instruction(self, *a, **k):
        return self._ensure().get_language_instruction()

    def advance_to_next_subtask(self, *a, **k):
        env = self._ensure()
        if hasattr(env, "advance_to_next_subtask"):
            return env.advance_to_next_subtask()
        return None

    def is_final_subtask(self, *a, **k):
        env = self._ensure()
        if hasattr(env, "cur_subtask_id"):
            return env.cur_subtask_id == 1
        return True


def register_native_simpler_tasks(prefix: str = "SimplerEnv/") -> list[str]:
    """Register all 25 native SimplerEnv tasks under ``<prefix><task>``. Idempotent."""
    from gymnasium.envs.registration import register, registry

    registered = []
    for name in ENVIRONMENTS:
        ns_id = f"{prefix}{name}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        register(
            id=ns_id,
            entry_point="roboverse_pack.tasks.simpler_env._native.gym_env:_make_native_simpler_env",
            kwargs={"task_name": name},
            max_episode_steps=80,
            disable_env_checker=True,
            order_enforce=False,
        )
        registered.append(ns_id)
    log.info(f"[simpler_env_native] registered {len(registered)} native SimplerEnv tasks under {prefix!r}")
    return registered
