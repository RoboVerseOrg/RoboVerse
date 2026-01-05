from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.state import TensorState


@register_task("pick_place.approach_il", "approach_il")
class ApproachILTaskFranka(BaseTaskEnv):
    """IL 版本的 approach 任务(最小实现).

    - 使用 saved pose 初始化 `bbq_sauce` / `basket` / `franka`
    - 暂不考虑成功条件: `_terminated()` 永远返回 False
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bbq_sauce",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            RigidObjCfg(
                name="basket",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/mjcf/basket.xml",
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    # 用于初始化 state 的 saved pose
    # 可用环境变量覆盖：
    #   export APPROACH_SAVED_POSE_PY_PATH=/abs/path/to/saved_poses_xxx.py
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_SAVED_POSE_PY_PATH = str(REPO_ROOT / "get_started/output/saved_poses_20260103_125412.py")
    SAVED_POSE_PY_PATH = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", DEFAULT_SAVED_POSE_PY_PATH)

    def _action_space(self) -> gym.Space:
        # 兼容大多数 IL runner：使用 [-1,1] 的 9 维动作（7 arm + 2 finger）
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        # 用户要求：先不管 terminate，直接 False
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @classmethod
    def _load_saved_pose_py(cls) -> dict:
        pose_path = os.environ.get("APPROACH_SAVED_POSE_PY_PATH", cls.SAVED_POSE_PY_PATH)
        p = Path(pose_path)
        if not p.exists():
            raise FileNotFoundError(f"SAVED_POSE_PY_PATH not found: {p}")

        mod_name = f"saved_pose_module_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import saved pose module from: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        poses = getattr(mod, "poses", None)
        if not isinstance(poses, dict):
            raise ValueError(f"`poses` must be a dict in: {p}")
        return poses

    def _get_initial_states(self) -> list[dict] | None:
        poses = self._load_saved_pose_py()
        if "objects" not in poses or "robots" not in poses:
            raise ValueError("Saved pose format error: missing 'objects' or 'robots'.")

        obj_name = "bbq_sauce"
        basket_name = "basket"
        robot_name = "franka"

        if obj_name not in poses["objects"]:
            raise ValueError(f"Saved pose missing object '{obj_name}'. Keys: {list(poses['objects'].keys())}")
        if basket_name not in poses["objects"]:
            raise ValueError(f"Saved pose missing object '{basket_name}'. Keys: {list(poses['objects'].keys())}")
        if robot_name not in poses["robots"]:
            raise ValueError(f"Saved pose missing robot '{robot_name}'. Keys: {list(poses['robots'].keys())}")

        init_one = {
            "objects": {
                obj_name: {
                    "pos": torch.as_tensor(poses["objects"][obj_name]["pos"], dtype=torch.float32),
                    "rot": torch.as_tensor(poses["objects"][obj_name]["rot"], dtype=torch.float32),
                },
                basket_name: {
                    "pos": torch.as_tensor(poses["objects"][basket_name]["pos"], dtype=torch.float32),
                    "rot": torch.as_tensor(poses["objects"][basket_name]["rot"], dtype=torch.float32),
                },
            },
            "robots": {
                robot_name: {
                    "pos": torch.as_tensor(poses["robots"][robot_name]["pos"], dtype=torch.float32),
                    "rot": torch.as_tensor(poses["robots"][robot_name]["rot"], dtype=torch.float32),
                    "dof_pos": dict(poses["robots"][robot_name].get("dof_pos", {}) or {}),
                }
            },
        }

        return [init_one for _ in range(int(self.num_envs))]
