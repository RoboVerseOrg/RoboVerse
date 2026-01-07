from __future__ import annotations

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
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState


@register_task("pick_place.track_il", "track_il")
class TrackILTaskFranka(BaseTaskEnv):
    """IL task for replay (terminate is always False except on last step if successful).

    Purpose: Allow `scripts/advanced/replay_demo.py` to directly replay v2 trajectory files
    saved by `eval_settle150` (each step is a dof_pos_target dict), and replay stably in
    MuJoCo single environment.

    Usage:
      export TRACK_IL_TRAJ_FILEPATH=/abs/path/to/eval_trajs/xxx_v2.pkl
      python scripts/advanced/replay_demo.py --task pick_place.track_il --sim mujoco --num_envs 1

    Optional:
      export TRACK_IL_DEMO_IDX=0  # Select which episode to use
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bbq_sauce",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq.usda",
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
            *[
                RigidObjCfg(
                    name=f"traj_marker_{i}",
                    urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                    mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                    usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                    scale=0.001,
                    physics=PhysicStateType.RIGIDBODY,
                    enabled_gravity=False,
                    collision_enabled=False,
                )
                for i in range(5)
            ],
        ],
        robots=["franka"],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
    )

    # Directory containing this file (roboverse_pack/tasks/pick_place/)
    TASK_DIR = Path(__file__).resolve().parent
    SUCCESS_XY_TOL_M = 0.10  # 10cm tolerance for success condition

    # Hardcoded default trajectory file path
    DEFAULT_TRAJ_FILEPATH = "roboverse_pack/tasks/pick_place/eval_trajs/track_franka_eval_settle_20260104_174810_v2.pkl"

    @classmethod
    def _resolve_traj_filepath(cls) -> str:
        # Explicit override via environment variable
        p = os.environ.get("TRACK_IL_TRAJ_FILEPATH", "").strip()
        if p:
            return p

        # Use hardcoded default trajectory file if it exists
        if os.path.exists(cls.DEFAULT_TRAJ_FILEPATH):
            return cls.DEFAULT_TRAJ_FILEPATH

        # Look for newest v2 traj under eval_trajs/ in the task directory
        eval_dir = cls.TASK_DIR / "eval_trajs"
        if eval_dir.exists():
            cands = sorted(eval_dir.glob("*_v2.pkl*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if cands:
                return str(cands[0])

        # Fallback: check repo root eval_trajs/ directory
        repo_root = cls.TASK_DIR.parents[3]
        eval_dir_fallback = repo_root / "eval_trajs"
        if eval_dir_fallback.exists():
            cands = sorted(eval_dir_fallback.glob("*_v2.pkl*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if cands:
                return str(cands[0])

        raise FileNotFoundError(
            "No trajectory file found. Please set environment variable TRACK_IL_TRAJ_FILEPATH=/abs/path/to/xxx_v2.pkl "
            f"or place trajectory files in {eval_dir} or {eval_dir_fallback}"
        )

    def __init__(self, scenario=None, device=None):
        # BaseTaskEnv.__init__ will read self.traj_filepath after handler launch and download/validate
        self.traj_filepath = self._resolve_traj_filepath()
        super().__init__(scenario=scenario or self.scenario, device=device)

    def _action_space(self) -> gym.Space:
        # replay_demo doesn't depend on action_space; provide a reasonable placeholder to avoid runner errors
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _reward(self, env_states: TensorState) -> torch.Tensor:
        # Only give success reward on the last frame of episode (for logging stats; replay doesn't depend on reward)
        last_step = self._episode_steps >= int(self.max_episode_steps) - 1
        success = self._success_xy(env_states)
        return (last_step & success).to(dtype=torch.float32, device=self.device)

    def _terminated(self, env_states: TensorState) -> torch.Tensor:
        # Note: BaseTaskEnv.step() calls _terminated() first, then increments _episode_steps,
        # so "last frame" corresponds to _episode_steps == max_episode_steps - 1
        last_step = self._episode_steps >= int(self.max_episode_steps) - 1
        success = self._success_xy(env_states)
        return (last_step & success).to(dtype=torch.bool, device=self.device)

    def _success_xy(self, env_states: TensorState) -> torch.Tensor:
        """Success condition: bbq_sauce (x,y) is within 10cm of basket (x,y) in both x and y directions."""
        bbq_pos = env_states.objects["bbq_sauce"].root_state[:, 0:3].to(self.device)
        basket_pos = env_states.objects["basket"].root_state[:, 0:3].to(self.device)
        dxy = torch.abs(bbq_pos[:, 0:2] - basket_pos[:, 0:2])
        tol = float(self.SUCCESS_XY_TOL_M)
        return (dxy[:, 0] <= tol) & (dxy[:, 1] <= tol)

    def _get_initial_states(self) -> list[dict] | None:
        # Load v2 traj and convert to v3; our demo_util has been patched to support eval_settle150-style actions
        init_states, all_actions, _ = get_traj(self.traj_filepath, self.scenario.robots[0], handler=self.handler)

        if not init_states:
            raise ValueError(f"No init_state found in trajectory file: {self.traj_filepath}")

        demo_idx = int(os.environ.get("TRACK_IL_DEMO_IDX", "0") or 0)
        demo_idx = max(0, min(demo_idx, len(init_states) - 1))

        # Align max_episode_steps to the loaded demo length (so replay_demo's timeout matches)
        if all_actions and len(all_actions) > demo_idx:
            try:
                self.max_episode_steps = len(all_actions[demo_idx])
            except Exception:
                pass

        # One init state per env: env i uses demo_idx + i (wrap around)
        selected = []
        for i in range(int(self.num_envs)):
            j = (demo_idx + i) % len(init_states)
            selected.append(init_states[j])
        return selected
