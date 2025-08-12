"""ManiSkill Pick-Cube task as a runnable BaseTaskEnv.

This refactors the previous config-only version to a task class that:
- Inherits the common BaseTaskEnv used by get_started examples
- Uses `_load_task_config` to populate scenario objects
- Implements `_terminated` to replace the old checker-based success logic

Success condition: the `cube` is lifted by at least 0.1 m along the z-axis from its reset height.
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.utils.state import TensorState
from scenario_cfg.objects import PrimitiveCubeCfg
from scenario_cfg.scenario import ScenarioCfg
from tasks.base import BaseTaskEnv
from tasks.checkers.position import position_shift_success
from tasks.registry import register_task


@register_task("maniskill.pick_cube", "pick_cube", "franka.pick_cube")
class PickCubeTask(BaseTaskEnv):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        # success when cube lifted in z by 0.1 m
        self._success_axis = "z"
        self._success_distance = 0.1
        self._cube_init_pos: torch.Tensor | None = None
        super().__init__(scenario, device)

    def _load_task_config(self, scenario: ScenarioCfg) -> ScenarioCfg:
        """Populate objects and episode length similar to get_started examples."""
        scenario = super()._load_task_config(scenario)

        # task horizon
        self.max_episode_steps = 250

        # add a red cube to the scene if not already present
        has_cube = any(getattr(obj, "name", None) == "cube" or obj == "cube" for obj in getattr(scenario, "objects", []))
        if not has_cube:
            scenario.objects = list(getattr(scenario, "objects", [])) + [
                PrimitiveCubeCfg(
                    name="cube",
                    size=(0.04, 0.04, 0.04),
                    mass=0.02,
                    physics=PhysicStateType.RIGIDBODY,
                    color=(1.0, 0.0, 0.0),
                )
            ]

        return scenario

    def _get_initial_states(self) -> list[dict] | None:
        """Optionally set initial object pose; robot pose comes from scenario/handler defaults.

        We place the cube on the table-ish height; exact table height depends on the scene.
        """
        # Keep it simple and leave robot states to defaults; just seed cube pose.
        # If the handler handles None gracefully, this can be set to None.
        try:
            num_envs = self.scenario.num_envs  # type: ignore[attr-defined]
        except Exception:
            # fallback to 1 when constructing initial states early
            num_envs = 1

        init_states = [
            {
                "objects": {
                    "cube": {
                        "pos": torch.tensor([0.45, 0.0, 0.025]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    }
                },
                "robots": {},
            }
            for _ in range(num_envs)
        ]
        return init_states

    def _observation(self, states: TensorState) -> torch.Tensor:
        """Return a simple observation: robot joint positions and end-effector position if available."""
        if not states.robots:
            return torch.empty((self.env.num_envs, 0), device=self.env.device)
        # use first robot
        robot_name, robot_state = next(iter(sorted(states.robots.items())))
        joint_pos = robot_state.joint_pos if robot_state.joint_pos is not None else torch.empty((self.env.num_envs, 0), device=self.env.device)
        ee_pos = None
        if robot_state.body_state is not None and robot_state.body_names:
            try:
                hand_idx = robot_state.body_names.index("panda_hand")
            except ValueError:
                hand_idx = 0
            ee_pos = robot_state.body_state[:, hand_idx, :3]
        else:
            ee_pos = torch.empty((self.env.num_envs, 0), device=self.env.device)
        return torch.cat([joint_pos, ee_pos], dim=1) if joint_pos.numel() or ee_pos.numel() else torch.empty((self.env.num_envs, 0), device=self.env.device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the cube has been lifted sufficiently along z from its reset height."""
        assert self._cube_init_pos is not None, "Initial cube position baseline must be set on reset()"
        cur_pos = self.env.get_pos("cube")
        return position_shift_success(cur_pos, self._cube_init_pos, self._success_axis, self._success_distance)

    def _reward(self, states: TensorState) -> torch.Tensor:
        # optional shaping: reward 1.0 upon success, else small penalty
        success = self._terminated(states)
        return success.float()

    # override to establish baseline after states are applied
    def reset(self, env_ids: list[int] | None = None):  # noqa: D401
        obs, info = super().reset(env_ids)
        # baseline must be taken after set_states inside super().reset
        ids = env_ids if env_ids is not None else list(range(self.env.num_envs))
        if self._cube_init_pos is None:
            self._cube_init_pos = torch.zeros((self.env.num_envs, 3), dtype=torch.float32, device=self.env.device)
        self._cube_init_pos[ids] = self.env.get_pos("cube", env_ids=ids)
        return obs, info
