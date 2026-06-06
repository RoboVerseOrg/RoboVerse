"""Shared base for MetaSim-native ManiSkill tasks that reproduce native dynamics 1:1.

A subclass supplies a ``ScenarioCfg`` built from :func:`recipe.maniskill_panda_cfg`,
:func:`recipe.table_workspace_cfg`, and its own task objects, plus a success checker. This base:

* injects the ManiSkill-faithful ``SimParamCfg`` recipe + decimation onto the scenario, and
* overrides :meth:`step` so the task consumes ManiSkill's native action contract — a normalized
  ``Box(-1, 1)`` (8-dim for the panda) — turning it into the absolute joint-position drive targets
  the sapien3 handler applies, exactly like ManiSkill's ``pd_joint_delta_pos`` controller.

The result: the shipped ``maniskill.*`` task, run on the standard ``BaseTaskEnv`` + sapien3 handler
path, tracks the native ManiSkill ``physx_cpu`` rollout to PhysX float32 roundoff (~5e-6 on object
pose over dozens of aggressive steps; ~1e-6 under demo-like motion).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from metasim.task.base import BaseTaskEnv

from .control import PandaPDJointDeltaPos
from .recipe import DECIMATION, maniskill_sim_params


class ManiSkillNativeTask(BaseTaskEnv):
    """Base task: ManiSkill PhysX recipe + the pd_joint_delta_pos action contract."""

    controller = PandaPDJointDeltaPos()
    robot_name = "panda"

    def __init__(self, scenario, device=None):
        # Apply the ManiSkill-faithful physics recipe without mutating the shared class attribute.
        scenario = copy.copy(scenario)
        scenario.sim_params = maniskill_sim_params()
        scenario.decimation = DECIMATION
        super().__init__(scenario, device)
        self._sorted_to_active: list[int] | None = None
        self._robot_instance = None

    def _get_initial_states(self):
        """Per-env initial states built from the scenario defaults (robots + objects)."""
        robots = {}
        for r in self.scenario.robots:
            robots[r.name] = {
                "pos": list(r.default_position),
                "rot": list(r.default_orientation),
                "dof_pos": dict(r.default_joint_positions or {}),
            }
        objects = {}
        for o in self.scenario.objects:
            objects[o.name] = {"pos": list(o.default_position), "rot": list(o.default_orientation)}
        return [{"objects": objects, "robots": robots} for _ in range(self.num_envs)]

    # -- controller wiring ---------------------------------------------------
    def _ensure_joint_index(self) -> None:
        """Cache the active→sorted joint mapping the handler's ``set_dof_targets`` expects."""
        robot = self.handler.object_ids[self.robot_name]
        active = [j.get_name() for j in robot.get_active_joints()]
        sorted_names = self.handler.get_joint_names(self.robot_name, sort=True)
        self._sorted_to_active = [active.index(n) for n in sorted_names]
        self._robot_instance = robot

    def _targets_from_action(self, actions) -> torch.Tensor:
        """ManiSkill normalized action → absolute joint targets in the handler's sorted order."""
        if self._sorted_to_active is None:
            self._ensure_joint_index()
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        a = np.atleast_2d(a)
        qpos = np.asarray(self._robot_instance.get_qpos(), dtype=np.float32).ravel()  # active order
        out = np.empty((a.shape[0], len(self._sorted_to_active)), dtype=np.float32)
        for env_i in range(a.shape[0]):
            targets = self.controller.compute_targets(qpos, a[env_i])  # active order
            out[env_i] = [targets[i] for i in self._sorted_to_active]
        return torch.from_numpy(out)

    def step(self, actions):
        """Accept the native ManiSkill action; convert via the vendored controller, then step."""
        return super().step(self._targets_from_action(actions))

    # -- handler-reading helpers (used by the ported reward/success fns) ------
    def obj_pos(self, name):
        """World position (3,) of object ``name``."""
        return np.asarray(self.handler.object_ids[name].get_pose().p, dtype=np.float64)

    def obj_quat(self, name):
        """World quaternion (wxyz) of object ``name``."""
        return np.asarray(self.handler.object_ids[name].get_pose().q, dtype=np.float64)

    def tcp_pos(self):
        """World position of the panda ``panda_hand_tcp`` link."""
        for link in self.handler.object_ids[self.robot_name].get_links():
            if link.get_name() == "panda_hand_tcp":
                return np.asarray(link.get_pose().p, dtype=np.float64)
        raise RuntimeError("panda_hand_tcp link not found")

    def robot_qvel(self):
        return np.asarray(self.handler.object_ids[self.robot_name].get_qvel(), dtype=np.float64).ravel()

    def qvel_arm(self):
        """Robot qvel with the 2 finger DOFs dropped (panda)."""
        return self.robot_qvel()[:-2]

    def is_robot_static(self, threshold: float = 0.2) -> bool:
        return bool(np.max(np.abs(self.qvel_arm())) <= threshold)

    def is_grasped(self, name: str) -> bool:
        from .grasp import is_grasped as _is_grasped

        return _is_grasped(self.handler, name)

    # -- reward / success / goal wiring (set by the factory from the spec) ----
    reward_fn = None  # callable(task) -> float
    success_fn = None  # callable(task) -> bool
    goal_sampler = None  # callable(task, rng) -> np.ndarray (stored as self.goal_pos)
    goal_pos = None

    def _reward(self, states):
        if self.reward_fn is None:
            return super()._reward(states)
        return torch.tensor([float(self.reward_fn(self))] * self.num_envs, dtype=torch.float32)

    def _terminated(self, states):
        if self.success_fn is not None:
            return torch.tensor([bool(self.success_fn(self))] * self.num_envs, dtype=torch.bool)
        if getattr(self, "checker", None) is not None:
            return self.checker.check(self.handler, states)
        return super()._terminated(states)

    def reset(self, states=None, env_ids=None, seed=None):
        out = super().reset(states, env_ids, seed)
        if self.goal_sampler is not None:
            rng = np.random.RandomState(0 if seed is None else seed)
            self.goal_pos = np.asarray(self.goal_sampler(self, rng), dtype=np.float64)
        if getattr(self, "checker", None) is not None:
            self.checker.reset(self.handler, env_ids=env_ids)
        return out
