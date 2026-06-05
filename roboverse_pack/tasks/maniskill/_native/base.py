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

    # -- checker plumbing (subclasses set ``checker``) -----------------------
    def _terminated(self, states):
        if getattr(self, "checker", None) is None:
            return super()._terminated(states)
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None, seed=None):
        out = super().reset(states, env_ids, seed)
        if getattr(self, "checker", None) is not None:
            self.checker.reset(self.handler, env_ids=env_ids)
        return out
