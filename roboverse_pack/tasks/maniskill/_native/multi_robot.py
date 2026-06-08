"""Shared base for MetaSim-native ManiSkill *multi-robot* tasks (e.g. TwoRobot*).

The single-robot :class:`ManiSkillNativeTask` consumes one robot's action; a multi-agent ManiSkill
task concatenates one ``pd_joint_delta_pos`` action per robot. This base splits the flat action into
per-robot slices, runs each through the vendored controller, sets the drive targets on every robot's
articulation, then steps the handler — reproducing the per-agent action contract 1:1 (verified
bitwise on TwoRobotPickCube). Reward/success/goal wiring mirrors the single-robot base.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from metasim.task.base import BaseTaskEnv

from .control import PandaPDJointDeltaPos
from .recipe import DECIMATION, maniskill_sim_params


class ManiSkillMultiRobotTask(BaseTaskEnv):
    """Base for multi-robot ManiSkill tasks; subclasses set ``robot_names`` + per-robot action dim."""

    controller = PandaPDJointDeltaPos()
    robot_names: list[str] = ["panda-0", "panda-1"]
    per_robot_action_dim = 8

    # reward/success/goal hooks (set by the subclass/factory)
    reward_fn = None
    success_fn = None
    goal_sampler = None
    goal_pos = None

    def __init__(self, scenario, device=None):
        scenario = copy.copy(scenario)
        scenario.sim_params = maniskill_sim_params()
        scenario.decimation = DECIMATION
        super().__init__(scenario, device)

    def _get_initial_states(self):
        robots = {}
        for r in self.scenario.robots:
            robots[r.name] = {
                "pos": list(r.default_position),
                "rot": list(r.default_orientation),
                "dof_pos": dict(r.default_joint_positions or {}),
            }
        objects = {
            o.name: {"pos": list(o.default_position), "rot": list(o.default_orientation)} for o in self.scenario.objects
        }
        return [{"objects": objects, "robots": robots} for _ in range(self.num_envs)]

    # -- handler-reading helpers (same surface as the single-robot base) ------
    def obj_pos(self, name):
        return np.asarray(self.handler.object_ids[name].get_pose().p, dtype=np.float64)

    def is_grasped(self, name, robot_name, max_angle=85.0):
        from .grasp import is_grasped as _is_grasped

        return _is_grasped(self.handler, name, robot_name=robot_name, max_angle=max_angle)

    def step(self, actions):
        """Split the flat action into per-robot slices, drive every robot, then step the handler."""
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        a = np.atleast_2d(a)[0]
        per = self.per_robot_action_dim
        for callback in self.pre_physics_step_callback:
            callback(actions)
        for i, rname in enumerate(self.robot_names):
            rob = self.handler.object_ids[rname]
            ajoints = rob.get_active_joints()
            q = np.asarray(rob.get_qpos(), dtype=np.float32).ravel()
            target = self.controller.compute_targets(q, a[i * per : (i + 1) * per])
            for k, j in enumerate(ajoints):
                j.set_drive_target(float(target[k]))
        self.handler.simulate()
        env_states = self.handler.get_states(mode="tensor")
        for callback in self.post_physics_step_callback:
            callback(env_states)
        rewards = self._reward(env_states)
        terminated = self._terminated(env_states)
        self._episode_steps = self._episode_steps + 1
        timeout = self._time_out(env_states)
        return (
            self._observation(env_states),
            rewards,
            terminated,
            timeout,
            {"privileged_observation": self._privileged_observation(env_states)},
        )

    def _reward(self, states):
        if self.reward_fn is None:
            return super()._reward(states)
        return torch.tensor([float(self.reward_fn(self))] * self.num_envs, dtype=torch.float32)

    def _terminated(self, states):
        if self.success_fn is not None:
            return torch.tensor([bool(self.success_fn(self))] * self.num_envs, dtype=torch.bool)
        return super()._terminated(states)

    def reset(self, states=None, env_ids=None, seed=None):
        out = super().reset(states, env_ids, seed)
        if self.goal_sampler is not None:
            if seed is not None or getattr(self, "_reset_rng", None) is None:
                self._reset_rng = np.random.RandomState(seed if seed is not None else 0)
            self.goal_pos = np.asarray(self.goal_sampler(self, self._reset_rng), dtype=np.float64)
        return out
