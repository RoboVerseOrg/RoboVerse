"""Randomized-reset mixin for ManiSkill dense-reward RL tasks.

The replay-based ManiSkill ports (``ManiskillBaseTask``) need a demo
trajectory file for their initial states. Dense-reward *RL* tasks don't —
they want randomized object/goal poses each episode, like ManiSkill3's
``_initialize_episode``. This mixin provides that and skips the demo-data
download, so dense tasks are self-sufficient (and verifiable without
shipping demo trajectories).

Usage::

    class PickCubeV1DenseTask(DenseRLResetMixin, PickCubeV1Task):
        cube_z = 0.02
        goal_z_range = (0.10, 0.30)   # pick: goal floats above table
        ...
"""

from __future__ import annotations

import torch

from metasim.task.base import BaseTaskEnv


class DenseRLResetMixin:
    """Randomized resets (cube on table + goal pose) without demo data."""

    cube_z: float = 0.02
    xy_range: float = 0.1  # object xy ~ uniform[-xy_range, xy_range]
    goal_z_range: tuple[float, float] = (0.02, 0.02)  # on-table by default
    reset_seed: int = 0
    # Dense RL tasks generate their own randomized initial states, so they
    # need no replay demo trajectory — None makes BaseTaskEnv skip the download.
    traj_filepath = None

    # Skip ManiskillBaseTask's demo-trajectory download — call BaseTaskEnv
    # directly so the dense task doesn't depend on replay data. Still apply the
    # family's default robot: bypassing ManiskillBaseTask.__init__ must not also
    # bypass the robot fill-in, or _get_initial_states below would index robots[0]
    # on an empty list.
    def __init__(self, scenario=None, device=None) -> None:
        scenario = self._apply_default_robots(scenario)
        BaseTaskEnv.__init__(self, scenario, device)

    def _get_initial_states(self) -> list[dict]:
        g = torch.Generator().manual_seed(self.reset_seed)

        def u(lo: float, hi: float) -> float:
            return float(torch.rand(1, generator=g) * (hi - lo) + lo)

        robot = self.scenario.robots[0]
        defaults = dict(getattr(robot, "default_joint_positions", {}) or {})
        rpos = torch.tensor(list(getattr(robot, "default_pos", (0.0, 0.0, 0.0))), dtype=torch.float32)
        rrot = torch.tensor(list(getattr(robot, "default_rot", (1.0, 0.0, 0.0, 0.0))), dtype=torch.float32)
        obj_names = [o.name for o in self.scenario.objects]

        states: list[dict] = []
        for _ in range(self.num_envs):
            objs: dict[str, dict] = {}
            for name in obj_names:
                if "goal" in name.lower():
                    pos = torch.tensor(
                        [u(-self.xy_range, self.xy_range), u(-self.xy_range, self.xy_range), u(*self.goal_z_range)],
                        dtype=torch.float32,
                    )
                else:  # cube / manipuland — rest on the table surface
                    pos = torch.tensor(
                        [u(-self.xy_range, self.xy_range), u(-self.xy_range, self.xy_range), self.cube_z],
                        dtype=torch.float32,
                    )
                objs[name] = {"pos": pos, "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])}
            states.append({
                "objects": objs,
                "robots": {
                    robot.name: {
                        "pos": rpos.clone(),
                        "rot": rrot.clone(),
                        "dof_pos": dict(defaults),
                        "dof_vel": {k: 0.0 for k in defaults},
                    }
                },
            })
        self._initial_states = states
        return states
