"""Register the MetaSim-native ManiSkill tabletop suite from :mod:`_native.specs`.

Each spec becomes a ``maniskill.<name>_native`` task that runs on the ManiSkill-faithful PhysX recipe
through the standard ``BaseTaskEnv`` + sapien3 handler path, reproducing native ManiSkill dynamics
1:1 (object pose ~5e-6 vs the native ``physx_cpu`` rollout). ``pick_cube_native`` is registered in
its own module; this module covers the rest of the single-primitive suite.
"""

from __future__ import annotations

import torch

from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveMultiBoxCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.state import TensorState

from ._native.base import ManiSkillNativeTask
from ._native.recipe import maniskill_panda_cfg, table_workspace_cfg
from ._native.specs import TASK_SPECS


def _build_object(name, kind, geom, mass, color, pos, kinematic):
    common = dict(name=name, mass=mass, color=list(color), default_position=tuple(pos))
    if kinematic:
        common["fix_base_link"] = True
    if kind == "box":
        return PrimitiveCubeCfg(size=list(geom), **common)
    if kind == "sphere":
        return PrimitiveSphereCfg(radius=float(geom), **common)
    if kind == "multibox":
        return PrimitiveMultiBoxCfg(boxes=list(geom), **common)
    raise ValueError(f"unsupported primitive kind: {kind}")


def _make_task_cls(task_key: str, spec: dict):
    objects = [table_workspace_cfg()] + [_build_object(*o) for o in spec["objects"]]
    base_pos, base_quat = spec["base"]
    scenario = ScenarioCfg(
        robots=[maniskill_panda_cfg(base_position=base_pos, base_orientation=base_quat)],
        objects=objects,
    )
    object_names = [o[0] for o in spec["objects"]]
    success_fn = spec["success"]
    max_steps = spec["max_steps"]

    class _Task(ManiSkillNativeTask):
        pass

    _Task.scenario = scenario
    _Task.max_episode_steps = max_steps

    def _terminated(self, states: TensorState) -> torch.Tensor:
        poses = {n: self.handler.object_ids[n].get_pose().p for n in object_names}
        return torch.tensor([bool(success_fn(poses))] * self.num_envs, dtype=torch.bool)

    _Task._terminated = _terminated
    _Task.__name__ = "".join(p.capitalize() for p in task_key.split("_")) + "NativeTask"
    _Task.__qualname__ = _Task.__name__
    return _Task


# Register every spec as ``maniskill.<name>_native`` (+ a short ``<name>_native`` alias).
for _key, _spec in TASK_SPECS.items():
    _cls = _make_task_cls(_key, _spec)
    register_task(f"maniskill.{_key}_native", f"{_key}_native")(_cls)
    globals()[_cls.__name__] = _cls
