"""The MetaSim kernel: the few names every benchmark port in RoboVerse is built from.

Twenty-odd benchmark families (ManiSkill, LIBERO, RLBench, CALVIN, mjlab, MuJoCo Playground,
HumanoidBench, SimplerEnv, RoboTwin, ...) were ported onto MetaSim. Measured across them, a port
needs five things, and this module is that set, re-exported from where each name is defined:

1. **Scenario** -- what exists: robots, objects, cameras, physics parameters (``ScenarioCfg`` and
   its entity configs). Every entity a task reads must be declared here; an entity that only lives
   inside a scene file is invisible to the state below and to every other backend.
2. **State** -- the one data type between simulator and content (``TensorState``: per-entity
   root/joint/body tensors, camera frames, query extras). Content reads states, never engines;
   a frame fixed to a body (an MJCF site) is derived from the state with ``site_pose``.
3. **Handler** -- a backend in five verbs: ``launch``, ``set_states(states, env_ids)``,
   ``set_dof_targets(actions)``, ``simulate()``, ``get_states(env_ids)``, plus names, capability
   flags and queries (``BaseQueryType``) for what a state does not carry (contacts).
4. **Task** -- one loop with hooks (``BaseTaskEnv``; ``RLTaskEnv`` adds joint-limit clipping and
   auto-reset): ``_process_action -> _apply_action -> simulate -> _on_step -> _reward /
   _terminated / _success / _time_out``, and ``reset -> _on_reset``. Success is a predicate over the
   state (the checkers); the registry makes a task addressable by name.
5. **Protocol** -- how a task is scored, separate from the task (``EvalProtocol``, ``evaluate``,
   ``BenchmarkSuite``): episodes, success rule, budgets, instructions, and a policy interface
   that runs in process or over the wire.

Everything else in MetaSim (randomization, renderers, viewers, teleop, IK, conversion tools) is
built on these and optional to a port. Importing this module loads no simulator backend.
"""

from __future__ import annotations

from metasim.bench import (
    BenchmarkResult,
    BenchmarkSuite,
    EvalProtocol,
    Policy,
    PolicyContext,
    TaskInstance,
    evaluate,
    register_suite,
)
from metasim.constants import PhysicStateType, SimType
from metasim.queries.base import BaseQueryType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.sim.base import BaseSimHandler
from metasim.task.base import BaseTaskEnv
from metasim.task.checkers import DetectedChecker, JointPosChecker, PositionShiftChecker, RelativeBboxDetector
from metasim.task.registry import get_task_class, register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.types import CameraState, ObjectState, RobotState, TensorState
from metasim.utils.configclass import configclass
from metasim.utils.site_util import SitePose, site_pose

__all__ = [
    "ArticulationObjCfg",
    "BaseActuatorCfg",
    "BaseQueryType",
    "BaseSimHandler",
    "BaseTaskEnv",
    "BenchmarkResult",
    "BenchmarkSuite",
    "CameraState",
    "DetectedChecker",
    "EvalProtocol",
    "JointPosChecker",
    "ObjectState",
    "PhysicStateType",
    "PinholeCameraCfg",
    "Policy",
    "PolicyContext",
    "PositionShiftChecker",
    "PrimitiveCubeCfg",
    "PrimitiveCylinderCfg",
    "PrimitiveSphereCfg",
    "RLTaskEnv",
    "RelativeBboxDetector",
    "RigidObjCfg",
    "RobotCfg",
    "RobotState",
    "ScenarioCfg",
    "SimParamCfg",
    "SimType",
    "SitePose",
    "TaskInstance",
    "TensorState",
    "configclass",
    "evaluate",
    "get_task_class",
    "register_suite",
    "register_task",
    "site_pose",
]
