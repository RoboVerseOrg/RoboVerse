"""Spec-compliant port of ManiSkill3's `PickCube-v1` task.

The existing `maniskill.pick_cube` (in `pick_cube.py`) is a
RoboVerse-flavored cube-lift task (4 cm cube, simple Δz checker, 250
max steps). It predates this integration push and we keep it as-is for
backward compatibility.

This module registers `maniskill.pick_cube_v1` — a closer match to the
canonical ManiSkill3 PickCube-v1 spec:

- 4 cm-overall cube (half-size 0.02) with `size=(0.04, 0.04, 0.04)`
- Goal: a 2.5 cm-radius sphere region above the cube
- Success: cube center within 2.5 cm of goal site
- Max episode steps: 50

What is *still* gapped vs canonical PickCube-v1 (out of scope for this
phase):

- Robot URDF is RoboVerse's stock Franka, not ManiSkill's
  `panda_v2.urdf` with finger friction 2.0
- Controller is per-joint position target, not `pd_joint_delta_pos`
- Reward is sparse (`is_detected`), not the dm_control-style
  `reaching + grasp + place·grasp + static·placed` formula
- Robot-static success criterion (`max|qvel[:-2]| < 0.2`) is not yet
  evaluated

These are tracked items 3/4/5/7/8 of the parity recipe in the
integration report:
<http://localhost:8000/#roboverse/maniskill_integration>.
"""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative3DSphereDetector
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask

# Match ManiSkill: half-size 0.02 → 4 cm cube; default density picks up
# ~0.064 kg with sapien's 1000 kg/m³ default — close to ManiSkill's
# implicit choice (they don't set mass either).
_CUBE = PrimitiveCubeCfg(
    name="cube",
    size=(0.04, 0.04, 0.04),
    physics=PhysicStateType.RIGIDBODY,
    color=(1.0, 0.0, 0.0),
)

# Goal site: kinematic sphere, no collision contribution to the
# physics step. PhysicStateType.XFORM = pure visual / pose-only.
_GOAL = PrimitiveSphereCfg(
    name="goal_site",
    radius=0.025,
    physics=PhysicStateType.XFORM,
    color=(0.0, 1.0, 0.0),
)

_DETECTOR = Relative3DSphereDetector(
    base_obj_name="goal_site",
    relative_pos=(0.0, 0.0, 0.0),
    radius=0.025,
)


@register_task("maniskill.pick_cube_v1", "pick_cube_v1")
class PickCubeV1Task(ManiskillBaseTask):
    """ManiSkill3 PickCube-v1 geometry + sparse success checker."""

    scenario = ScenarioCfg(
        objects=[_CUBE, _GOAL],
        robots=["franka"],
    )

    # We reuse the v2 PickCube demo data the existing task already
    # references — the demo trajectories ship a 4 cm cube too, so the
    # geometry change here is observation-compatible.
    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"

    max_episode_steps = 50  # canonical ManiSkill PickCube-v1 cap

    checker = DetectedChecker(
        obj_name="cube",
        detector=_DETECTOR,
    )
