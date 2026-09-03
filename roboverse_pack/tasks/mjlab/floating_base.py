"""mjlab floating-base locomotion task scaffolds (g1, go1).

These tasks load mjlab's bare unitree_g1 / unitree_go1 MJCFs through
MetaSim's `RobotCfg` path. The fix-branch `fix/mujoco-mjlab-task-parity`
lets these compile despite the author-declared `<freejoint>` inside
each robot's root body (MetaSim now promotes the freejoint to the
attachment wrapper). With `add_default_ground=False` the physics step
matches raw mujoco within 1e-13 (g1) and ~0.16 rad (go1, residual due
to MetaSim's all-pairs self-collision-exclude semantics).
"""

from __future__ import annotations

from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

from ._locator import lazy_scenario, mjlab_asset

_G1_XML = "asset_zoo/robots/unitree_g1/xmls/g1.xml"
_GO1_XML = "asset_zoo/robots/unitree_go1/xmls/go1.xml"


def _floating_scenario(robot_name: str, mjcf_relpath: str, num_joints: int) -> ScenarioCfg:
    robot = RobotCfg(
        name=robot_name,
        num_joints=num_joints,
        fix_base_link=False,
        mjcf_path=mjlab_asset(mjcf_relpath),
        enabled_gravity=True,
        enabled_self_collisions=False,
        control_type={},  # users wire actuator targets directly via the handler
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


@register_task("mjlab.velocity_flat_g1")
class MjlabVelocityFlatG1(BaseTaskEnv):
    """Unitree G1 humanoid scaffold (29 DoF)."""

    scenario = lazy_scenario(lambda: _floating_scenario("g1", _G1_XML, num_joints=29))
    max_episode_steps = 4000  # mjlab episode_length_s = 20s / dt 0.005


@register_task("mjlab.velocity_rough_g1")
class MjlabVelocityRoughG1(BaseTaskEnv):
    """G1 humanoid on rough terrain (terrain wrapper not yet ported)."""

    scenario = lazy_scenario(lambda: _floating_scenario("g1", _G1_XML, num_joints=29))
    max_episode_steps = 4000


@register_task("mjlab.velocity_flat_go1")
class MjlabVelocityFlatGo1(BaseTaskEnv):
    """Unitree Go1 quadruped scaffold (12 DoF)."""

    scenario = lazy_scenario(lambda: _floating_scenario("go1", _GO1_XML, num_joints=12))
    max_episode_steps = 4000


@register_task("mjlab.velocity_rough_go1")
class MjlabVelocityRoughGo1(BaseTaskEnv):
    """Go1 quadruped on rough terrain (terrain wrapper not yet ported)."""

    scenario = lazy_scenario(lambda: _floating_scenario("go1", _GO1_XML, num_joints=12))
    max_episode_steps = 4000


@register_task("mjlab.tracking_flat_g1")
class MjlabTrackingFlatG1(BaseTaskEnv):
    """G1 humanoid reference-trajectory tracking on flat ground."""

    scenario = lazy_scenario(lambda: _floating_scenario("g1", _G1_XML, num_joints=29))
    max_episode_steps = 2000  # mjlab episode_length_s = 10s / dt 0.005


@register_task("mjlab.tracking_flat_g1_no_state_est")
class MjlabTrackingFlatG1NoStateEst(BaseTaskEnv):
    """Same physical scaffold as Tracking-Flat-G1; observation differs in mjlab."""

    scenario = lazy_scenario(lambda: _floating_scenario("g1", _G1_XML, num_joints=29))
    max_episode_steps = 2000
