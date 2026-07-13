# Copyright (c) mjlab contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mjlab (https://github.com/mujocolab/mjlab).
# Changes: RoboVerse task scaffold that loads mjlab's `i2rt_yam` MJCF through MetaSim's `RobotCfg` and mirrors
#   mjlab's sim settings (dt / decimation / episode length); mjlab's lift-cube reward/goal shaping is not ported
#   here.
# Full license: roboverse_pack/tasks/mjlab/LICENSE
"""mjlab lift-cube task scaffold (YAM arm).

Loads the bare YAM MJCF as a fixed-base robot. The lift-cube goal /
reward shaping from mjlab's `Mjlab-Lift-Cube-Yam` is not (yet) ported
— this wrapper exists so the YAM physics scaffold is discoverable
through the registry, matching the harness used in
`tools/mjlab_integration/full_runner.py`.

Physics step matches raw mujoco within ~1.6 rad over 200 frames; the
gap is contact-exclude semantics rather than a model bug (see
report mjlab_integration/index.html).
"""

from __future__ import annotations

from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

from ._locator import mjlab_asset

_YAM_XML = "asset_zoo/robots/i2rt_yam/xmls/yam.xml"


def _yam_scenario() -> ScenarioCfg:
    robot = RobotCfg(
        name="yam",
        num_joints=7,
        fix_base_link=True,
        mjcf_path=mjlab_asset(_YAM_XML),
        enabled_gravity=True,
        enabled_self_collisions=False,
        control_type={},
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.002),
        decimation=4,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


@register_task("mjlab.lift_cube_yam")
class MjlabLiftCubeYam(BaseTaskEnv):
    """YAM arm scaffold (7 DoF, no actuators in the bare MJCF)."""

    scenario = _yam_scenario()
    max_episode_steps = 2500  # mjlab episode_length_s = 5s / dt 0.002


@register_task("mjlab.lift_cube_yam_rgb")
class MjlabLiftCubeYamRgb(BaseTaskEnv):
    """RGB-observation variant; physics is identical to the parent task."""

    scenario = _yam_scenario()
    max_episode_steps = 2500


@register_task("mjlab.lift_cube_yam_depth")
class MjlabLiftCubeYamDepth(BaseTaskEnv):
    """Depth-observation variant; physics is identical to the parent task."""

    scenario = _yam_scenario()
    max_episode_steps = 2500


@register_task("mjlab.multi_cube_seg_yam")
class MjlabMultiCubeSegYam(BaseTaskEnv):
    """Multi-cube segmentation variant; physics is identical to the parent task."""

    scenario = _yam_scenario()
    max_episode_steps = 2500
