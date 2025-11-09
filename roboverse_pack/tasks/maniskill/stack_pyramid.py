from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import CombinedDetector, RelativeBboxDetector
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.stack_pyramid", "stack_pyramid")
class StackPyramidCfg(ManiskillBaseTask):
    """The StackPyramid task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    StackPyramid
    ### group:
    Maniskill
    ### description:
    The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.
    ### randomizations:
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other.
    ### success:
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, or blue cube are grasped by the robot (robot must let go of the cubes)
    ### badges:
    - demo
    - sparse
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#stackpyramid-v1
    ### platforms:
    - mujoco
    - isaaclab
    ### notes:
    ### video_url:
    stack_pyramid.mp4.
    """  # noqa: D205

    episode_length = 800

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cubeA",
                size=[0.04, 0.04, 0.04],
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="cubeB",
                size=[0.04, 0.04, 0.04],
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=[0.0, 1.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="cubeC",
                size=[0.04, 0.04, 0.04],
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=[0.0, 0.0, 1.0],
            ),
        ]
    )

    dector = CombinedDetector(
        detector1=RelativeBboxDetector(
            base_obj_name="cubeA",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
            fixed=False,
        ),
        detector2=RelativeBboxDetector(
            base_obj_name="cubeB",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
            fixed=False,
        ),
    )

    checker = DetectedChecker(
        obj_name="cubeC",
        detector=dector,
    )

    traj_filepath = "roboverse_data/trajs/maniskill/stack_pyramid/v2"
