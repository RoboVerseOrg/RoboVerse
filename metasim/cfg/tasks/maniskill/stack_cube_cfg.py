from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class StackCubeCfg(ManiskillTaskCfg):
    """.. Description:

    ### 📦 Source Metadata (from ManiSkill or other official sources)
    ### title:
    stack_cube
    ### group:
    Maniskill
    ### description:
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.

    ### randomizations:
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    ### success:
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)

    ### badges:
    - demos
    - dense
    - sparse

    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#stackcube-v1
    ### poster_url:
    (none)

    ---

    ### 🧩 Developer Defined Metadata (customized for RoboVerse or local usage)

    ### video_url:
    stack_cube.mp4

    ### platforms:
    - genesis
    - isaacgym
    - isaaclab
    - mujoco
    - sapien3

    ### notes:
    This task was adapted for RoboVerse.

    Robot must:
    - Grasp the red cube without disturbing the green cube
    - Carefully stack and release the cube without toppling

    Developer Tips:
    - Tune cube size or z-tolerance for easier stacking
    - Collision buffer adjustment may help with physical stability
    """

    episode_length = 250
    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=[0.04, 0.04, 0.04],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="base",
            size=[0.04, 0.04, 0.04],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[0.0, 0.0, 1.0],
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/stack_cube/v2"

    ## TODO: detect velocity
    checker = DetectedChecker(
        obj_name="cube",
        detector=RelativeBboxDetector(
            base_obj_name="base",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )
