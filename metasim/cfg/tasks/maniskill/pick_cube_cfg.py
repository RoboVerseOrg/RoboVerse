"""The pick up cube task from ManiSkill."""

from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class PickCubeCfg(ManiskillTaskCfg):
    """.. Description:

    ### 📦 Source Metadata (from ManiSkill or other official sources)
    ### title:
    pick_cube
    ### group:
    Maniskill
    ### description:
    A simple task where the objective is to grasp a red cube with the Panda robot and move it to a target goal position. This is also the baseline task to test whether a robot with manipulation capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

    ### randomizations:
    - the cube’s xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube’s z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    ### success:
    - the cube position is within goal_thresh (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)

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
    pick_cube.mp4

    ### platforms:


    ### notes:
    The pick up cube task from ManiSkill was adapted for RoboVerse.


    Developer Tips:
    - The robot is tasked to pick up a cube.
    - Note that the checker is not same as the original one (checking if the cube is near the target position). The current one checks if the cube is lifted up 0.1 meters.
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
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2"
    checker = PositionShiftChecker(
        obj_name="cube",
        distance=0.1,
        axis="z",
    )
