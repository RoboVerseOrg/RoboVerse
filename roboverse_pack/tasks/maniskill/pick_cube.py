from __future__ import annotations

from metasim.bench.prompts import PromptBank
from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.checkers.checkers import PositionShiftChecker
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.pick_cube", "pick_cube")
class PickCubeTask(ManiskillBaseTask):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            )
        ],
        robots=["franka"],
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"

    max_episode_steps = 250

    prompt_bank = PromptBank(
        canonical="Pick up the red cube and lift it 10 cm off the table.",
        train=(
            "Grasp the red cube and raise it about ten centimetres.",
            "Lift the red block up off the table.",
            "Pick the red cube up and hold it above the table.",
            "Grab the red cube and lift it upward by 0.1 m.",
            "Raise the red cube at least ten centimetres.",
            "Take hold of the red block and lift it clear of the table.",
            "Pick up the cube and bring it 10 cm higher.",
            "Close the gripper on the red cube and lift it.",
        ),
        heldout=(
            "Hoist the little red box a hand's width into the air.",
            "Get the red cube up off the surface by roughly a decimetre.",
            "Elevate the crimson block above the tabletop.",
            "Seize the red cube and carry it upward ten centimetres.",
        ),
    )

    checker = PositionShiftChecker(
        obj_name="cube",
        distance=0.1,
        axis="z",
    )
