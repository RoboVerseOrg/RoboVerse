from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCylinderCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask

# Import chess board from setup_chess
_CHESS_BOARD = RigidObjCfg(
    name="chess_board_base_visual",
    usd_path="roboverse_data/assets/rlbench/setup_chess/chess_board_base_visual/usd/chess_board_base_visual.usd",
    physics=PhysicStateType.GEOM,
)

_WHITE_CHECKERS = [
    PrimitiveCylinderCfg(
        name=f"checker{idx}",
        radius=0.017945,
        height=0.00718,
        color=[1.0, 1.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    )
    for idx in range(12)
]

_RED_CHECKERS = [
    PrimitiveCylinderCfg(
        name=f"checker{idx}",
        radius=0.017945,
        height=0.00718,
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    )
    for idx in range(12, 24)
]


@register_task("rlbench.setup_checkers", "setup_checkers", "franka.setup_checkers")
class SetupCheckersTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[_CHESS_BOARD] + _WHITE_CHECKERS + _RED_CHECKERS,
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/setup_checkers/v2/franka_v2.pkl.gz"
    # TODO: add checker
