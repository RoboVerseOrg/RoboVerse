from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_knife_in_knife_block", "put_knife_in_knife_block", "franka.put_knife_in_knife_block")
class PutKnifeInKnifeBlockTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="chopping_board_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/chopping_board_visual/usd/chopping_board_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="knife_block_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/knife_block_visual/usd/knife_block_visual.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="knife_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/knife_visual/usd/knife_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_in_knife_block/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task(
    "rlbench.put_knife_on_chopping_board", "put_knife_on_chopping_board", "franka.put_knife_on_chopping_board"
)
class PutKnifeOnChoppingBoardTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="chopping_board_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/chopping_board_visual/usd/chopping_board_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="knife_block_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/knife_block_visual/usd/knife_block_visual.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="knife_visual",
                usd_path="roboverse_data/assets/rlbench/put_knife_in_knife_block/knife_visual/usd/knife_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_on_chopping_board/v2/franka_v2.pkl.gz"
    # TODO: add checker
