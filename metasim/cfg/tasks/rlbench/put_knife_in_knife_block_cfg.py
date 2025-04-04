from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_OBJECTS = [
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
]


@configclass
class PutKnifeInKnifeBlockCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_in_knife_block/v2"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class PutKnifeOnChoppingBoardCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_on_chopping_board/v2"
    objects = _OBJECTS
    # TODO: add checker
