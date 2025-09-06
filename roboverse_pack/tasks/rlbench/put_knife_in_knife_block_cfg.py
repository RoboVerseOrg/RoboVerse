from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

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
class PutKnifeInKnifeBlockTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_in_knife_blockv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class PutKnifeOnChoppingBoardTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_knife_on_chopping_boardv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
