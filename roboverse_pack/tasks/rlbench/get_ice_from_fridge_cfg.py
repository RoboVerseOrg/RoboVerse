from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class GetIceFromFridgeTask(RLBenchTask):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/get_ice_from_fridgev2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="fridge_base",
            usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/fridge_base/usd/fridge_base.usd",
        ),
        RigidObjCfg(
            name="cup_visual",
            usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/cup_visual/usd/cup_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
