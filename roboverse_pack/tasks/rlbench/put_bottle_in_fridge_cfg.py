from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutBottleInFridgeTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_bottle_in_fridgev2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="bottle_visual",
            usd_path="roboverse_data/assets/rlbench/put_bottle_in_fridge/bottle_visual/usd/bottle_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        ArticulationObjCfg(
            name="fridge_base",
            usd_path="roboverse_data/assets/rlbench/put_bottle_in_fridge/fridge_base/usd/fridge_base.usd",
        ),
    ]
    # TODO: add checker
