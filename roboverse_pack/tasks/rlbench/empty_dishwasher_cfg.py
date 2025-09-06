from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class EmptyDishwasherTask(RLBenchTask):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/empty_dishwasherv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="dishwasher",
            usd_path="roboverse_data/assets/rlbench/empty_dishwasher/dishwasher/usd/dishwasher.usd",
        ),
        RigidObjCfg(
            name="dishwasher_plate_visual",
            usd_path="roboverse_data/assets/rlbench/empty_dishwasher/dishwasher_plate_visual/usd/dishwasher_plate_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
