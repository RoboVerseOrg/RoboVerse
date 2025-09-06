from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PickUpCupTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/pick_up_cupv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="cup1_visual",
            usd_path="roboverse_data/assets/rlbench/pick_up_cup/cup1_visual/usd/cup1_visual.usd",
            physics=PhysicStateType.XFORM,
        ),
        RigidObjCfg(
            name="cup2_visual",
            usd_path="roboverse_data/assets/rlbench/pick_up_cup/cup2_visual/usd/cup2_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
