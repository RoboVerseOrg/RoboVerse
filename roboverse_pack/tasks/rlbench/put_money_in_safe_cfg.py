from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutMoneyInSafeTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_money_in_safev2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="dollar_stack",
            usd_path="roboverse_data/assets/rlbench/put_money_in_safe/dollar_stack/usd/dollar_stack.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        ArticulationObjCfg(
            name="safe_body",
            usd_path="roboverse_data/assets/rlbench/put_money_in_safe/safe_body/usd/safe_body.usd",
        ),
    ]
    # TODO: add checker
