from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PutMoneyInSafeCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_money_in_safe/v2"
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
