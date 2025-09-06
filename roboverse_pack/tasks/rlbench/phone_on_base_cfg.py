from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PhoneOnBaseTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/phone_on_basev2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="phone_visual",
            usd_path="roboverse_data/assets/rlbench/phone_on_base/phone_visual/usd/phone_visual.usd",
            physics=PhysicStateType.XFORM,
        ),
        RigidObjCfg(
            name="phone_case_visual",
            usd_path="roboverse_data/assets/rlbench/phone_on_base/phone_case_visual/usd/phone_case_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
