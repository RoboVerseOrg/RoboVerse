from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class LightBulbOutTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/light_bulb_outv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="bulb",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd",  # reuse light_bulb_in asset
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bulb_holder0",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd",  # reuse light_bulb_in asset
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="bulb_holder1",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd",  # reuse light_bulb_in asset
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="lamp_base",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd",  # reuse light_bulb_in asset
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
