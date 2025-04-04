from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class LightBulbInCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/light_bulb_in/v2"
    objects = [
        RigidObjCfg(
            name="bulb0",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bulb1",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb1/usd/bulb1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="bulb_holder0",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="bulb_holder1",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="lamp_base",
            usd_path="roboverse_data/assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
