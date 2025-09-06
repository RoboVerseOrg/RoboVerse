from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PlugChargerInPowerSupplyTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/plug_charger_in_power_supplyv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="charger",
            usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/charger/usd/charger.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="task_wall",
            usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/task_wall/usd/task_wall.usd",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="plug",
            usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/plug/usd/plug.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
