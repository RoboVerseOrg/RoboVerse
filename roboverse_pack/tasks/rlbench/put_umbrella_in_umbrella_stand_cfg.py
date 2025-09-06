from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutUmbrellaInUmbrellaStandTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_umbrella_in_umbrella_standv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="umbrella_visual",
            usd_path="roboverse_data/assets/rlbench/put_umbrella_in_umbrella_stand/umbrella_visual/usd/umbrella_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="stand_visual",
            usd_path="roboverse_data/assets/rlbench/put_umbrella_in_umbrella_stand/stand_visual/usd/stand_visual.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
