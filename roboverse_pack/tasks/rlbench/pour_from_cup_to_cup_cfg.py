from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PourFromCupToCupTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/pour_from_cup_to_cupv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="cup_distractor_visual0",
            usd_path="roboverse_data/assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual0/usd/cup_distractor_visual0.usd",
            physics=PhysicStateType.XFORM,
        ),
        RigidObjCfg(
            name="cup_distractor_visual1",
            usd_path="roboverse_data/assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual1/usd/cup_distractor_visual1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup_distractor_visual2",
            usd_path="roboverse_data/assets/rlbench/pour_from_cup_to_cup/cup_distractor_visual2/usd/cup_distractor_visual2.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup_target_visual",
            usd_path="roboverse_data/assets/rlbench/pour_from_cup_to_cup/cup_target_visual/usd/cup_target_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup_source_visual",
            usd_path="roboverse_data/assets/rlbench/pour_from_cup_to_cup/cup_source_visual/usd/cup_source_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
