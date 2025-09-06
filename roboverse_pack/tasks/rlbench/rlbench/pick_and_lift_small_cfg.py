from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PickAndLiftSmallCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/pick_and_lift_small/v2"
    objects = [
        RigidObjCfg(
            name="triangular_prism",
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd",
        ),
        RigidObjCfg(
            name="star_visual",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd",
        ),
        RigidObjCfg(
            name="moon_visual",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd",
        ),
        RigidObjCfg(
            name="cylinder",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd",
        ),
        PrimitiveCubeCfg(
            name="cube",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.02089, 0.02089, 0.02089],
            color=[0.0, 0.85, 1.0],
        ),
        PrimitiveSphereCfg(
            name="success_visual",
            physics=PhysicStateType.XFORM,
            color=[1.0, 0.14, 0.14],
            radius=0.04,
        ),
    ]
    # TODO: add checker
