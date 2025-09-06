from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PlaceShapeInShapeSorterTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/place_shape_in_shape_sorterv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="shape_sorter",
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/rlbench/place_shape_in_shape_sorter/shape_sorter/usd/shape_sorter.usd",
        ),
        RigidObjCfg(
            name="triangular_prism",
            physics=PhysicStateType.RIGIDBODY,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd",  # reuse same asset
        ),
        RigidObjCfg(
            name="star_visual",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd",  # reuse same asset
        ),
        RigidObjCfg(
            name="moon_visual",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd",  # reuse same asset
        ),
        RigidObjCfg(
            name="cylinder",
            physics=PhysicStateType.XFORM,
            usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd",  # reuse same asset
        ),
        PrimitiveCubeCfg(
            name="cube",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.02089, 0.02089, 0.02089],
            color=[0.0, 0.85, 1.0],
        ),
    ]
    # TODO: add checker
