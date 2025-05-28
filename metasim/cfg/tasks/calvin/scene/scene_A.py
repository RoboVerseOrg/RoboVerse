from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass


@configclass
class SceneA(BaseTaskCfg):
    objects = [
        ArticulationObjCfg(
            name="table",
            usd_path="roboverse_data/assets/calvin/COMMON/table_a/usd/table.usd",
            urdf_path="roboverse_data/assets/calvin/COMMON/table_a/urdf/calvin_table_A.urdf",
            fix_base_link=True,
            scale=0.8,
        ),
        PrimitiveCubeCfg(
            name="block_red",
            mass=0.1,  # origin is 1, smaller mass is easier to grasp
            size=(0.056, 0.04, 0.04),
            color=(1.0, 0.0, 0.0),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="block_blue",
            mass=0.1,  # origin is 1, smaller mass is easier to grasp
            size=(0.08, 0.04, 0.04),
            color=(0.0, 0.0, 1.0),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="block_pink",
            mass=0.1,  # origin is 1, smaller mass is easier to grasp
            size=(0.04, 0.04, 0.04),
            color=(1.0, 0.0, 1.0),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]


SCENE_A = SceneA()
