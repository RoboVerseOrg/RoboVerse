from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.task.registry import register_task
from metasim.utils.math import quat_from_euler_np

from .base_table import BaseCalvinTableTask

all_joint_names = {
    "franka": [
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "table": ["base__button", "base__switch", "base__slide", "base__drawer"],
}


@register_task("calvin.base_table_D")
class BaseCalvinTableTask_C(BaseCalvinTableTask):
    def __init__(self, *args, **kwargs):
        self.scenario.objects = [
            ArticulationObjCfg(
                name="table",
                scale=0.8,
                default_position=[0, 0, 0],
                default_orientation=[1, 0, 0, 0],
                fix_base_link=True,
                urdf_path="roboverse_data/assets/calvin/calvin_table_D/urdf/calvin_table_D.urdf",
                extra_resources=[
                    "roboverse_data/assets/calvin/calvin_table_D/textures/dark_wood__black_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/dark_wood__gray_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/dark_wood.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/light_wood__black_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/light_wood__gray_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/light_wood.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/wood__black_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/wood__gray_handle.png",
                    "roboverse_data/assets/calvin/calvin_table_D/textures/wood.png",
                    "roboverse_data/assets/calvin/calvin_table_D/meshes/base_link.mtl",
                    "roboverse_data/assets/calvin/calvin_table_D/meshes/drawer_link.mtl",
                    "roboverse_data/assets/calvin/calvin_table_D/meshes/plank_link.mtl",
                    "roboverse_data/assets/calvin/calvin_table_D/meshes/slide_link.mtl",
                    "roboverse_data/assets/calvin/calvin_table_D/meshes/switch_link.mtl",
                ],
            ),
            RigidObjCfg(
                name="pink_cube",
                scale=0.8,
                default_position=[1.28661989e-01, -3.77756105e-02, 4.59989266e-01 + 0.01],
                default_orientation=quat_from_euler_np(1.10200730e-04, 3.19760378e-05, -3.94522179e-01),
                fix_base_link=False,
                urdf_path="roboverse_data/assets/calvin/blocks/block_pink_big.urdf",
            ),
            RigidObjCfg(
                name="blue_cube",
                scale=0.8,
                default_position=[-2.83642665e-01, 8.05351014e-02, 4.60989238e-01 + 0.01],
                default_orientation=quat_from_euler_np(-1.10251078e-05, -5.25663348e-05, -9.06438129e-01),
                fix_base_link=False,
                urdf_path="roboverse_data/assets/calvin/blocks/block_blue_small.urdf",
            ),
            RigidObjCfg(
                name="red_cube",
                scale=0.8,
                default_position=[2.32403619e-01, -4.04295856e-02, 4.59990009e-01 + 0.01],
                default_orientation=quat_from_euler_np(4.12287744e-08, -8.05700103e-09, -2.17741510e00),
                fix_base_link=False,
                urdf_path="roboverse_data/assets/calvin/blocks/block_red_middle.urdf",
            ),
        ]
        super().__init__(*args, **kwargs)

    traj_filepath = "roboverse_data/trajs/calvin/calvin_traj_ann/env_D_out/task_100_v2.pkl"
