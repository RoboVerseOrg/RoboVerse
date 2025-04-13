import json
import os

from metasim.cfg.checkers import BaseChecker
from metasim.cfg.objects import NonConvexRigidObjCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

config_filepath = "roboverse_data/trajs/simpler_env/MoveNearGoogleInScene-v0/task_config.json"

object_config_dict = {
    "baked_opened_coke_can_v2": RigidObjCfg(
        name="baked_opened_coke_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_opened_coke_can_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_redbull_can_v2": RigidObjCfg(
        name="baked_opened_redbull_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_opened_redbull_can_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_apple_v2": RigidObjCfg(
        name="baked_apple_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_apple_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "blue_plastic_bottle": RigidObjCfg(
        name="blue_plastic_bottle",
        urdf_path="roboverse_data/assets/simpler_env/models/blue_plastic_bottle/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_pepsi_can_v2": RigidObjCfg(
        name="baked_opened_pepsi_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_opened_pepsi_can_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "orange": RigidObjCfg(
        name="orange",
        urdf_path="roboverse_data/assets/simpler_env/models/orange/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_7up_can_v2": RigidObjCfg(
        name="baked_opened_7up_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_opened_7up_can_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_soda_can_v2": RigidObjCfg(
        name="baked_opened_soda_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_opened_soda_can_v2/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_sponge_v2": RigidObjCfg(
        name="baked_sponge_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/baked_sponge_v2/mobility.urdf",
        fix_base_link=False,
    ),
}


@configclass
class SimplerEnvMoveNearCfg(BaseTaskCfg):
    # source_benchmark = BenchmarkType.SIMPLERENVGRASPSINGLEOPENEDCOKECAN
    # task_type = TaskType.TABLETOP_MANIPULATION

    # episode_length = 200

    # objects = [
    #     RigidObjCfg(
    #         name="opened_coke_can", urdf_path="assets/simpler_env/models/coke_can/mobility.urdf", fix_base_link=False
    #     ),
    #     NonConvexRigidObjCfg(
    #         name="scene",
    #         usd_path="assets/simpler_env/scenes/google_pick_coke_can_1_v4/google_pick_coke_can_1_v4.glb",
    #         urdf_path="assets/simpler_env/scenes/google_pick_coke_can_1_v4/mobility.urdf",
    #         fix_base_link=True,
    #         mesh_pose=[0, 0, 0, 0.707, 0.707, 0, 0],
    #     ),
    # ]
    # traj_filepath = MISSING
    # checker = BaseChecker()

    def __init__(self, subtask_id=0):
        # load json file
        with open(config_filepath) as f:
            config_json = json.load(f)
            assert subtask_id < len(config_json)
            self.config_dict = config_json[subtask_id]
        self.source_benchmark = BenchmarkType.SIMPLERENVMOVENEAR
        self.task_type = TaskType.TABLETOP_MANIPULATION
        self.episode_length = 200
        self.objects = [
            object_config_dict[self.config_dict["extras"]["triplet"][0]],
            object_config_dict[self.config_dict["extras"]["triplet"][1]],
            object_config_dict[self.config_dict["extras"]["triplet"][2]],
            NonConvexRigidObjCfg(
                name="scene",
                usd_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/google_pick_coke_can_1_v4.glb",
                urdf_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/mobility.urdf",
                fix_base_link=True,
                mesh_pose=[0, 0, 0, 0.707, 0.707, 0, 0],
            ),
        ]
        self.traj_filepath = os.path.join(
            "roboverse_data/trajs/simpler_env/MoveNearGoogleInScene-v0", self.config_dict["path"]
        )
        self.checker = BaseChecker()
        self.decimation = 20
