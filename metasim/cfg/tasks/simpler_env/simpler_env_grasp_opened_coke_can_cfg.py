from metasim.cfg.checkers import BaseChecker
from metasim.cfg.objects import NonConvexRigidObjCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

traj_filepath_list = [
    "roboverse_data/trajs/simpler_env/GraspSingleOpenedCokeCanInScene-v0/trace_1737906479.0466104_v2.pkl",
    "roboverse_data/trajs/simpler_env/GraspSingleOpenedCokeCanInScene-v0/trace_1737906931.4694195_v2.pkl",
    "roboverse_data/trajs/simpler_env/GraspSingleOpenedCokeCanInScene-v0/trace_1737907890.1946998_v2.pkl",
    "roboverse_data/trajs/simpler_env/GraspSingleOpenedCokeCanInScene-v0/trace_1737908281.1093962_v2.pkl",
]


@configclass
class SimplerEnvGraspOpenedCokeCanCfg(BaseTaskCfg):
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
        self.source_benchmark = BenchmarkType.SIMPLERENV
        self.task_type = TaskType.TABLETOP_MANIPULATION
        self.episode_length = 200
        self.objects = [
            RigidObjCfg(
                name="opened_coke_can",
                urdf_path="roboverse_data/assets/simpler_env/models/coke_can/mobility.urdf",
                fix_base_link=False,
            ),
            NonConvexRigidObjCfg(
                name="scene",
                usd_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/google_pick_coke_can_1_v4.glb",
                urdf_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/mobility.urdf",
                fix_base_link=True,
                mesh_pose=[0, 0, 0, 0.707, 0.707, 0, 0],
            ),
        ]
        self.traj_filepath = traj_filepath_list[subtask_id]
        self.checker = BaseChecker()
        self.decimation = 20
        self.required_states = True
