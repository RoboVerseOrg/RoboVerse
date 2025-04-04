from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

# A list of all the subtasks for the RLAffordOpenDoor task (recorded trajectory)
subtask_config_list = [
    (
        "roboverse_data/assets/rlafford/44781_link_0/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_44781_link_0_1736703883.3484986_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/44781_link_1/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_44781_link_1_1736703884.2928157_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45007_link_1/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45007_link_1_1736703885.3024418_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45162_link_0/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45162_link_0_1736703886.295176_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45168_link_1/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45168_link_1_1736703887.29869_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45176_link_0/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45176_link_0_1736703888.2908022_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45194_link_0/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45194_link_0_1736703889.2496083_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45194_link_1/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45194_link_1_1736703890.2499464_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45238_link_0/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45238_link_0_1736703891.2527387_v2.pkl",
    ),
    (
        "roboverse_data/assets/rlafford/45238_link_1/mobility.urdf",
        "roboverse_data/trajs/rlafford/OpenDoor-v0/trace_45238_link_1_1736703892.244452_v2.pkl",
    ),
]


@configclass
class RlAffordOpenDoorCfg(BaseTaskCfg):
    """Base class for the RLAffordOpenDoor task cfg."""

    def __init__(self, subtask_id=0):
        """Initialize the RLAffordOpenDoor task cfg."""
        source_benchmark = BenchmarkType.RLAOPENDOOR
        task_type = TaskType.TABLETOP_MANIPULATION
        object_path, traj_path = subtask_config_list[subtask_id]
        self.objects = [
            ArticulationObjCfg(
                name="cabinet",
                urdf_path=object_path,
                fix_base_link=True,
            )
        ]
        self.episode_length = 250
        self.traj_filepath = traj_path
        self.joint_name = "joint" + "_" + object_path.split("/")[-2].split("_")[-1]
        self.checker = JointPosChecker("cabinet", self.joint_name, "ge", 0.05)
        self.decimation = 1
        self.required_states = True
