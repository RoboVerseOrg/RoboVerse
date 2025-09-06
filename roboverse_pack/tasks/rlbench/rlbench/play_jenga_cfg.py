from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PlayJengaCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/play_jenga/v2"
    objects = [
        RigidObjCfg(
            name="Cuboid",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid0",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid1",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid2",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid3",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid4",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid5",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid6",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid7",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid8",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid9",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid10",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid11",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid12",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="target_cuboid",
            usd_path="roboverse_data/assets/rlbench/play_jenga/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
