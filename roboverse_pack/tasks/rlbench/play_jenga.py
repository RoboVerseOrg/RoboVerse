from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.play_jenga", "play_jenga", "franka.play_jenga")
class PlayJengaTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
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
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/play_jenga/v2/franka_v2.pkl.gz"
    # TODO: add checker
