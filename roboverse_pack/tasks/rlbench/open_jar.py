from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_jar", "open_jar", "franka.open_jar")
class OpenJarTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="jar0",
                usd_path="roboverse_data/assets/rlbench/open_jar/jar0/usd/jar0.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="jar1",
                usd_path="roboverse_data/assets/rlbench/open_jar/jar1/usd/jar1.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="jar_lid0",
                usd_path="roboverse_data/assets/rlbench/open_jar/jar_lid0/usd/jar_lid0.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="jar_lid1",
                usd_path="roboverse_data/assets/rlbench/open_jar/jar_lid1/usd/jar_lid1.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="target",
                usd_path="roboverse_data/assets/rlbench/open_jar/target/usd/target.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_jar/v2/franka_v2.pkl.gz"
    # TODO: add checker
