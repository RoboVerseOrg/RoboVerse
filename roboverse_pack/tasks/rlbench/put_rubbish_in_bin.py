from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_rubbish_in_bin", "put_rubbish_in_bin", "franka.put_rubbish_in_bin")
class PutRubbishInBinTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bin_visual",
                usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/bin_visual/usd/bin_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="tomato1_visual",
                usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/tomato1_visual/usd/tomato1_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="tomato2_visual",
                usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/tomato2_visual/usd/tomato2_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="rubbish_visual",
                usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/rubbish_visual/usd/rubbish_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_rubbish_in_bin/v2/franka_v2.pkl.gz"
    # TODO: add checker
