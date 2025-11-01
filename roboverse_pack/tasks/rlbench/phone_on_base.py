from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.phone_on_base", "phone_on_base", "franka.phone_on_base")
class PhoneOnBaseTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="phone_visual",
                usd_path="roboverse_data/assets/rlbench/phone_on_base/phone_visual/usd/phone_visual.usd",
                physics=PhysicStateType.XFORM,
            ),
            RigidObjCfg(
                name="phone_case_visual",
                usd_path="roboverse_data/assets/rlbench/phone_on_base/phone_case_visual/usd/phone_case_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/phone_on_base/v2/franka_v2.pkl.gz"
    # TODO: add checker
