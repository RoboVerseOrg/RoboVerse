from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.basketball_in_hoop", "basketball_in_hoop", "franka.basketball_in_hoop")
class BasketballInHoopTask(RLBenchTask):
    """RLBench basketball_in_hoop task, migrated from https://github.com/stepjam/RLBench/blob/master/rlbench/tasks/basketball_in_hoop.py."""

    max_episode_steps = 200
    traj_filepath = "roboverse_data/trajs/rlbench/basketball_in_hoop/v2/franka_v2.pkl.gz"
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="basket_ball_hoop_visual",
                usd_path="roboverse_data/assets/rlbench/basketball_in_hoop/basket_ball_hoop_visual/usd/basket_ball_hoop_visual_colored.usd",
                mesh_path="roboverse_data/assets/rlbench/basketball_in_hoop/basket_ball_hoop_visual/mesh/basket_ball_hoop_visual_colored.obj",
                physics=PhysicStateType.XFORM,
            ),
            RigidObjCfg(
                name="ball",
                usd_path="roboverse_data/assets/rlbench/basketball_in_hoop/ball/usd/ball_textured.usd",
                mesh_path="roboverse_data/assets/rlbench/basketball_in_hoop/ball/mesh/ball_textured.obj",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    checker = DetectedChecker(
        obj_name="ball",
        detector=RelativeBboxDetector(
            base_obj_name="basket_ball_hoop_visual",
            relative_quat=[-0.52125348147482, 0.66573167191783, 0.4157939106384, 0.33497995900038],
            relative_pos=[-0.060456029040702, -0.27864555866622, -0.17772846479924],
            checker_lower=[-0.025, -0.025, -0.05],
            checker_upper=[0.025, 0.025, 0.1],
            # debug_vis=True,
        ),
    )
