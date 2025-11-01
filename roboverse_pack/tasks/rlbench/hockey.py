from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.hockey", "hockey", "franka.hockey")
class HockeyTask(RLBenchTask):
    max_episode_steps = 300
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="hockey_goal_visual",
                usd_path="roboverse_data/assets/rlbench/hockey/hockey_goal_visual/usd/hockey_goal_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="hockey_stick",
                usd_path="roboverse_data/assets/rlbench/hockey/hockey_stick/usd/hockey_stick.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="hockey_ball",
                usd_path="roboverse_data/assets/rlbench/hockey/hockey_ball/usd/hockey_ball.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/hockey/v2/franka_v2.pkl.gz"
    checker = DetectedChecker(
        obj_name="hockey_ball",
        detector=RelativeBboxDetector(
            base_obj_name="hockey_goal_visual",
            relative_pos=[0, -0.022753269108366, 0.083899194072109],
            relative_quat=[0.00012045016205838, 0.99993911881697, 0.010847761490316, 0.0020174791069082],
            checker_lower=[-0.056, -0.028, 0],
            # checker_upper=[0.056, 0.028, 0.168],
            checker_upper=[0.056, 0.045, 0.168],  # XXX
            # debug_vis=True,
        ),
    )
