from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class BasketballInHoopCfg(RLBenchTaskCfg):
    """
    .. Description:

    ### 📦 Source Metadata (from RLBench or other official sources)
    ### title:
    basketball_in_loop
    ### group:
    RLBench

    ---
    ### 🧩 Developer Defined Metadata (customized for RoboVerse or local usage)

    ### video_url:
    basketball_in_loop.mp4

    ### platforms:
    - isaaclab

    ### notes:
    This task was adapted for RoboVerse.
    RLBench basketball_in_hoop task, migrated from https://github.com/stepjam/RLBench/blob/master/rlbench/tasks/basketball_in_hoop.py.
    """

    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/basketball_in_hoop/v2"
    objects = [
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
    ]
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
