from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_groceries_in_cupboard", "put_groceries_in_cupboard", "franka.put_groceries_in_cupboard")
class PutGroceriesInCupboardTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cupboard",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/cupboard/usd/cupboard.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="chocolate_jello_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/chocolate_jello_visual/usd/chocolate_jello_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="strawberry_jello_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/strawberry_jello_visual/usd/strawberry_jello_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="spam_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/spam_visual/usd/spam_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="sugar_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/sugar_visual/usd/sugar_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="crackers_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/crackers_visual/usd/crackers_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mustard_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/mustard_visual/usd/mustard_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="soup_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/soup_visual/usd/soup_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="tuna_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/tuna_visual/usd/tuna_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="coffee_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/coffee_visual/usd/coffee_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_groceries_in_cupboard/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task(
    "rlbench.put_all_groceries_in_cupboard", "put_all_groceries_in_cupboard", "franka.put_all_groceries_in_cupboard"
)
class PutAllGroceriesInCupboardTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cupboard",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/cupboard/usd/cupboard.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="chocolate_jello_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/chocolate_jello_visual/usd/chocolate_jello_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="strawberry_jello_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/strawberry_jello_visual/usd/strawberry_jello_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="spam_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/spam_visual/usd/spam_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="sugar_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/sugar_visual/usd/sugar_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="crackers_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/crackers_visual/usd/crackers_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mustard_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/mustard_visual/usd/mustard_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="soup_visual",
                usd_path="roboverse_data/assets/rlbench/put_groceries_in_cupboard/soup_visual/usd/soup_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_all_groceries_in_cupboard/v2/franka_v2.pkl.gz"
    # TODO: add checker
