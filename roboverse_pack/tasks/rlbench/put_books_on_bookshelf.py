from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_books_on_bookshelf", "put_books_on_bookshelf", "franka.put_books_on_bookshelf")
class PutBooksOnBookshelfTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bookshelf_visual",
                usd_path="roboverse_data/assets/rlbench/put_books_on_bookshelf/bookshelf_visual/usd/bookshelf_visual.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="book0_visual",
                usd_path="roboverse_data/assets/rlbench/put_books_on_bookshelf/book0_visual/usd/book0_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="book1_visual",
                usd_path="roboverse_data/assets/rlbench/put_books_on_bookshelf/book1_visual/usd/book1_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="book2_visual",
                usd_path="roboverse_data/assets/rlbench/put_books_on_bookshelf/book2_visual/usd/book2_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_books_on_bookshelf/v2/franka_v2.pkl.gz"
    # TODO: add checker
