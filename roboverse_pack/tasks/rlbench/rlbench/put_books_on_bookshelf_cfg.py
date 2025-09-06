from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PutBooksOnBookshelfCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_books_on_bookshelf/v2"
    objects = [
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
    ]
    # TODO: add checker
