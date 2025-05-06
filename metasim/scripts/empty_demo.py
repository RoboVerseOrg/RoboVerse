from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from typing import Literal

import imageio
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "mujoco", "blender"] = "isaaclab"
    renderer: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "mujoco", "blender"] | None = (
        None
    )

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None
    stop_on_runout: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)
scenario.cameras = [PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=(0.0, 1.0, 0.0),
        physics=PhysicStateType.RIGIDBODY,
        mjcf_path="roboverse_data/assets/maniskill/cube/cube.mjcf",
    )
]

handler_class = get_sim_handler_class(SimType(args.sim))
handler = handler_class(scenario)
handler.launch()

handler.set_states([
    {
        "cube": {
            "pos": torch.tensor([0.3, 0.3, 0.05]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
        "franka": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": 0.0,
                "panda_joint5": 0.0,
                "panda_joint6": 0.0,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
        },
    }
])
if args.sim == "isaaclab":
    obs, _ = handler.reset()
else:
    handler.simulate()
    obs = handler.get_observation()

imageio.imwrite("test.png", obs["rgb"][0])
