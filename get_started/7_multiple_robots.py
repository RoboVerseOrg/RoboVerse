from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.utils import ObsSaver
from metasim.cfg.robots import FrankaCfg, H1Cfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

FRANKA_CFG = FrankaCfg()
H1_CFG = H1Cfg()


@dataclass
class Args:
    num_envs: int = 1
    sim: str = "isaaclab"
    z_pos: float = 0.0
    decimation: int = 20


def main():
    args = tyro.cli(Args)
    scenario = ScenarioCfg(
        robots=[
            FRANKA_CFG.replace(name="franka_1"),
            FRANKA_CFG.replace(name="franka_2"),
            H1_CFG.replace(name="h1_1"),
            H1_CFG.replace(name="h1_2"),
        ],
        sim=args.sim,
        num_envs=args.num_envs,
        decimation=args.decimation,
    )
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(3.0, -3.0, 3.0), look_at=(0.0, -0.2, 0.0))]
    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    init_states = [
        {
            "robots": {
                "franka_1": {"pos": torch.tensor([1.0, 1.0, args.z_pos]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "franka_2": {"pos": torch.tensor([1.0, -1.0, args.z_pos]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "h1_1": {"pos": torch.tensor([-1.0, 1.0, args.z_pos + 1.0]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "h1_2": {
                    "pos": torch.tensor([-1.0, -1.0, args.z_pos + 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "objects": {},
        }
    ] * scenario.num_envs
    obs, extras = env.reset(states=init_states)

    os.makedirs("get_started/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/7_multirobot_{args.sim}.mp4")
    obs_saver.add(obs)

    step = 0
    while step < 200:
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        jn: (
                            torch.rand(1).item() * (robot.joint_limits[jn][1] - robot.joint_limits[jn][0])
                            + robot.joint_limits[jn][0]
                        )
                        for jn in robot.actuators.keys()
                        if robot.actuators[jn].fully_actuated
                    }
                }
                for robot in scenario.robots
            }
            for _ in range(scenario.num_envs)
        ]
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)
        step += 1

    env.handler.close()
    obs_saver.save()


if __name__ == "__main__":
    main()
