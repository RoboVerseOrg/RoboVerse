from __future__ import annotations

#########################################
## Setup logging
#########################################
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

#########################################
### Add command line arguments
#########################################
from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class Args:
    robot: str = "franka"
    num_envs: int = 1
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] = "isaaclab"


args = tyro.cli(Args)

#########################################
### Normal code
#########################################

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

# Import ObsSaver for saving observations
import sys

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "get_started"))
from utils import ObsSaver


def main():
    num_envs: int = args.num_envs
    task = get_task("pick_cube")
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(task=None, robots=[robot], cameras=[camera], sim=args.sim)

    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    ## Reset
    obs, extras = env.reset()

    # Initialize observation saver
    obs_saver = ObsSaver(video_path=f"roboverse_data/outputs/random_action_{args.robot}_{args.sim}.mp4")
    obs_saver.add(obs)

    step = 0
    q_min = torch.ones(len(robot.joint_limits.values()), device="cuda:0") * 100
    q_max = torch.ones(len(robot.joint_limits.values()), device="cuda:0") * -100

    for step in range(15):
        log.debug(f"Step {step}")

        # Generate random joint space actions
        j_limits = torch.tensor(list(robot.joint_limits.values()))
        j_ranges = j_limits[:, 1] - j_limits[:, 0]
        q = j_ranges.unsqueeze(0) * torch.rand((num_envs, robot.num_joints)) + j_limits[:, 0].unsqueeze(0)
        assert q.shape == (num_envs, robot.num_joints)
        assert torch.all(q >= j_limits[:, 0]) and torch.all(q <= j_limits[:, 1])
        q = q.to("cuda:0")

        actions = [
            {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
            for i_env in range(num_envs)
        ]
        q_min = torch.min(torch.stack([q_min, q[0]], -1), -1)[0]
        q_max = torch.max(torch.stack([q_max, q[0]], -1), -1)[0]

        log.info(f"q: {[f'{x:.2f}' for x in q[0].tolist()]}")
        log.info(f"q_min: {[f'{x:.2f}' for x in q_min.tolist()]}")
        log.info(f"q_max: {[f'{x:.2f}' for x in q_max.tolist()]}")

        for _ in range(30):
            obs, reward, success, time_out, extras = env.step(actions)
            env.handler.refresh_render()

        obs_saver.add(obs)

    obs_saver.save()
    env.handler.close()


if __name__ == "__main__":
    main()
