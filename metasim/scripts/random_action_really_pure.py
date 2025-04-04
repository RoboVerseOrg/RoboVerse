from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class


@dataclass
class Args:
    num_envs: int = 1
    sim: str = "isaaclab"
    robot: str = "franka"
    z_pos: float = 0.0


def main():
    args = tyro.cli(Args)
    scenario = ScenarioCfg(robot=args.robot, sim=args.sim, num_envs=args.num_envs)

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    init_states = [{scenario.robot.name: {"pos": [0.0, 0.0, args.z_pos]}}] * scenario.num_envs
    env.reset(states=init_states)

    robot_joint_limits = scenario.robot.joint_limits
    step = 0
    while True:
        log.debug(f"Step {step}")
        actions = [
            {
                "dof_pos_target": {
                    joint_name: (
                        torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
                        + robot_joint_limits[joint_name][0]
                    )
                    for joint_name in robot_joint_limits.keys()
                }
            }
            for _ in range(scenario.num_envs)
        ]
        env.step(actions)
        env.render()
        step += 1

    env.handler.close()


if __name__ == "__main__":
    main()
