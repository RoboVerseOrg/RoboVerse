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

from get_started.utils import ObsSaver
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_handler_class
from scenario_cfg.cameras import PinholeCameraCfg
from scenario_cfg.robots import FrankaCfg, H1Cfg
from scenario_cfg.scenario import ScenarioCfg

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

FRANKA_CFG = FrankaCfg()
H1_CFG = H1Cfg()


@dataclass
class Args:
    num_envs: int = 1
    sim: str = "mujoco"
    z_pos: float = 0.0
    decimation: int = 20
    max_steps: int = 100
    save_video: bool = True


def main():
    args = tyro.cli(Args)

    # Add camera for video recording
    camera = PinholeCameraCfg(
        name="main_camera",
        pos=[4.0, 4.0, 3.0],
        look_at=[0.0, 0.0, 1.0],
        width=640,
        height=480,
        data_types=["rgb"],
    )

    scenario = ScenarioCfg(
        robots=[
            FRANKA_CFG.replace(name="franka_1"),
            FRANKA_CFG.replace(name="franka_2"),
            H1_CFG.replace(name="h1_1"),
            H1_CFG.replace(name="h1_2"),
        ],
        cameras=[camera] if args.save_video else [],
        simulator=args.sim,
        num_envs=args.num_envs,
        decimation=args.decimation,
    )

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_handler_class(SimType(args.sim))
    env = env_class(scenario)
    env.launch()

    init_states = [
        {
            "robots": {
                "franka_1": {
                    "pos": torch.tensor([1.0, 1.0, args.z_pos]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
                "franka_2": {
                    "pos": torch.tensor([1.0, -1.0, args.z_pos]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
                "h1_1": {
                    "pos": torch.tensor([-1.0, 1.0, args.z_pos + 1]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
                "h1_2": {
                    "pos": torch.tensor([-1.0, -1.0, args.z_pos + 1]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
            "objects": {},
        }
    ] * scenario.num_envs
    env.set_states(init_states)

    # Initialize video saver
    obs_saver = None
    if args.save_video:
        video_path = f"get_started/output/7_multiple_robots_{args.sim}.mp4"
        obs_saver = ObsSaver(video_path=video_path)
        log.info(f"Will save video to: {video_path}")

    step = 0
    while step < args.max_steps:
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
        for robot in scenario.robots:
            env.set_dof_targets(robot.name, actions)
        env.simulate()

        # Save observations for video
        if obs_saver is not None:
            states = env.get_states()
            obs_saver.add(states)

        step += 1

    # Save video at the end
    if obs_saver is not None:
        obs_saver.save()
        log.info("Video saved successfully!")


if __name__ == "__main__":
    main()
