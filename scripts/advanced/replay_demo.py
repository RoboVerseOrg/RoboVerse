from __future__ import annotations

import logging
import os
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.cameras import PinholeCameraCfg

# from metasim.scenario.randomization import RandomizationCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver

rootutils.setup_root(__file__, pythonpath=True)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "kitchen_open_bottom_drawer"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    # random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay.mp4"
    stop_on_runout: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int, robot: RobotCfg):
    # Multi-agent safe with no change: each ``env_actions[action_idx]`` is already
    # the per-step ``{robot_name: action}`` dict spanning every agent (get_traj
    # merged them), so this just forwards it. ``robot`` is unused/legacy.
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def get_runout(all_actions, action_idx: int):
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


def main():
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scene_cfg = task_cls.scenario.scene if task_cls.scenario.scene is not None else args.scene
    if scene_cfg is None:
        log.warning("Scene is not specified by task or args; proceeding with None.")

    # A task that already declares more than one robot is a multi-agent (e.g.
    # bimanual) cell: keep every agent the task defines and load each agent's
    # trajectory slice. In that case the single ``--robot`` flag is ignored.
    task_robots = task_cls.scenario.robots
    multi_agent = task_robots is not None and len(task_robots) > 1

    common = dict(
        scene=scene_cfg,
        cameras=[camera],
        # random=args.random,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    if multi_agent or args.robot == "None":
        scenario = task_cls.scenario.update(**common)
    else:
        scenario = task_cls.scenario.update(robots=[args.robot], **common)

    num_envs: int = scenario.num_envs

    if args.sim == "isaacsim":
        scenario.update(decimation=2)
        if scenario.robots[0].name == "franka":
            # use smaller stiffness and damping for fingers for fine-grained control
            from metasim.scenario.robot import BaseActuatorCfg

            scenario.robots[0].actuators["panda_finger_joint1"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )
            scenario.robots[0].actuators["panda_finger_joint2"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    traj_filepath = env.traj_filepath
    ## Data
    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    # Single robot -> pass the one RobotCfg (byte-identical legacy path); multiple
    # robots -> pass the list so get_traj merges every agent's slice into the
    # same 3-tuple shape (each per-step action dict spans all robots).
    robot_arg = scenario.robots if len(scenario.robots) > 1 else scenario.robots[0]
    init_states, all_actions, all_states = get_traj(traj_filepath, robot_arg, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)
    os.makedirs("test_output", exist_ok=True)

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    ## Main loop
    step = 0
    while True:
        log.debug(f"Step {step}")
        tic = time.time()
        if args.object_states:
            ## TODO: merge states replay into env.step function
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, num_envs)
            env.handler.set_states(states)
            env.handler.refresh_render()
            obs = env.handler.get_states(mode="tensor")

            ## XXX: hack
            success = env.checker.check(env.handler, obs)
            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break

        else:
            actions = get_actions(all_actions, step, num_envs, scenario.robots[0])
            obs, reward, success, time_out, extras = env.step(actions)

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

            if success.all() or time_out.all():
                break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("Run out of actions, stopping")
            break

    obs_saver.save()
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()


if __name__ == "__main__":
    main()
