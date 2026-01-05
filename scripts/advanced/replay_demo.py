from __future__ import annotations

import logging
import os
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image

from metasim.scenario.cameras import PinholeCameraCfg

# from metasim.scenario.randomization import RandomizationCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

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

    # Trajectory selection / replay controls
    traj_filepath: str | None = None
    demo_start_idx: int = 0
    demo_count: int = 1
    save_video_per_demo: bool = True
    save_image_per_demo: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int, robot: RobotCfg):
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


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


###########################################################
## Main
###########################################################
def main():
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scene_cfg = task_cls.scenario.scene if task_cls.scenario.scene is not None else args.scene
    if scene_cfg is None:
        log.warning("Scene is not specified by task or args; proceeding with None.")

    if args.robot == "None":
        scenario = task_cls.scenario.update(
            # robots=[args.robot],
            scene=scene_cfg,
            cameras=[camera],
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

    else:
        scenario = task_cls.scenario.update(
            robots=[args.robot],
            scene=scene_cfg,
            cameras=[camera],
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

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
    traj_filepath = args.traj_filepath or getattr(env, "traj_filepath", None)
    if traj_filepath is None:
        raise ValueError(
            "Trajectory filepath is not set. Pass --traj-filepath /abs/path/to/xxx_v2.pkl "
            "or use a task that defines env.traj_filepath (e.g. pick_place.track_il)."
        )
    ## Data
    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(
        traj_filepath, scenario.robots[0], env.handler
    )  # XXX: only support one robot
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ########################################################
    ## Main
    ########################################################

    os.makedirs("test_output", exist_ok=True)

    # Resolve replay demo indices.
    demo_start = max(0, int(args.demo_start_idx))
    demo_count = max(1, int(args.demo_count))
    demo_end = min(demo_start + demo_count, len(all_actions) if all_actions is not None else len(init_states))
    demo_indices = list(range(demo_start, demo_end))
    if not demo_indices:
        raise ValueError(f"No demos to replay: start={demo_start}, count={demo_count}, total={len(init_states)}")

    # Replay each demo sequentially (recommended for IsaacSim).
    for demo_i in demo_indices:
        log.info(f"[replay_demo] Replaying demo {demo_i}/{len(init_states) - 1}")

        # Slice one demo as a single-env batch so existing helper fns keep working.
        demo_init_states = [init_states[demo_i]]
        demo_actions = [all_actions[demo_i]] if all_actions is not None else None
        demo_states = [all_states[demo_i]] if (args.object_states and all_states is not None) else None

        # Per-demo saver (avoid OOM when demo_count is large).
        image_dir = args.save_image_dir
        video_path = args.save_video_path
        if args.save_video_per_demo and video_path is not None:
            root, ext = os.path.splitext(video_path)
            if not ext:
                ext = ".mp4"
            video_path = f"{root}_demo{demo_i:04d}{ext}"
        if args.save_image_per_demo and image_dir is not None:
            image_dir = os.path.join(image_dir, f"demo{demo_i:04d}")
        obs_saver = ObsSaver(image_dir=image_dir, video_path=video_path)

        # Reset with the demo init state.
        tic = time.time()
        obs, extras = env.reset(states=demo_init_states, env_ids=list(range(num_envs)))
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")
        obs_saver.add(obs)

        # Main loop for this demo
        step = 0
        while True:
            log.debug(f"[replay_demo] demo={demo_i} step={step}")
            tic = time.time()
            if args.object_states:
                if demo_states is None:
                    raise ValueError("All states are None, please check the trajectory file")
                states = get_states(demo_states, step, num_envs)
                env.handler.set_states(states)
                env.handler.refresh_render()
                obs = env.handler.get_states()

                # NOTE: This branch is mainly for state-only replay; success checker is task-specific.
                try:
                    success = env.checker.check(env.handler, obs)
                    if success.any():
                        log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
                    if success.all():
                        break
                except Exception:
                    pass
            else:
                if demo_actions is None:
                    raise ValueError("No actions found in the trajectory data")
                actions = get_actions(demo_actions, step, num_envs, scenario.robots[0])
                obs, reward, success, time_out, extras = env.step(actions)

                if time_out.any():
                    log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")
                    break

            toc = time.time()
            log.trace(f"Time to step: {toc - tic:.2f}s")

            tic = time.time()
            obs_saver.add(obs)
            toc = time.time()
            log.trace(f"Time to save obs: {toc - tic:.2f}s")
            step += 1

            if args.stop_on_runout and demo_actions is not None and get_runout(demo_actions, step):
                log.info("Run out of actions, stopping this demo")
                break

        obs_saver.save()
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()


if __name__ == "__main__":
    main()
