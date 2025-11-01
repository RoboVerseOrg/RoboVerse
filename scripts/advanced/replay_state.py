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
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

rootutils.setup_root(__file__, pythonpath=True)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.utils.kinematics import get_ee_state


@configclass
class Args:
    """Replay trajectory for a given task."""

    task: str = "put_banana"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx", "isaacsim"] = (
        "sapien3"
    )
    renderer: (
        Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3", "isaacsim"] | None
    ) = None

    num_envs: int = 1
    try_add_table: bool = True
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = True

    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay.mp4"
    stop_on_runout: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def _suffix_path(p: str | None, suffix: str) -> str | None:
    if p is None:
        return None
    base, ext = os.path.splitext(p)
    if ext:
        return f"{base}_{suffix}{ext}"
    return f"{p}_{suffix}"


def get_actions(all_actions, action_idx: int, num_envs: int, robot: RobotCfg):
    envs_actions = all_actions[:num_envs]
    return [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    return [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]


def get_runout(all_actions, action_idx: int):
    return all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])


class ObsSaver:
    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []
        self.image_idx = 0

    def add(self, state: TensorState):
        if self.image_dir is None and self.video_path is None:
            return
        try:
            rgb_data = next(iter(state.cameras.values())).rgb  # (N, H, W, 3) or (1, H, W, 3)
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(max(1, rgb_data.shape[0] ** 0.5)))
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


def main():
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    num_envs: int = scenario.num_envs

    device = torch.device("cpu")
    t0 = time.time()
    env = task_cls(scenario, device=device)
    log.trace(f"Time to launch: {time.time() - t0:.2f}s")

    traj_filepath = env.traj_filepath
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    t0 = time.time()
    init_states, all_actions, all_states = get_traj(traj_filepath, scenario.robots[0], env.handler)
    log.trace(f"Time to load data: {time.time() - t0:.2f}s")

    # Check if states are available in trajectory
    if all_states is None or len(all_states) == 0:
        log.error("No states found in trajectory file. Please ensure the trajectory was saved with --save-states")
        env.close()
        return

    log.info(f"Loaded {len(all_states)} episodes with states")
    log.info(f"Episode 0 has {len(all_states[0])} states")

    os.makedirs("test_output", exist_ok=True)

    saver_state = ObsSaver(
        image_dir=args.save_image_dir,
        video_path=args.save_video_path,
    )

    t0 = time.time()
    env.reset()

    # Use states from first episode (all_states[0] contains list of states for episode 0)
    episode_states = all_states[0]  # List of state dicts for each timestep
    total = len(episode_states)

    log.info(f"Replaying {total} states from episode 0")

    for step in range(total):
        log.debug(f"[STATE] Step {step}/{total - 1}")

        # Get state dict for this step
        state_dict = episode_states[step]

        # Set the state in the handler (expects list of state dicts for multi-env)
        env.handler.set_states([state_dict] * num_envs)

        env.handler.refresh_render()
        obs = env.handler.get_states()
        saver_state.add(obs)

        # Log EE state
        ee_states = get_ee_state(obs, robot_config=scenario.robots[0])
        log.debug(f"EE state at step {step}: {ee_states}")

        try:
            success = env.checker.check(env.handler)
            if success.any():
                log.info(f"[STATE] Env {success.nonzero().squeeze(-1).tolist()} succeeded at step {step}!")
            if success.all():
                break
        except Exception as e:
            log.debug(f"Checker error: {e}")
            pass

    saver_state.save()
    env.close()
    log.trace(f"State replay done in {time.time() - t0:.2f}s, replayed {total} states")


if __name__ == "__main__":
    main()
