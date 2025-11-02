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

    task: str = "close_box"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx", "isaacsim"] = (
        "isaacsim"
    )
    renderer: (
        Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3", "isaacsim"] | None
    ) = None

    num_envs: int = 1
    try_add_table: bool = True
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

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


def get_actions(all_actions, action_idx: int, num_envs: int, robot: RobotCfg):  # noqa: D103
    envs_actions = all_actions[:num_envs]
    return [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]


def get_states(all_states, action_idx: int, num_envs: int):  # noqa: D103
    envs_states = all_states[:num_envs]
    return [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]


def get_runout(all_actions, action_idx: int):  # noqa: D103
    return all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])


class ObsSaver:  # noqa: D101
    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []
        self.image_idx = 0

    def add(self, state: TensorState):  # noqa: D102
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

    def save(self):  # noqa: D102
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


def main():  # noqa: D103
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    env = task_cls(scenario, device=device)
    log.trace(f"Time to launch: {time.time() - t0:.2f}s")

    traj_filepath = env.traj_filepath
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    t0 = time.time()
    init_states, all_actions, _ = get_traj(traj_filepath, scenario.robots[0], env.handler)  # 仅用 actions
    log.trace(f"Time to load data: {time.time() - t0:.2f}s")

    os.makedirs("test_output", exist_ok=True)

    saver_act = ObsSaver(
        image_dir=_suffix_path(args.save_image_dir, "act") if args.save_image_dir else None,
        video_path=_suffix_path(args.save_video_path, "act") if args.save_video_path else None,
    )
    captured_states: list = []

    t0 = time.time()
    obs, _ = env.reset()
    saver_act.add(obs)

    captured_states.append(env.handler.get_states())

    step = 0
    while True:
        log.debug(f"[ACT] Step {step}")
        t1 = time.time()
        actions = get_actions(all_actions, step, num_envs, scenario.robots[0])
        obs, reward, success, time_out, extras = env.step(actions)
        log.trace(f"Time to step: {time.time() - t1:.2f}s")
        ee_states = get_ee_state(obs, robot_config=scenario.robots[0])
        log.info(f"EE state at step {step}: {ee_states}")
        saver_act.add(obs)

        try:
            captured_states.append(env.handler.get_states())
        except Exception:
            captured_states.append(env.handler.get_states())

        if success.any():
            log.info(f"[ACT] Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
        if time_out.any():
            log.info(f"[ACT] Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")
        if success.all() or time_out.all():
            break

        step += 1
        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("[ACT] Run out of actions, stopping")
            break

    saver_act.save()
    log.trace(f"Action replay done in {time.time() - t0:.2f}s, captured {len(captured_states)} state snapshots.")

    saver_state = ObsSaver(
        image_dir=_suffix_path(args.save_image_dir, "state") if args.save_image_dir else None,
        video_path=_suffix_path(args.save_video_path, "state") if args.save_video_path else None,
    )

    t0 = time.time()
    env.reset()
    total = len(captured_states)
    for i in range(total):
        log.debug(f"[STATE] Step {i}/{total - 1}")
        states_i = captured_states[i]
        env.handler.set_states(states_i)

        env.handler.refresh_render()
        obs = env.handler.get_states()
        saver_state.add(obs)

        try:
            success = env.checker.check(env.handler)
            if success.any():
                log.info(f"[STATE] Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break
        except Exception:
            pass

    saver_state.save()
    env.close()
    log.trace(f"State replay done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
