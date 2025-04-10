from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import tyro
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)
from metasim.types import EnvState


@dataclass
class Args:
    task: str
    robot: str
    sim: str

    save_dir: str


args = tyro.cli(Args)


def save_obs(states: list[EnvState], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    from torchvision.utils import make_grid, save_image

    try:
        rgb_data = next(iter(states[0]["cameras"].values()))["rgb"]
    except Exception as e:
        log.error(f"{e=}")
        return

    if rgb_data.shape != (256, 256, 3):
        log.error(f"{type(rgb_data)=}")
        log.error(f"{rgb_data.shape=}")
        return
    rgb_data = rgb_data.unsqueeze(0)
    image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_image(image, filepath)


def main():
    if args.sim == "isaacgym":
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.cfg.sensors import PinholeCameraCfg
    from metasim.constants import SimType
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_sim_env_class

    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb", "depth"],
        width=256,
        height=256,
        pos=(1.5, -1.5, 1.5),  # near
        look_at=(0.0, 0.0, 0.0),
    )

    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        cameras=[camera],
        sim=args.sim,
        headless=True,
    )
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robot, env.handler)
    obs, _ = env.reset(states=init_states[: scenario.num_envs])
    save_obs(obs, f"{args.save_dir}/rgb_{0:04d}.png")
    for i in range(1):
        obs, _, _, _, _ = env.step([all_actions[0][0]])
        save_obs(obs, f"{args.save_dir}/rgb_{i + 1:04d}.png")
    env.close()


if __name__ == "__main__":
    main()
