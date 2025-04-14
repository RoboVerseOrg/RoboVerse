from __future__ import annotations

import os
import time

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def set_np_formatting():
    """Formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    sim_name = cfg.environment.sim_name.lower()
    set_np_formatting()
    if sim_name == "isaacgym":
        import isaacgym  # noqa: F401

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.utils.setup_util import SimType, get_robot, get_sim_env_class, get_task
    from roboverse_learn.rl.algos import get_algorithm
    from roboverse_learn.rl.env import RLEnvWrapper

    cprint("Start Building the Environment", "green", attrs=["bold"])
    task = get_task(cfg.train.task_name)
    robot = get_robot(cfg.train.robot_name)
    scenario = ScenarioCfg(task=task, robot=robot)
    scenario.cameras = []

    tic = time.time()
    scenario.num_envs = cfg.environment.num_envs
    scenario.headless = cfg.environment.headless

    env_class = get_sim_env_class(SimType(sim_name))
    if sim_name == "mujoco":
        from metasim.sim import GymEnvWrapper
        from metasim.sim.mujoco import MujocoHandler
        from metasim.sim.parallel import ParallelSimWrapper

        env_class = GymEnvWrapper(ParallelSimWrapper(MujocoHandler))
    env = env_class(scenario)

    env = RLEnvWrapper(
        gym_env=env,
        seed=cfg.experiment.seed,
        verbose=False,
    )

    if hasattr(env, "set_seed"):
        env.set_seed(cfg.experiment.seed)

    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    output_dif = os.path.join("outputs", cfg.experiment.output_name)
    os.makedirs(output_dif, exist_ok=True)

    algo_name = cfg.train.algo.lower()
    agent = get_algorithm(algo_name, env=env, output_dif=output_dif, full_config=OmegaConf.to_container(cfg))

    log.info(f"Algorithm: {cfg.train.algo}")
    log.info(f"Number of environments: {cfg.environment.num_envs}")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment.output_name,
        sync_tensorboard=True,
        mode=cfg.wandb.mode,
        settings=wandb.Settings(
            _disable_stats=True,  # Reduce overhead
            _disable_meta=True,  # Reduce overhead
        ),
    )

    if cfg.experiment.resume_training:
        agent.load(cfg.experiment.checkpoint)
    agent.train()
    wandb.finish()


if __name__ == "__main__":
    main()
