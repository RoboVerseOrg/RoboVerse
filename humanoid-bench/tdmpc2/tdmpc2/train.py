import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import hydra
import torch
from termcolor import colored

from tdmpc2.common.buffer import Buffer
from tdmpc2.common.logger import Logger
from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.trainer.offline_trainer import OfflineTrainer
from tdmpc2.trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
            `task`: task name (or mt30/mt80 for multi-task training)
            `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
            `steps`: number of training/environment steps (default: 10M)
            `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
            $ python train.py task=mt80 model_size=48
            $ python train.py task=mt30 model_size=317
            $ python train.py task=dog-run steps=7000000
    ```
    """
    # assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
