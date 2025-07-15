import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tdmpc2.common import MODEL_SIZE, TASK_SET


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """Parses a Hydra config. Mostly for convenience."""
    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # Convenience
    cfg.work_dir = Path(hydra.utils.get_original_cwd()) / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)  # Bin size for discrete regression

    # Model size
    if cfg.get("model_size", None) is not None:
        assert cfg.model_size in MODEL_SIZE.keys(), (
            f"Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}"
        )
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        if cfg.task == "mt30" and cfg.model_size == 19:
            cfg.latent_dim = 512  # This checkpoint is slightly smaller

    # Multi-task
    cfg.multitask = cfg.task in TASK_SET.keys()
    if cfg.multitask:
        cfg.task_title = cfg.task.upper()
        # Account for slight inconsistency in task_dim for the mt30 experiments
        cfg.task_dim = 96 if cfg.task == "mt80" or cfg.model_size in {1, 317} else 64
    else:
        cfg.task_dim = 0
    cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

    return cfg
