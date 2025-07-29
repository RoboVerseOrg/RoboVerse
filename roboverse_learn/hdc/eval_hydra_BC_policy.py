"""
Evaluate one BC checkpoint with HDCWrapper + MetaSim.
"""

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import csv
import os
import pathlib

import hydra
import numpy as np
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf
from roboverse_learn.hdc.hdc_wrapper import HDCWrapper  # noqa: F401
from roboverse_learn.rsl_rl.rsl_rl.runners.eval_runner_BC_modified import EvalRunnerBCModified

from metasim.cfg.scenario import ScenarioCfg

from metasim.cfg.control import ControlCfg


# ---------- helpers -------------------------------------------------
def to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    return obj


def load_BC_workspace(train_cfg, checkpoint_path):
    """Instantiate workspace and load .ckpt."""
    OmegaConf.resolve(train_cfg)
    cls = hydra.utils.get_class(train_cfg._target_)
    ws = cls(train_cfg)
    ck = pathlib.Path(checkpoint_path)
    if not ck.is_file():
        raise FileNotFoundError(ck)
    print(f"[loader] restore {ck}")
    ws.load_checkpoint(path=ck)
    return ws


# ---------- main ----------------------------------------------------
@hydra.main(version_base=None, config_path="./cfg", config_name="config_base")
def play(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_humanoid_workspace = cfg.humanoid_workspace
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))


    control: ControlCfg = ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)
    # scenario & env
    scenario = ScenarioCfg(
        task="hdc:eval",
        robots=["h1_verse"],
        sim="isaacgym",
        num_envs=1,
        headless=False,
        objects=[],
        cameras=[],
    )

    scenario.control = control


    env = HDCWrapper(cfg, scenario)

    # tweak env cfg for eval
    env_cfg, train_cfg = cfg, cfg.train
    env_cfg.env.num_envs = 1
    env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    env_cfg.terrain.curriculum = False
    env_cfg.add_eval_noise = False
    env_cfg.env.episode_length_s = 20
    env_cfg.env.test = True




    # load policy
    train_cfg.runner.resume = True
    ws = load_BC_workspace(cfg_humanoid_workspace, cfg.BC_ckpt_path)
    policy = ws.model.to(env.device)

    # run evaluation
    To = cfg.humanoid_workspace.n_obs_steps
    runner = EvalRunnerBCModified(
        env=env,
        policy=policy,
        train_cfg=train_cfg,
        device=env.device,
        To=To,
        clip_action=True,
    )
    results = to_py(runner.eval())

    # write csv
    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)
    motion_stub = pathlib.Path(cfg.motion.motion_file).stem
    ckpt_dir = pathlib.Path(cfg.BC_ckpt_path).parent.name
    csv_path = f"{out_dir}/{motion_stub}_{ckpt_dir}.csv"

    header_needed = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as fp:
        w = csv.writer(fp)
        if header_needed:
            w.writerow(["ckpt"] + list(results.keys()))
        w.writerow([pathlib.Path(cfg.BC_ckpt_path).stem] + list(results.values()))
    print(f"✓ Results appended to {csv_path}")


if __name__ == "__main__":
    play()
