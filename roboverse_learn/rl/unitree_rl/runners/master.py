from __future__ import annotations
from typing import Type

import copy
import datetime
import os
import pickle as pkl
import shutil
import sys

import torch

from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.helper import get_class, get_log_dir, get_load_path
from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes


class BaseRunnerWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict, log_dir: str):
        self.env = env
        self.device = env.device
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict()
        self.train_cfg = train_cfg
        self.log_dir = log_dir

    def load(self, path):
        raise NotImplementedError

    def learn(self, max_iterations):
        raise NotImplementedError

    def get_policy(self):
        raise NotImplementedError


class MasterRunner:
    def __init__(
        self,
        task_cls: Type[EnvTypes],
        scenario: ScenarioCfg,
        log_path: str | None = None,
        lib_name: str = "rsl_rl",
        device: str | torch.device | None = None,
    ):
        self.task_cls = task_cls
        self.task_name = getattr(task_cls, "task_name", task_cls.__name__)
        self.runners = {}
        self.envs = {}
        self.scenario = scenario

        env_cfg_cls: Type[BaseEnvCfg] = getattr(task_cls, "env_cfg_cls", BaseEnvCfg)
        train_cfg_cls = getattr(task_cls, "train_cfg_cls", None)
        runner_cls = get_class(lib_name, suffix="Wrapper", library="roboverse_learn.rl.unitree_rl.runners")

        module = sys.modules[task_cls.__module__]
        env_cls_path = getattr(module, "__file__", None)

        now = log_path if log_path else datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")

        robot_cfgs = scenario.robots if isinstance(scenario.robots, list) else [scenario.robots]
        for robot in robot_cfgs:
            scenario_copy = copy.deepcopy(scenario)
            scenario_copy.robots = [robot]
            scenario_copy.__post_init__()

            resolved_device = device
            if resolved_device is None:
                resolved_device = (
                    "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")
                )

            env_cfg = env_cfg_cls()
            env: EnvTypes = task_cls(
                scenario=scenario_copy,
                device=resolved_device,
                env_cfg=env_cfg,
            )

            train_cfg = train_cfg_cls() if callable(train_cfg_cls) else train_cfg_cls

            log_dir = get_log_dir(task_name=self.task_name, now=now)
            runner: BaseRunnerWrapper = runner_cls(env=env, train_cfg=train_cfg, log_dir=log_dir)
            self.runners[env.robot.name] = runner
            self.envs[env.robot.name] = env

            if not log_path:
                params_path = f"{log_dir}/params"
                if not os.path.exists(params_path):
                    os.makedirs(params_path, exist_ok=True)
                if env_cls_path:
                    shutil.copy2(env_cls_path, params_path)
                pkl.dump(env_cfg, open(f"{params_path}/env_cfg.pkl", "wb"))
                pkl.dump(train_cfg, open(f"{params_path}/train_cfg.pkl", "wb"))

    def learn(self, max_iterations=10000):
        if not self.runners:
            raise RuntimeError("No runners instantiated for training.")
        first_runner = next(iter(self.runners.values()))
        first_runner.learn(max_iterations=max_iterations)

    def load(self, resume_dir: str, checkpoint: int = None):
        self.policys = {}
        for _robot_name, _runner in self.runners.items():
            log_dir = get_log_dir(task_name=self.task_name, now=resume_dir)
            _runner.load(get_load_path(load_root=log_dir, checkpoint=checkpoint))
            self.policys[_robot_name] = _runner.get_policy()
        return self.policys
