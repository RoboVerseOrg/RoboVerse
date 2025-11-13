from __future__ import annotations

import copy

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes
from roboverse_learn.rl.unitree_rl.helper import (get_args, make_objects, get_log_dir,
                                                  make_robots, set_seed, get_load_path,
                                                  PolicyExporterLSTM, export_policy_as_jit,
                                                  get_export_jit_path)
from roboverse_learn.rl.unitree_rl.runners import EnvWrapperTypes, MasterRunner

def prepare(args):
    task_cls = get_task_class(args.task)
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    overrides = {
        "num_envs": args.num_envs,
        "simulator": args.sim,
        "headless": args.headless,
    }

    if args.robots:
        overrides["robots"] = make_robots(args.robots)
        overrides["cameras"] = [
            camera
            for robot in overrides["robots"]
            if hasattr(robot, "cameras")
            for camera in getattr(robot, "cameras", [])
        ]

    if args.objects:
        overrides["objects"] = make_objects(args.objects)

    scenario.update(**overrides)

    device = "cpu" if args.sim == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    master_runner = MasterRunner(
        task_cls=task_cls,
        scenario=scenario,
        log_path=args.resume,
        lib_name="rsl_rl",
        device=device,
    )

    return master_runner

def play(args):
    master_runner = prepare(args)
    name_0 = list(master_runner.runners.keys())[0]
    if args.resume:
        if args.jit_load:
            log_dir = get_log_dir(task_name=master_runner.task_name, now=args.resume)
            policy_0 = torch.jit.load(get_load_path(load_root=log_dir, checkpoint=args.checkpoint))
        else:
            policys = master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
            policy_0 = policys[name_0]
    else:
        raise ValueError("Please provide the resume dir for eval policy.")

    runner_0 = master_runner.runners[name_0]
    env_0: EnvTypes = runner_0.env
    envwrapper_0: EnvWrapperTypes = runner_0.env_wrapper
    cfg_0 = env_0.cfg

    cfg_0.curriculum.enabled = False
    cfg_0.commands.resampling_time = 1e6  # effectively disable command changes

    # export jit policy
    export_jit_path = get_export_jit_path(get_log_dir(task_name=master_runner.task_name, now=args.resume), master_runner.scenario)
    actor_critic = runner_0.runner.alg.policy
    if hasattr(actor_critic, "memory_a"):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(export_jit_path)
    else:
        export_policy_as_jit(actor_critic.actor, export_jit_path)
    print("Exported policy as jit script to: ", export_jit_path)

    # unenable noise and randomization for eval

    env_0.reset()
    obs, _, _, _, _ = env_0.step(torch.zeros(env_0.num_envs, env_0.num_actions, device=env_0.device))
    obs = envwrapper_0.get_observations()


    for i in range(1000000):
        # set fixed command
        env_0.commands_manager.value[:, 0] = 0.5
        env_0.commands_manager.value[:, 1] = 0.0
        env_0.commands_manager.value[:, 2] = 0.0
        actions = policy_0(obs)
        obs, _, _, _ = envwrapper_0.step(actions)

def train(args):
    master_runner = prepare(args)
    if args.resume:
        master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
    master_runner.learn(max_iterations=args.iter)

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.eval:
        play(args)
    else:
        train(args)
