import rootutils

rootutils.setup_root(__file__, pythonpath=True)
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.skillblender_rl.utils import (
    export_policy_as_jit,
    get_args,
    get_export_jit_path,
    get_load_path,
    get_log_dir,
    get_wrapper,
)


def play(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1
    scenario.task.commands.curriculum = False
    scenario.task.ppo_cfg.runner.resume = True
    scenario.task.random.friction.enabled = False
    scenario.task.random.mass.enabled = False
    scenario.task.random.push.enabled = False

    log_dir = get_log_dir(args, scenario)
    task_wrapper = get_wrapper(args.task)
    env = task_wrapper(scenario)

    load_path = get_load_path(args, scenario)

    obs = env.get_observations()
    # load policy
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
    )
    ppo_runner.load(load_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        export_jit_path = get_export_jit_path(args, scenario)
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_jit_path)
        print("Exported policy as jit script to: ", export_jit_path)

    for i in range(1000):
        # set fixed command
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)
