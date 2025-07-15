import importlib
import os
import pathlib
import sys
import warnings
from functools import partial as bind

# def warn_with_traceback(
#       message, category, filename, lineno, file=None, line=None):
#   log = file if hasattr(file, 'write') else sys.stderr
#   import traceback
#   traceback.print_stack(file=log)
#   log.write(warnings.formatwarning(
#       message, category, filename, lineno, line))
# warnings.showwarning = warn_with_traceback

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers


def main(argv=None):
    embodied.print(r"---  ___                           __   ______ ---")
    embodied.print(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---")
    embodied.print(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---")
    embodied.print(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---")

    from . import agent as agt

    parsed, other = embodied.Flags(configs=["defaults"]).parse_known(argv)
    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    print("Run script:", args.script)

    logdir = embodied.Path(args.logdir)
    if args.script != "parallel_env":
        logdir.mkdirs()
        config.save(logdir / "config.yaml")

    embodied.timer.global_timer.enabled = args.timer

    if args.script == "train":
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    elif args.script == "train_eval":
        eval_config = config
        if "humanoid" in config.task:
            eval_config = eval_config.update({"env.humanoid.is_eval": True})
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "eval_replay", is_eval=True),
            bind(make_env, config),
            bind(make_env, eval_config),
            bind(make_logger, config),
            args,
        )

    elif args.script == "train_holdout":
        assert config.eval_dir
        embodied.run.train_holdout(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, config.eval_dir),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    elif args.script == "eval_only":
        embodied.run.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    elif args.script == "parallel":
        embodied.run.parallel(
            bind(make_agent, config),
            bind(make_replay, config, "replay", rate_limit=True),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    elif args.script == "env":
        envid = args.env_replica
        if envid < 0:
            envid = int(os.environ["JOB_COMPLETION_INDEX"])
        embodied.run.parallel_env(bind(make_env, config), envid, args)

    else:
        raise NotImplementedError(args.script)


def make_agent(config):
    from . import agent as agt

    env = make_env(config, 0)
    agent = agt.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent


def make_logger(config):
    step = embodied.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    outputs = [
        embodied.logger.TerminalOutput(config.filter, "Agent"),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"),
        # embodied.logger.TensorBoardOutput(
        #     logdir, config.run.log_video_fps, config.tensorboard_videos
        # ),
    ]
    if config.run.get("wandb", False):
        wandb_run_name = f"{config.task}.{config.method}.{config.seed}"
        outputs.append(embodied.logger.WandBOutput(logdir, wandb_run_name, config))
    logger = embodied.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
    directory = directory and embodied.Path(config.logdir) / directory
    size = config.replay_size // 10 if is_eval else config.replay_size
    kwargs = {}
    if rate_limit and config.run.train_ratio > 0:
        kwargs["samples_per_insert"] = config.run.train_ratio / config.batch_length
        kwargs["tolerance"] = 10 * config.batch_size
        kwargs["min_size"] = config.batch_size
    replay = embodied.replay.Replay(config.batch_length, size, directory, **kwargs)
    return replay


def make_env(config, index, **overrides):
    from embodied.envs import from_gym

    suite, task = config.task.split("_", 1)
    if suite == "procgen":
        import procgen  # noqa
    ctor = {
        "dummy": "embodied.envs.dummy:Dummy",
        "gym": "embodied.envs.from_gym:FromGym",
        "dm": "embodied.envs.from_dmenv:FromDM",
        "crafter": "embodied.envs.crafter:Crafter",
        "dmc": "embodied.envs.dmc:DMC",
        "atari": "embodied.envs.atari:Atari",
        "atari100k": "embodied.envs.atari:Atari",
        "dmlab": "embodied.envs.dmlab:DMLab",
        "minecraft": "embodied.envs.minecraft:Minecraft",
        "loconav": "embodied.envs.loconav:LocoNav",
        "pinpad": "embodied.envs.pinpad:PinPad",
        "langroom": "embodied.envs.langroom:LangRoom",
        "procgen": lambda task, **kw: from_gym.FromGym(f"procgen:procgen-{task}-v0", **kw),
        "humanoid": "embodied.envs.from_gymnasium:FromGymnasium",
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(":")
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.get("use_seed", False):
        kwargs["seed"] = hash((config.seed, index))
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif not space.discrete:
            env = wrappers.NormalizeAction(env, name)
            if args.discretize:
                env = wrappers.DiscretizeAction(env, name, args.discretize)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


if __name__ == "__main__":
    main()
