import re
from collections import defaultdict
from functools import partial as bind

import numpy as np

import embodied


def train_holdout(make_agent, make_train_replay, make_eval_replay, make_env, make_logger, args):
    agent = make_agent()
    train_replay = make_train_replay()
    eval_replay = make_eval_replay()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    epstats = embodied.Agg()

    batch_steps = args.batch_size * args.batch_length
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)

    @embodied.timer.section("log_step")
    def log_step(tran, worker):
        episode = episodes[worker]
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", tran["reward"], agg="stack")

        if tran["is_first"]:
            episode.reset()

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f"policy_{key}", tran[key], agg="stack")
        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg="sum")
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg="avg")
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg="max")

        if tran["is_last"]:
            result = episode.result()
            logger.add(
                {
                    "score": result.pop("score"),
                    "length": result.pop("length") - 1,
                },
                prefix="episode",
            )
            rew = result.pop("rewards")
            result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.num_envs)]
    driver = embodied.Driver(fns, args.driver_parallel)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(train_replay.add)
    driver.on_step(log_step)

    train_dataset = agent.dataset(embodied.Batch([train_replay.dataset] * args.batch_size))
    eval_dataset = agent.dataset(embodied.Batch([eval_replay.dataset] * args.batch_size))

    carry = [agent.init_train(args.batch_size)]

    def train_step(tran, worker):
        if len(train_replay) < args.batch_size or step < args.train_fill:
            return
        for _ in range(should_train(step)):
            with embodied.timer.section("dataset_next"):
                batch = next(train_dataset)
            outs, carry[0], mets = agent.train(batch, carry[0])
            agg.add(mets, prefix="train")

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

    print("Start training loop")
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)

        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix="epstats")
            if len(train_replay):
                logger.add(agent.report(next(train_dataset)), prefix="report")
            if len(eval_replay):
                logger.add(agent.report(next(eval_dataset)), prefix="eval")
            logger.add(embodied.timer.stats(), prefix="timer")
            logger.add(train_replay.stats(), prefix="replay")
            logger.add(usage.stats(), prefix="usage")
            logger.write(fps=True)

        if should_save(step):
            checkpoint.save()

    logger.close()
