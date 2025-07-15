import re
import sys
import threading
import time
from collections import defaultdict, deque

import cloudpickle
import numpy as np

import embodied

prefix = lambda d, p: {f"{p}/{k}": v for k, v in d.items()}


def parallel(make_agent, make_replay, make_env, make_logger, args):
    if args.num_envs:
        assert args.actor_batch <= args.num_envs, (args.actor_batch, args.num_envs)
    for option in ("actor_addr", "replay_addr", "logger_addr"):
        random = str(np.random.randint(1025, 65535))
        args = args.update({option: args[option].format(random=random)})

    make_env = cloudpickle.dumps(make_env)
    make_agent = cloudpickle.dumps(make_agent)
    make_replay = cloudpickle.dumps(make_replay)
    make_logger = cloudpickle.dumps(make_logger)

    workers = [embodied.distr.Process(parallel_env, make_env, i, args, True) for i in range(args.num_envs)]
    workers.append(embodied.distr.Process(parallel_agent, make_agent, args))
    workers.append(embodied.distr.Process(parallel_replay, make_replay, args))
    workers.append(embodied.distr.Process(parallel_logger, make_logger, args))
    embodied.distr.run(workers, args.duration)


def parallel_agent(make_agent, args):
    make_agent = cloudpickle.loads(make_agent)
    barrier = threading.Barrier(2)
    agent = make_agent()
    workers = []
    workers.append(embodied.distr.Thread(parallel_actor, agent, barrier, args))
    workers.append(embodied.distr.Thread(parallel_learner, agent, barrier, args))
    embodied.distr.run(workers, args.duration)


def parallel_actor(agent, barrier, args):
    initial = agent.init_policy(args.actor_batch)
    initial = embodied.treemap(lambda x: x[0], initial)
    allstates = defaultdict(lambda: initial)
    barrier.wait()  # Do not collect data before learner restored checkpoint.
    fps = embodied.FPS()

    should_log = embodied.when.Clock(args.log_every)
    logger = embodied.distr.Client(
        args.logger_addr,
        name="ActorLogger",
        connect=True,
        maxinflight=8 * args.actor_threads,
    )
    replay = embodied.distr.Client(
        args.replay_addr,
        name="ActorReplay",
        connect=True,
        maxinflight=8 * args.actor_threads,
    )

    @embodied.timer.section("actor_workfn")
    def workfn(obs):
        envids = obs.pop("envid")
        fps.step(obs["is_first"].size)
        with embodied.timer.section("get_states"):
            states = [allstates[a] for a in envids]
            states = embodied.treemap(lambda *xs: list(xs), *states)
        act, states = agent.policy(obs, states)
        act["reset"] = obs["is_last"].copy()
        with embodied.timer.section("put_states"):
            for i, a in enumerate(envids):
                allstates[a] = embodied.treemap(lambda x: x[i], states)
        trans = {"envids": envids, **obs, **act}
        [x.setflags(write=False) for x in trans.values()]
        return act, trans

    @embodied.timer.section("actor_donefn")
    def donefn(trans):
        replay.add_batch(trans)
        logger.trans(trans)
        if should_log():
            stats = {}
            stats["parallel/fps_actor"] = fps.result()
            stats["parallel/ep_states"] = len(allstates)
            stats.update(prefix(server.stats(), "server/actor"))
            stats.update(prefix(logger.stats(), "client/actor_logger"))
            stats.update(prefix(replay.stats(), "client/actor_replay"))
            logger.add(stats)

    server = embodied.distr.ProcServer(args.actor_addr, ipv6=args.ipv6)
    server.bind("act", workfn, donefn, args.actor_threads, args.actor_batch)
    server.run()


def parallel_learner(agent, barrier, args):
    logdir = embodied.Path(args.logdir)
    agg = embodied.Agg()
    usage = embodied.Usage(**args.usage)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    fps = embodied.FPS()

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    logger = embodied.distr.Client(args.logger_addr, name="LearnerLogger", maxinflight=1, connect=True)
    barrier.wait()
    should_save()  # Register that we just saved.

    replays = []

    def parallel_dataset(prefetch=1):
        replay = embodied.distr.Client(args.replay_addr, name=f"LearnerReplay{len(replays)}", connect=True)
        replays.append(replay)
        futures = deque([replay.sample_batch({}) for _ in range(prefetch)])
        while True:
            futures.append(replay.sample_batch({}))
            yield futures.popleft().result()

    dataset = agent.dataset(parallel_dataset)
    state = agent.init_train(args.batch_size)

    while True:
        with embodied.timer.section("learner_batch_next"):
            batch = next(dataset)
        with embodied.timer.section("learner_train_step"):
            outs, state, mets = agent.train(batch, state)
        time.sleep(0.0001)
        agg.add(mets)
        fps.step(batch["is_first"].size)

        if should_log():
            with embodied.timer.section("learner_metrics"):
                stats = {}
                stats.update(prefix(agg.result(), "train"))
                stats.update(prefix(agent.report(batch), "report"))
                stats.update(prefix(embodied.timer.stats(), "timer/agent"))
                stats.update(prefix(usage.stats(), "usage/agent"))
                stats.update({"parallel/fps_learner": fps.result()})
                stats.update(prefix(logger.stats(), "client/learner_logger"))
                stats.update(prefix(replays[0].stats(), "client/learner_replay0"))
            logger.add(stats)

        if should_save():
            checkpoint.save()


def parallel_replay(make_replay, args):
    make_replay = cloudpickle.loads(make_replay)

    replay = make_replay()
    dataset = iter(replay.dataset(args.batch_size))

    should_log = embodied.when.Clock(args.log_every)
    logger = embodied.distr.Client(args.logger_addr, name="ReplayLogger", connect=True, maxinflight=1)
    usage = embodied.Usage(**args.usage.update(nvsmi=False))

    should_save = embodied.when.Clock(args.save_every)
    cp = embodied.Checkpoint(embodied.Path(args.logdir) / "replay.ckpt")
    cp.replay = replay
    cp.load_or_save()

    def add_batch(data):
        for i, envid in enumerate(data.pop("envids")):
            replay.add({k: v[i] for k, v in data.items()}, envid)
        return {}

    server = embodied.distr.Server(args.replay_addr, name="Replay")
    server.bind("add_batch", add_batch, workers=1)
    server.bind("sample_batch", lambda _: next(dataset), workers=1)
    with server:
        while True:
            server.check()
            should_save() and cp.save()
            time.sleep(1)
            if should_log():
                stats = prefix(replay.stats(), "replay")
                stats.update(prefix(embodied.timer.stats(), "timer/replay"))
                stats.update(prefix(usage.stats(), "usage/replay"))
                stats.update(prefix(logger.stats(), "client/replay_logger"))
                stats.update(prefix(server.stats(), "server/replay"))
                logger.add(stats)


def parallel_logger(make_logger, args):
    make_logger = cloudpickle.loads(make_logger)

    logger = make_logger()
    should_log = embodied.when.Clock(args.log_every)
    usage = embodied.Usage(**args.usage.update(nvsmi=False))

    should_save = embodied.when.Clock(args.save_every)
    cp = embodied.Checkpoint(embodied.Path(args.logdir) / "logger.ckpt")
    cp.step = logger.step
    cp.load_or_save()

    parallel = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    updated = defaultdict(lambda: None)
    dones = defaultdict(lambda: True)

    log_keys_max = re.compile(args.log_keys_max)
    log_keys_sum = re.compile(args.log_keys_sum)
    log_keys_avg = re.compile(args.log_keys_avg)

    @embodied.timer.section("logger_addfn")
    def addfn(metrics):
        logger.add(metrics)

    @embodied.timer.section("logger_transfn")
    def transfn(trans):
        now = time.time()
        envids = trans.pop("envids")
        logger.step.increment(len(trans["is_first"]))
        parallel.add("ep_starts", trans["is_first"].sum(), agg="sum")
        parallel.add("ep_ends", trans["is_last"].sum(), agg="sum")

        for i, addr in enumerate(envids):
            tran = {k: v[i] for k, v in trans.items()}

            updated[addr] = now
            episode = episodes[addr]
            if tran["is_first"]:
                episode.reset()
                parallel.add("ep_abandoned", int(not dones[addr]), agg="sum")
            dones[addr] = tran["is_last"]

            episode.add("score", tran["reward"], agg="sum")
            episode.add("length", 1, agg="sum")
            episode.add("rewards", tran["reward"], agg="stack")

            video_addrs = list(episodes.keys())[: args.log_video_streams]
            if addr in video_addrs:
                for key in args.log_keys_video:
                    if key in tran:
                        episode.add(f"policy_{key}", tran[key], agg="stack")

            for key in trans.keys():
                if log_keys_max.match(key):
                    episode.add(key, tran[key], agg="max")
                if log_keys_sum.match(key):
                    episode.add(key, tran[key], agg="sum")
                if log_keys_avg.match(key):
                    episode.add(key, tran[key], agg="avg")

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

        for addr, last in list(updated.items()):
            if now - last >= args.log_episode_timeout:
                print("Dropping episode statistics due to timeout.")
                del episodes[addr]
                del updated[addr]

    server = embodied.distr.Server(args.logger_addr, name="Logger")
    server.bind("add", addfn)
    server.bind("trans", transfn)
    with server:
        while True:
            server.check()
            should_save() and cp.save()
            time.sleep(1)
            if should_log():
                with embodied.timer.section("logger_metrics"):
                    logger.add(parallel.result(), prefix="parallel")
                    logger.add(epstats.result(), prefix="epstats")
                    logger.add(embodied.timer.stats(), prefix="timer/logger")
                    logger.add(usage.stats(), prefix="usage/logger")
                    logger.add(server.stats(), prefix="server/logger")
                logger.write()


def parallel_env(make_env, envid, args, logging=False):
    make_env = cloudpickle.loads(make_env)
    assert envid >= 0, envid
    name = f"Env{envid}"

    _print = lambda x: embodied.print(f"[{name}] {x}")
    should_log = embodied.when.Clock(args.log_every)
    if logging:
        logger = embodied.distr.Client(args.logger_addr, name=f"{name}Logger", connect=True, maxinflight=1)
    fps = embodied.FPS()
    if envid == 0:
        usage = embodied.Usage(**args.usage.update(nvsmi=False))

    _print("Make env")
    env = make_env(envid)
    actor = embodied.distr.Client(args.actor_addr, envid, name, args.ipv6, pings=10, maxage=60, connect=True)

    done = True
    while True:
        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        with embodied.timer.section("env_step"):
            obs = env.step(act)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        score += obs["reward"]
        length += 1
        fps.step(1)
        done = obs["is_last"]
        if done:
            _print(f"Episode of length {length} with score {score:.4f}")

        with embodied.timer.section("env_request"):
            future = actor.act({"envid": envid, **obs})
        try:
            with embodied.timer.section("env_response"):
                act = future.result()
            act = {k: v for k, v in act.items() if not k.startswith("log_")}
        except embodied.distr.NotAliveError:
            # Wait until we are connected again, so we don't unnecessarily reset the
            # environment hundreds of times while the server is unavailable.
            _print("Lost connection to server")
            actor.connect()
            done = True
        except embodied.distr.RemoteError as e:
            _print(f"Shutting down env due to agent error: {e}")
            sys.exit(0)

        if should_log() and logging and envid == 0:
            stats = {f"parallel/fps_env{envid}": fps.result()}
            stats.update(prefix(usage.stats(), f"usage/env{envid}"))
            stats.update(prefix(logger.stats(), f"client/env{envid}_logger"))
            stats.update(prefix(actor.stats(), f"client/env{envid}_actor"))
            stats.update(prefix(embodied.timer.stats(), f"timer/env{envid}"))
            logger.add(stats)
