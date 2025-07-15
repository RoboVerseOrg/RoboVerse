import pathlib
import sys
from collections import deque
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))

import embodied
import numpy as np
import pytest
import utils

PORTS = iter(range(7000, 8000))


class TestParallel:
    @pytest.mark.parametrize("train_ratio", (-1, 1, 128))
    def test_run_loop(self, tmpdir, train_ratio):
        addr = "ipc:///tmp/stats"
        received = deque(maxlen=1)
        server = embodied.distr.Server(addr)
        server.bind("report", lambda stats: received.append(stats))
        server.start()

        args = self._make_args(tmpdir, train_ratio)
        embodied.run.parallel(
            bind(self._make_agent, addr),
            bind(self._make_replay, args),
            self._make_env,
            self._make_logger,
            args,
        )
        stats = received[0]
        print("Stats:", stats)
        assert stats["lifetime"] > 7
        assert stats["env_steps"] > 1000
        if args.train_ratio > -1:
            replay_steps = stats["env_steps"] * args.train_ratio
            assert np.allclose(stats["replay_steps"], replay_steps, 100, 0.1)
        else:
            assert stats["replay_steps"] > 100
        assert stats["reports"] >= 1
        assert stats["saves"] >= 2
        assert stats["loads"] == 0

        args = self._make_args(tmpdir, train_ratio)
        embodied.run.parallel(
            bind(self._make_agent, addr),
            bind(self._make_replay, args),
            self._make_env,
            self._make_logger,
            args,
        )
        stats = received[0]
        assert stats["loads"] == 1

    def _make_agent(self, queue):
        env = self._make_env(0)
        agent = utils.TestAgent(env.obs_space, env.act_space, queue)
        env.close()
        return agent

    def _make_env(self, index):
        from embodied.envs import dummy

        return dummy.Dummy("disc", size=(64, 64), length=100)

    def _make_replay(self, args):
        kwargs = {"length": args.batch_length, "capacity": 1e4}
        if args.train_ratio > -1:
            kwargs["samples_per_insert"] = args.train_ratio / args.batch_length
        return embodied.replay.Replay(**kwargs)

    def _make_logger(self):
        return embodied.Logger(
            embodied.Counter(),
            [
                embodied.logger.TerminalOutput(),
            ],
        )

    def _make_args(self, logdir, train_ratio):
        return embodied.Config(
            logdir=str(logdir),
            num_envs=4,
            duration=10,
            log_every=3,
            save_every=5,
            train_ratio=float(train_ratio),
            train_fill=100,
            batch_size=8,
            batch_length=16,
            expl_until=0,
            from_checkpoint="",
            usage=dict(psutil=True, nvsmi=False),
            log_zeros=False,
            log_video_streams=4,
            log_video_fps=20,
            log_keys_video=["image"],
            log_keys_sum="^$",
            log_keys_avg="^$",
            log_keys_max="^$",
            log_episode_timeout=60.0,
            actor_addr=f"tcp://localhost:{next(PORTS)}",
            replay_addr=f"ipc:///tmp/replay{next(PORTS)}",
            logger_addr=f"ipc:///tmp/logger{next(PORTS)}",
            actor_batch=2,
            actor_threads=4,
            env_replica=-1,
            ipv6=False,
            timer=True,
        )
