import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import numpy as np

import embodied

from . import chunk as chunklib
from . import limiters, selectors


class Replay:
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        chunksize=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        assert not capacity or min_size <= capacity

        self.length = length
        self.capacity = capacity
        self.directory = directory and embodied.Path(directory)
        self.chunksize = chunksize

        self.sampler = selectors.Uniform(seed)
        if samples_per_insert:
            self.limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            self.limiter = limiters.MinSize(min_size)

        self.chunks = {}
        self.refs = {}
        self.refs_lock = threading.RLock()

        self.items = {}
        self.fifo = deque()
        self.itemid = 0

        self.current = {}
        self.streams = defaultdict(deque)
        self.rwlock = embodied.RWLock()

        if self.directory:
            self.directory.mkdirs()
            self.workers = ThreadPoolExecutor(8, "replay_saver")
            self.promises = {}

        self.metrics = {
            "samples": 0,
            "sample_wait_dur": 0,
            "sample_wait_count": 0,
            "inserts": 0,
            "insert_wait_dur": 0,
            "insert_wait_count": 0,
        }

    def __len__(self):
        return len(self.items)

    def stats(self):
        ratio = lambda x, y: x / y if y else np.nan
        m = self.metrics
        chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
        stats = {
            "items": len(self.items),
            "chunks": len(self.chunks),
            "streams": len(self.streams),
            "ram_gb": chunk_nbytes / (1024**3),
            "inserts": m["inserts"],
            "samples": m["samples"],
            "replay_ratio": ratio(self.length * m["samples"], m["inserts"]),
            "insert_wait_avg": ratio(m["insert_wait_dur"], m["inserts"]),
            "insert_wait_frac": ratio(m["insert_wait_count"], m["inserts"]),
            "sample_wait_avg": ratio(m["sample_wait_dur"], m["samples"]),
            "sample_wait_frac": ratio(m["sample_wait_count"], m["samples"]),
        }
        for key in self.metrics:
            self.metrics[key] = 0
        return stats

    def clear(self):
        self.chunks.clear()
        self.refs.clear()
        self.items.clear()
        self.fifo.clear()
        self.itemid = 0
        self.current.clear()
        self.streams.clear()

    @embodied.timer.section("replay_add")
    def add(self, step, worker=0):
        with self.rwlock.reading:
            step = {k: v for k, v in step.items() if not k.startswith("log_")}
            step = {k: np.asarray(v) for k, v in step.items()}
            # step['id'] = np.asarray(embodied.uuid(step.get('id')))

            if worker not in self.current:
                chunk = chunklib.Chunk(self.chunksize)
                with self.refs_lock:
                    self.refs[chunk.uuid] = 1
                self.chunks[chunk.uuid] = chunk
                self.current[worker] = (chunk.uuid, 0)

            chunkid, index = self.current[worker]
            stream = self.streams[worker]
            chunk = self.chunks[chunkid]
            assert chunk.length == index, (chunk.length, index)
            chunk.append(step)
            assert chunk.length == index + 1, (chunk.length, index + 1)
            stream.append((chunkid, index))
            with self.refs_lock:
                self.refs[chunkid] += 1

            index += 1
            if index < chunk.size:
                self.current[worker] = (chunkid, index)
            else:
                self._complete(chunk, worker)
            assert len(self.streams) == len(self.current)

            if len(stream) >= self.length:
                dur = self._wait(self.limiter.want_insert, "Replay insert is waiting")
                # These increments are not thread safe, so the metrics will be slightly
                # wrong and it's faster than introducing a lock.
                self.metrics["inserts"] += 1
                self.metrics["insert_wait_dur"] += dur
                self.metrics["insert_wait_count"] += int(dur >= 0.001)
                chunkid, index = stream.popleft()
                self._insert(chunkid, index)

    @embodied.timer.section("replay_sample")
    def _sample(self):
        dur = self._wait(self.limiter.want_sample, "Replay sample is waiting")
        self.limiter.sample()
        # These increments are not thread safe, so the metrics will be slightly
        # wrong and it's faster than introducing a lock.
        self.metrics["samples"] += 1
        self.metrics["sample_wait_dur"] += dur
        self.metrics["sample_wait_count"] += int(dur >= 0.001)
        while True:
            with embodied.timer.section("draw"):
                itemid = self.sampler()
            with embodied.timer.section("lookup"):
                # Look up the item or repeat if it was already removed in the meantime.
                try:
                    chunkid, index = self.items[itemid]
                    chunk = self.chunks[chunkid]
                    break
                except KeyError:
                    continue
        available = chunk.length - index
        if available >= self.length:
            with embodied.timer.section("slice"):
                seq = chunk.slice(index, self.length)
        else:
            with embodied.timer.section("compose"):
                parts = [chunk.slice(index, available)]
                remaining = self.length - available
                while remaining > 0:
                    chunk = self.chunks[chunk.succ]
                    take = min(remaining, chunk.length)
                    parts.append(chunk.slice(0, take))
                    remaining -= take
                seq = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0].keys()}
        with embodied.timer.section("isfirst"):
            if "is_first" in seq:
                seq["is_first"] = seq["is_first"].copy()
                seq["is_first"][0] = True
        return seq

    def _insert(self, chunkid, index):
        itemid = self.itemid
        self.itemid += 1
        self.items[itemid] = (chunkid, index)
        self.sampler[itemid] = (chunkid, index)
        self.fifo.append(itemid)
        self.limiter.insert()
        while self.capacity and len(self.items) > self.capacity:
            self._remove()

    def _remove(self):
        self.limiter.remove()
        itemid = self.fifo.popleft()
        del self.sampler[itemid]
        chunkid, index = self.items.pop(itemid)
        with self.refs_lock:
            self.refs[chunkid] -= 1
            if self.refs[chunkid] < 1:
                del self.refs[chunkid]
                chunk = self.chunks.pop(chunkid)
                if chunk.succ in self.refs:
                    self.refs[chunk.succ] -= 1

    def dataset(self, batch=None):
        if batch:
            while True:
                seqs = []
                for _ in range(batch):
                    seqs.append(self._sample())
                stacked = {k: np.stack([seq[k] for seq in seqs]) for k in seqs[0].keys()}
                yield stacked
        else:
            while True:
                yield self._sample()

    @embodied.timer.section("replay_save")
    def save(self, wait=False):
        if not self.directory:
            return
        with self.rwlock.writing:
            [x.result() for x in list(self.promises.values())]
            for worker, (chunkid, _) in self.current.items():
                chunk = self.chunks[chunkid]
                if chunk.length > 0:
                    self._complete(chunk, worker)
            wait and [x.result() for x in list(self.promises.values())]

    @embodied.timer.section("replay_load")
    def load(self, data=None, directory=None, amount=None):
        assert data is None
        directory = directory or self.directory
        amount = amount or self.capacity or np.inf
        if not directory:
            return
        revsorted = lambda x: list(reversed(sorted(list(x))))
        directory = embodied.Path(directory)
        names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
        names_ondisk = revsorted(x.name for x in directory.glob("*.npz"))
        names_ondisk = [x for x in names_ondisk if x not in names_loaded]
        if not names_ondisk:
            return

        numitems = self._numitems(names_loaded + names_ondisk)
        uuids = [embodied.uuid(x.split("-")[1]) for x in names_ondisk]
        total = 0
        numchunks = 0
        for uuid in uuids:
            numchunks += 1
            total += numitems[uuid]
            if total >= amount:
                break

        load = bind(chunklib.Chunk.load, error="none")
        filenames = [directory / x for x in names_ondisk[:numchunks]]

        with ThreadPoolExecutor(16, "replay_loader") as pool:
            chunks = [x for x in pool.map(load, filenames) if x]

        # We need to recompute the number of items per chunk now because some
        # chunks be corrupted and thus not available.
        # numitems = self._numitems(chunks + list(self.chunks.values()))
        numitems = self._numitems(chunks)

        with self.rwlock.writing:
            with self.refs_lock:
                for chunk in chunks:
                    self.chunks[chunk.uuid] = chunk
                    self.refs[chunk.uuid] = 0
                for chunk in reversed(chunks):
                    amount = numitems[chunk.uuid]
                    self.refs[chunk.uuid] += amount
                    if chunk.succ in self.refs:
                        self.refs[chunk.succ] += 1
                    for index in range(amount):
                        self._insert(chunk.uuid, index)

    @embodied.timer.section("complete_chunk")
    def _complete(self, chunk, worker):
        succ = chunklib.Chunk(self.chunksize)
        with self.refs_lock:
            self.refs[chunk.uuid] -= 1
            self.refs[succ.uuid] = 2
        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid
        if self.directory:
            (worker in self.promises) and self.promises.pop(worker).result()
            self.promises[worker] = self.workers.submit(chunk.save, self.directory)
        return succ

    def _numitems(self, chunks):
        chunks = [x.filename if hasattr(x, "filename") else x for x in chunks]
        chunks = list(reversed(sorted([embodied.Path(x).stem for x in chunks])))
        times, uuids, succs, lengths = zip(*[x.split("-") for x in chunks])
        uuids = [embodied.uuid(x) for x in uuids]
        succs = [embodied.uuid(x) for x in succs]
        lengths = {k: int(v) for k, v in zip(uuids, lengths)}
        future = {}
        for uuid, succ in zip(uuids, succs):
            future[uuid] = lengths[uuid] + future.get(succ, 0)
        numitems = {}
        for uuid, succ in zip(uuids, succs):
            numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)
        numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
        return numitems

    def _wait(self, predicate, message, sleep=0.01, notify=1.0):
        # Early exit if rate is not currently limited.
        if predicate(reason=False):
            return 0
        # Wait without messages to avoid creating string objects.
        start = time.time()
        duration = 0
        while duration < notify:
            duration = time.time() - start
            if predicate(reason=False):
                return duration
            time.sleep(sleep)
        # Get the reason once to provide a useful user message.
        allowed, reason = predicate(reason=True)
        if allowed:
            return time.time() - start
        else:
            print(f"{message} ({reason})")
            time.sleep(sleep)
        # Keep waiting without messages to avoid creating string objects.
        while not predicate(reason=False):
            time.sleep(sleep)
        return time.time() - start
