from collections import defaultdict, deque

import numpy as np

from . import sampletree


class Fifo:
    def __init__(self):
        self.queue = deque()

    def __call__(self):
        return self.queue[0]

    def __setitem__(self, key, steps):
        self.queue.append(key)

    def __delitem__(self, key):
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # This is very slow but typically not used.
            self.queue.remove(key)


class Uniform:
    def __init__(self, seed=0):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        index = self.rng.integers(0, len(self.keys)).item()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class Prioritized:
    def __init__(self, exponent=1.0, initial=1.0, zero_on_sample=False, branching=16, seed=0):
        self.exponent = exponent
        self.initial = initial
        self.zero_on_sample = zero_on_sample
        self.tree = sampletree.SampleTree(branching, seed)
        self.prios = defaultdict(lambda: self.initial)
        self.stepitems = defaultdict(list)
        self.items = {}

    def prioritize(self, stepids, priorities):
        if not isinstance(stepids[0], bytes):
            stepids = [x.tobytes() for x in stepids]
        for stepid, priority in zip(stepids, priorities):
            self.prios[stepid] = priority
        items = []
        for stepid in stepids:
            items += self.stepitems[stepid]
        for key in list(set(items)):
            self.tree.update(key, self._aggregate(key))

    def __call__(self):
        key = self.tree.sample()
        if self.zero_on_sample:
            zeros = [0.0] * len(self.items[key])
            self.prioritize(self.items[key], zeros)
        return key

    def __setitem__(self, key, steps):
        stepids = [x["id"].tobytes() for x in steps]
        self.items[key] = stepids
        [self.stepitems[stepid].append(key) for stepid in stepids]
        self.tree.insert(key, self._aggregate(key))

    def __delitem__(self, key):
        self.tree.remove(key)
        stepids = self.items.pop(key)
        for stepid in stepids:
            stepitems = self.stepitems[stepid]
            stepitems.remove(key)
            if not stepitems:
                del self.stepitems[stepid]
                del self.prios[stepid]

    def _aggregate(self, key):
        # Both list comprehensions in this function are a performance bottleneck
        # because they are called very often.
        prios = [self.prios[stepid] for stepid in self.items[key]]
        if self.exponent != 1.0:
            prios = [x**self.exponent for x in prios]
        return sum(prios)
