from contextlib import contextmanager
import time
from collections import defaultdict
import re
import sys

average_times = defaultdict(lambda: (0,0))

@contextmanager
def timeit(name):
    start = time.time()
    yield
    end = time.time()
    total_time, num_calls = average_times[name]
    total_time += end-start
    num_calls += 1
    print("TIME:", name, end-start, "| AVG", total_time / num_calls, f"| TOTAL {total_time} {num_calls}")
    average_times[name] = (total_time, num_calls)

def time_decorator(func):
    def with_times(*args, **kwargs):
        with timeit(func.__name__):
            return func(*args, **kwargs)
    return with_times


def recover_map(lines):
    info = {}
    pattern = re.compile(".* (.*) .* \| .* (.*\\b) .*\| .* (.*) (.*)")

    for l in lines:
        if not l.startswith("TIME"):
            continue
    
        match = pattern.match(l)

        name = match.group(1)
        avg = float(match.group(2))
        total_time = float(match.group(3))
        total_calls = float(match.group(4))
        info[name] = (avg, total_time, total_calls)
    
    return info

def compare_files(fileA, fileB):
    with open(fileA) as fA:
        linesA = fA.readlines()
    
    with open(fileB) as fB:
        linesB = fB.readlines()
    
    mapA = recover_map(linesA)
    mapB = recover_map(linesB)

    keysA = set(mapA.keys())
    keysB = set(mapB.keys())

    inter = keysA.intersection(keysB)
    print("Missing A", keysA.difference(inter))
    print("Missing B", keysB.difference(inter))

    keys_ordered = list(sorted([(mapA[k][1], k) for k in inter], reverse=True))

    for _, k in keys_ordered:
        print(f"{k} {mapA[k]} {mapB[k]}")


if __name__ == "__main__":
    fA = sys.argv[1]
    fB = sys.argv[2]
    compare_files(fA, fB)