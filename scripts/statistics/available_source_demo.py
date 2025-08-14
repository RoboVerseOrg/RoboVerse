import glob
import gzip
import json
import pickle

import yaml


def load_traj_file(path: str):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".yaml"):
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def main():
    demos = glob.glob("roboverse_data/trajs/**/*.pkl.gz", recursive=True) + glob.glob(
        "roboverse_data/trajs/**/*.pkl", recursive=True
    )
    total = 0
    stat_bench = {}
    for demo_path in sorted(demos):
        print(demo_path)
        data = load_traj_file(demo_path)
        bench_name = demo_path.split("/")[2]
        task_name = demo_path.split("/")[3]
        if bench_name not in stat_bench:
            stat_bench[bench_name] = {}
        if task_name not in stat_bench[bench_name]:
            stat_bench[bench_name][task_name] = {}
        for key in data:
            if key not in stat_bench[bench_name][task_name]:
                stat_bench[bench_name][task_name][key] = 0
            stat_bench[bench_name][task_name][key] += len(data[key])
            total += len(data[key])
    print(total)
    for bench_name in stat_bench:
        print(bench_name, "*" * 10)
        for task_name in stat_bench[bench_name]:
            print(task_name, "*" * 5)
            for key in stat_bench[bench_name][task_name]:
                print(key, stat_bench[bench_name][task_name][key])


if __name__ == "__main__":
    main()
