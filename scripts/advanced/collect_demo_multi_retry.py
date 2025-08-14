import glob
import os
import random
import re

import yaml


def to_snake_case(camel_str: str) -> str:
    """Converts a string from camel case to snake case.

    Args:
        camel_str: A string in camel case.

    Returns:
        A string in snake case (i.e. with '_')
    """
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


with open("dashboard/conf_dataset.yml") as f:
    all_tasks = yaml.load(f, Loader=yaml.FullLoader)


tasks = all_tasks["tasks"]

benchmarks = tasks.keys()

tasks_list = []
for benchmark in benchmarks:
    tasks_list.extend(tasks[benchmark])

print(tasks_list)

# breakpoint()

rendered_dir = "roboverse_demo/demo_isaaclab"

already_rendered = [
    (path.split("/")[-1].split("-")[0], int(path.split("/")[-1].split("-")[1][-1]), path.split("/")[-1].split("-")[-1])
    for path in glob.glob(os.path.join(rendered_dir, "*"))
    if "standard_render" not in path
]


random.shuffle(tasks_list)
# tasks_list = ["pick_cube"]
# tasks_list = ["stack_cube"]

for task, level, name in already_rendered:
    command = f"python metasim/scripts/collect_demo.py --run_unfinished --task={task} --sim=isaaclab --headless --cust_name {name} --random.level {level}"
    print(command)
    os.system(command)
    command = f"/isaac-sim/python.sh metasim/scripts/collect_demo.py --run_unfinished --task={task} --sim=isaaclab --headless --cust_name {name} --random.level {level}"
    print(command)
    os.system(command)
