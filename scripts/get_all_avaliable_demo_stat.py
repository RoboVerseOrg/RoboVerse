import glob
import os

ROOT = "data_isaaclab/demo"

tasks_available = sorted(os.listdir(ROOT))

for task in tasks_available:
    try:
        robot_available = sorted(os.listdir(os.path.join(ROOT, task)))
    except:
        continue
    for robot in robot_available:
        paths = glob.glob(f"{ROOT}/{task}/{robot}/demo_*/metadata.json")
        print(f"Task: {task}, Robot: {robot}, Number of demos: {len(paths)}")
