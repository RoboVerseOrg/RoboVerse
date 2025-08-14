import glob
import os

import cv2

ROOT = "data_isaaclab/demo"
ROOT = "roboverse_demo/demo_isaaclab"

total_traj = 0
total_frame = 0

for dir_ in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, dir_)):
        continue
    files = glob.glob(f"{ROOT}/{dir_}/*/demo_*/metadata.json")
    rgb_files = glob.glob(f"{ROOT}/{dir_}/*/demo_*/*.mp4")
    frames = []
    this_task_frames = 0
    for rgb_file in rgb_files:
        cap = cv2.VideoCapture(rgb_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        this_task_frames += length
    print(dir_, len(files), this_task_frames)
    total_traj += len(files)
    total_frame += this_task_frames
print("total_traj", total_traj)
print("total_frame", total_frame)
