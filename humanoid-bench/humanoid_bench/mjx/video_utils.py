from glob import glob

import natsort
import numpy as np
from moviepy.editor import VideoFileClip

from humanoid_bench.mjx.visualization_utils import make_grid, save_numpy_as_gif, save_numpy_as_video


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid_video_from_numpy(video_array, ncol, output_name="./output.mp4", speedup=1, padding=5, **kwargs):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=padding)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)


def make_grid_gif_from_numpy(video_array, ncol, output_name="./output.gif", speedup=1, fps=10):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=5)
        grid_frames.append(grid_frame)
    save_numpy_as_gif(np.array(grid_frames), output_name, fps=fps)


def make_grid_video(video_list, ncol, output_name="./output.mp4", speedup=1, **kwargs):
    videos = []
    for video_path in video_list:
        myclip = VideoFileClip(video_path)
        if myclip.size[0] > 256:
            myclip = myclip.resize(height=256)
        if speedup != 1:
            myclip = myclip.speedx(speedup)
        frames = []
        for frame in myclip.iter_frames():
            frames.append(frame)
        videos.append(np.array(frames))
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=5)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)


if __name__ == "__main__":
    # make_grid_video(glob('./data/debug/*.mp4')[:25], 5)
    video_dir = "./data/local/0908_visual_bc/0908_visual_bc_2022_09_08_09_36_00_0001/eval_200/valid/videos/"
    video_list = natsort.natsorted(glob(video_dir + "*.mp4"))
    output_name = video_dir + "grid.mp4"
    make_grid_video(video_list, len(video_list) // 2, output_name)
