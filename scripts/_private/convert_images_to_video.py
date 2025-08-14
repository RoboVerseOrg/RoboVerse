import os
from glob import glob
from multiprocessing import Pool

import cv2
import imageio as iio
import numpy as np
from tqdm.rich import tqdm_rich as tqdm

from metasim.utils.io_util import write_16bit_depth_video

folders = [os.path.dirname(s) for s in open("paths.txt").read().splitlines()]
folders = [f for f in folders if not os.path.exists(os.path.join(f, "depth_uint16.mkv"))]


def single(folder):
    ## depths
    depth_paths = glob(os.path.join(folder, "depth_*.png"))
    rgb_paths = glob(os.path.join(folder, "rgb_*.png"))
    n_depths = len(depth_paths)
    n_rgbs = len(rgb_paths)
    assert n_depths == n_rgbs
    assert os.path.exists(os.path.join(folder, f"depth_{n_depths - 1:04d}.png"))
    assert os.path.exists(os.path.join(folder, f"rgb_{n_rgbs - 1:04d}.png"))

    depth_paths = [os.path.join(folder, f"depth_{i:04d}.png") for i in range(n_depths)]
    rgb_paths = [os.path.join(folder, f"rgb_{i:04d}.png") for i in range(n_rgbs)]

    ## convert to video
    depth_imgs = [cv2.imread(d, cv2.IMREAD_UNCHANGED) for d in depth_paths]
    rgb_imgs = [iio.imread(d) for d in rgb_paths]
    write_16bit_depth_video(
        os.path.join(folder, "depth_uint16.mkv"),
        depth_imgs,
    )
    iio.mimwrite(
        os.path.join(folder, "depth_uint8.mp4"),
        [(d / 65535 * 255).astype(np.uint8) for d in depth_imgs],
        fps=30,
        quality=10,
    )
    iio.mimwrite(
        os.path.join(folder, "rgb.mp4"),
        rgb_imgs,
        fps=30,
        quality=10,
    )


def main():
    with Pool(processes=64) as p, tqdm(total=len(folders)) as pbar:
        for _ in p.imap_unordered(single, folders):
            pbar.update()
            pbar.refresh()


if __name__ == "__main__":
    main()
