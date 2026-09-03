"""Utilities for reading and writing files."""

from __future__ import annotations

import os

import cv2
import numpy as np


def touch(path: str):
    """Touch a file to update the access and modification times."""
    ## ref: https://stackoverflow.com/a/12654798
    with open(path, mode="a"):
        os.utime(path, None)


def write_16bit_depth_video(video_path: str, frames: list[np.ndarray], fps: int = 30):
    """Write a list of 16-bit depth frames to a video.

    Args:
        video_path: The path to save the video. Should end with ".mkv" or ".avi".
        frames: A list of 16-bit depth frames. Each frame is a numpy array of shape (H, W) or (H, W, 1). Either float32 (0 to 1) or uint16 (0 to 65535).
        fps: The frame rate of the video.
    """
    ## ref: https://stackoverflow.com/a/77028617
    if frames[0].dtype == np.float32:
        frames = [(frame * 65535).astype(np.uint16) for frame in frames]

    if len(frames[0].shape) == 3:
        assert frames[0].shape[-1] == 1
        frames = [frame.squeeze(-1) for frame in frames]
    else:
        assert len(frames[0].shape) == 2
    assert video_path.endswith(".mkv") or video_path.endswith(".avi")
    h, w = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        filename=video_path,
        apiPreference=cv2.CAP_FFMPEG,
        fourcc=cv2.VideoWriter_fourcc(*"FFV1"),
        fps=fps,
        frameSize=(w, h),
        params=[
            cv2.VIDEOWRITER_PROP_DEPTH,
            cv2.CV_16U,
            cv2.VIDEOWRITER_PROP_IS_COLOR,
            0,  # false
        ],
    )
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def read_16bit_depth_video(video_path: str) -> list[np.ndarray]:
    """Read a 16-bit depth video.

    Args:
        video_path: The path to the video. Should end with ".mkv" or ".avi".

    Returns:
        A list of 16-bit depth frames. Each frame is a numpy array of shape (H, W). The dtype is uint16.
    """
    ## ref: https://stackoverflow.com/a/77028617
    assert video_path.endswith(".mkv") or video_path.endswith(".avi")
    video_capture = cv2.VideoCapture(
        filename=video_path,
        apiPreference=cv2.CAP_FFMPEG,
        params=[
            cv2.CAP_PROP_CONVERT_RGB,
            0,  # false
        ],
    )
    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    return frames
