import argparse
import os

import imageio
import numpy as np
from PIL import Image


def images_to_video(image_folder, video_path, frame_size=(1920, 1080), fps=30):
    images = sorted([
        img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")
    ])

    if not images:
        print("No images found in the specified directory!")
        return

    writer = imageio.get_writer(video_path, fps=fps)

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = imageio.imread(img_path)

        if img.shape[1] > frame_size[0] or img.shape[0] > frame_size[1]:
            print("Warning: frame size is smaller than the one of the images.")
            print("Images will be resized to match frame size.")
            img = np.array(Image.fromarray(img).resize(frame_size))

        writer.append_data(img)

    writer.close()
    print("Video created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()
    images_to_video(args.image_folder, args.video_path)
