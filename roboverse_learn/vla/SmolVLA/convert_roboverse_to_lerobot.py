"""Convert RoboVerse demonstrations into the LeRobot dataset format for SmolVLA.

This converter is compatible with LeRobot 0.3+ API. It walks every episode directory
under a RoboVerse demo root, reads `metadata.json` and the accompanying `rgb.mp4`,
and writes trajectories to `$HF_LEROBOT_HOME/<repo_id>` using LeRobot's storage layout.

Each frame stores:
* `observation.image`: RGB frame from the single RoboVerse camera
* `observation.state`: the absolute joint positions (`joint_qpos`)
* `action`: the target joint positions for the next step (`joint_qpos_target`)
* `task`: the textual task description (`task_desc`)

Example usage:

```bash
python roboverse_learn/vla/SmolVLA/convert_roboverse_to_lerobot.py \
  --input-root roboverse_demo/demo_mujoco \
  --repo-id <your_hf_name>/roboverse-pick_butter \
  --overwrite
```

Install requirements before running:
```bash
pip install lerobot imageio imageio-ffmpeg
```
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tyro

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError as exc:
    raise ImportError(
        "Install `lerobot` before running this script (e.g. `pip install lerobot`)."
    ) from exc


@dataclass
class Args:
    input_root: Path = Path("roboverse_demo/demo_mujoco")
    repo_id: str = "your_hf_name/roboverse-pick_butter"
    fps: int = 30
    robot_type: str = "franka"
    video_name: str = "rgb.mp4"
    metadata_name: str = "metadata.json"
    state_key: str = "joint_qpos"
    action_key: str = "joint_qpos_target"
    prompt_key: str = "task_desc"
    overwrite: bool = False
    push_to_hub: bool = False
    image_writer_threads: int = 8
    image_writer_processes: int = 2


def _iter_episode_dirs(root: Path, metadata_name: str, video_name: str) -> Iterable[Path]:
    """Iterate over valid episode directories that contain both metadata and video."""
    for meta_path in sorted(root.rglob(metadata_name)):
        episode_dir = meta_path.parent
        if (episode_dir / video_name).exists():
            yield episode_dir


def _load_metadata(meta_path: Path) -> dict:
    """Load metadata JSON file."""
    with meta_path.open("r") as handle:
        return json.load(handle)


def _first_frame_shape(video_path: Path) -> tuple[int, int, int]:
    """Get the shape of the first frame in the video."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Install `imageio` and `imageio-ffmpeg` to decode RoboVerse videos."
        ) from exc
    frame = iio.imread(video_path, index=0)
    return tuple(frame.shape)


def _frame_iter(video_path: Path):
    """Iterate over frames in a video file."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Install `imageio` and `imageio-ffmpeg` to decode RoboVerse videos."
        ) from exc
    return iio.imiter(video_path, plugin="FFMPEG")


def convert(args: Args) -> None:
    """Convert RoboVerse demonstrations to LeRobot dataset format."""
    input_root = args.input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    episode_dirs = list(_iter_episode_dirs(input_root, args.metadata_name, args.video_name))
    if not episode_dirs:
        raise RuntimeError(f"No episodes with {args.metadata_name} found under {input_root}")

    print(f"Found {len(episode_dirs)} valid episodes (with both metadata.json and rgb.mp4)")

    first_meta = _load_metadata(episode_dirs[0] / args.metadata_name)
    if args.state_key not in first_meta or args.action_key not in first_meta:
        raise KeyError(f"Metadata missing '{args.state_key}' or '{args.action_key}'")
    state_dim = len(first_meta[args.state_key][0])
    action_dim = len(first_meta[args.action_key][0])
    frame_shape = _first_frame_shape(episode_dirs[0] / args.video_name)

    print(f"Dataset configuration:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Frame shape: {frame_shape}")

    output_dir = (HF_LEROBOT_HOME / args.repo_id).resolve()
    if output_dir.exists():
        if args.overwrite:
            print(f"Removing existing dataset at {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Dataset already exists at {output_dir}. Pass --overwrite to replace it."
            )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": frame_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    written = 0
    skipped = 0
    for episode_idx, episode_dir in enumerate(episode_dirs):
        meta_path = episode_dir / args.metadata_name
        video_path = episode_dir / args.video_name
        meta = _load_metadata(meta_path)
        states = np.asarray(meta[args.state_key], dtype=np.float32)
        actions = np.asarray(meta[args.action_key], dtype=np.float32)
        prompt = str(meta.get(args.prompt_key, episode_dir.parent.name))

        total = min(len(states), len(actions))
        if total == 0:
            print(f"[SKIP] {episode_dir.name}: empty state/action data")
            skipped += 1
            continue

        frame_count = 0
        for frame_idx, frame in enumerate(_frame_iter(video_path)):
            if frame_idx >= total:
                break
            np_frame = np.asarray(frame, dtype=np.uint8)
            # Use the new LeRobot 0.3+ API: task is a separate positional argument
            dataset.add_frame(
                {
                    "observation.image": np_frame,
                    "observation.state": states[frame_idx],
                    "action": actions[frame_idx],
                },
                task=prompt,
            )
            frame_count += 1

        if frame_count == 0:
            print(f"[SKIP] {episode_dir.name}: no frames decoded")
            skipped += 1
            continue

        if frame_count != total:
            print(
                f"[WARN] {episode_dir.name}: frame count mismatch "
                f"(frames={frame_count}, states={len(states)}, actions={len(actions)})"
            )

        dataset.save_episode()
        written += 1

        if (episode_idx + 1) % 10 == 0:
            print(f"Progress: {written} episodes written, {skipped} skipped")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Total episodes processed: {len(episode_dirs)}")
    print(f"  Successfully written: {written}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    if written == 0:
        raise RuntimeError("No episodes were converted; check dataset integrity.")

    if args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["roboverse", args.robot_type],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Successfully pushed to Hub!")


def main() -> None:
    convert(tyro.cli(Args))


if __name__ == "__main__":
    main()
