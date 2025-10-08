"""Convert RoboVerse demonstrations into the LeRobot dataset format.

The converter walks every episode directory under a RoboVerse demo root, reads
`metadata.json` and the accompanying `rgb.mp4`, and writes trajectories to
`$HF_LEROBOT_HOME/<repo_id>` using LeRobot's storage layout. Each frame stores:

* `image`: RGB frame from the single RoboVerse camera.
* `state`: the absolute joint positions (`joint_qpos`).
* `actions`: the target joint positions for the next step (`joint_qpos_target`).
* `prompt`: the textual task description (`task_desc`).

Example usage:

```
uv run roboverse_learn/vla/convert_roboverse_to_lerobot.py \
  --input-root /home/celia/codes/RoboVerse/roboverse_demo \
  --repo-id your-hf-handle/roboverse-pick-butter
```

Install requirements before running:
```
uv pip install lerobot imageio imageio-ffmpeg
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
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Install `lerobot` before running this script (e.g. `uv pip install lerobot`)."
    ) from exc


@dataclass
class Args:
    input_root: Path = Path("~/codes/RoboVerse/roboverse_demo")
    repo_id: str = "your_hf_name/roboverse-pick-butter"
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
    for meta_path in sorted(root.rglob(metadata_name)):
        episode_dir = meta_path.parent
        if (episode_dir / video_name).exists():
            yield episode_dir


def _load_metadata(meta_path: Path) -> dict:
    with meta_path.open("r") as handle:
        return json.load(handle)


def _first_frame_shape(video_path: Path) -> tuple[int, int, int]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install `imageio` and `imageio-ffmpeg` to decode RoboVerse videos."
        ) from exc
    frame = iio.imread(video_path, index=0)
    return tuple(frame.shape)


def _frame_iter(video_path: Path):
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install `imageio` and `imageio-ffmpeg` to decode RoboVerse videos."
        ) from exc
    return iio.imiter(video_path, plugin="FFMPEG")


def convert(args: Args) -> None:
    input_root = args.input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    episode_dirs = list(_iter_episode_dirs(input_root, args.metadata_name, args.video_name))
    if not episode_dirs:
        raise RuntimeError(f"No episodes with {args.metadata_name} found under {input_root}")

    first_meta = _load_metadata(episode_dirs[0] / args.metadata_name)
    if args.state_key not in first_meta or args.action_key not in first_meta:
        raise KeyError(f"Metadata missing '{args.state_key}' or '{args.action_key}'")
    state_dim = len(first_meta[args.state_key][0])
    action_dim = len(first_meta[args.action_key][0])
    frame_shape = _first_frame_shape(episode_dirs[0] / args.video_name)

    output_dir = (HF_LEROBOT_HOME / args.repo_id).resolve()
    if output_dir.exists():
        if args.overwrite:
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
            "image": {
                "dtype": "image",
                "shape": frame_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    written = 0
    for episode_dir in episode_dirs:
        meta_path = episode_dir / args.metadata_name
        video_path = episode_dir / args.video_name
        meta = _load_metadata(meta_path)
        states = np.asarray(meta[args.state_key], dtype=np.float32)
        actions = np.asarray(meta[args.action_key], dtype=np.float32)
        prompt = str(meta.get(args.prompt_key, episode_dir.parent.name))

        total = min(len(states), len(actions))
        if total == 0:
            continue

        frame_count = 0
        for frame_idx, frame in enumerate(_frame_iter(video_path)):
            if frame_idx >= total:
                break
            np_frame = np.asarray(frame, dtype=np.uint8)
            dataset.add_frame(
                {
                    "image": np_frame,
                    "state": states[frame_idx],
                    "actions": actions[frame_idx],
                    "task": prompt,
                }
            )
            frame_count += 1

        if frame_count == 0:
            continue
        if frame_count != total:
            print(
                f"[WARN] Length mismatch in {episode_dir}: frames={frame_count},"
                f" states={len(states)}, actions={len(actions)}."
            )
        dataset.save_episode()
        written += 1

    if written == 0:
        raise RuntimeError("No episodes were converted; check dataset integrity.")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["roboverse", args.robot_type],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main() -> None:
    convert(tyro.cli(Args))


if __name__ == "__main__":
    main()
