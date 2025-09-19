from typing import Iterator, Tuple, Any
import os
import json
import imageio.v2 as iio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'


class RoboVerseDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RoboVerse demos (mp4 + metadata)."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sentence embed model
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (features)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            doc='RGB image (H,W,3).'
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            doc='Depth (visualized as uint8) (H,W,1).'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='EE state (7 dims).'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Delta action (next_state - current_state) (7 dims).'
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,), dtype=np.float32
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                    'depth_min': tfds.features.Tensor(shape=(None,), dtype=np.float32),
                    'depth_max': tfds.features.Tensor(shape=(None,), dtype=np.float32),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits. `path` can be relative to cwd or absolute."""
        # Use manual path 'demo/' relative to current working dir by default.
        # If running under TFDS specifics, consider using dl_manager.manual_dir.
        return {
            'train': self._generate_examples(path='demo/'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator: find episodes and parse them."""

        def _get_episode_paths(root_path):
            """Traverse task/robot/episode nested layout and collect episode dirs."""
            roots = []
            if not os.path.exists(root_path):
                return roots
            for task in sorted(os.listdir(root_path)):
                task_path = os.path.join(root_path, task)
                if not os.path.isdir(task_path):
                    continue
                for robot in sorted(os.listdir(task_path)):
                    robot_path = os.path.join(task_path, robot)
                    if not os.path.isdir(robot_path):
                        continue
                    for ep in sorted(os.listdir(robot_path)):
                        ep_path = os.path.join(robot_path, ep)
                        if os.path.isdir(ep_path):
                            roots.append(ep_path)
            return roots

        def _count_frames(reader):
            """Robust frame counting for imageio Reader."""
            try:
                return reader.count_frames()
            except Exception:
                # fallback: probe until get_data raises
                i = 0
                while True:
                    try:
                        reader.get_data(i)
                        i += 1
                    except Exception:
                        break
                return i

        def _safe_reader(path):
            """Return imageio reader or None on failure."""
            if not os.path.isfile(path):
                return None
            try:
                return iio.get_reader(path, "ffmpeg")
            except Exception:
                return None

        def _parse_example(episode_path):
            """Parse one episode folder into (id, sample) or return None on failure."""
            # load metadata
            meta_file = os.path.join(episode_path, "metadata.json")
            if not os.path.isfile(meta_file):
                return None

            with open(meta_file, "r") as f:
                meta = json.load(f)

            # get keys (support both names)
            ee_states = meta.get("ee_state") or meta.get("robot_ee_state") or []
            if not ee_states:
                return None
            task_desc = meta.get("task_desc", "") or (meta.get("task_desc_list", [""])[0] if meta.get("task_desc_list") else "")
            depth_min = meta.get("depth_min", [])
            depth_max = meta.get("depth_max", [])

            # open readers
            rgb_path = os.path.join(episode_path, "rgb.mp4")
            depth_path = os.path.join(episode_path, "depth_uint8.mp4")  # visualized depth mp4

            rgb_reader = _safe_reader(rgb_path)
            depth_reader = _safe_reader(depth_path)

            if rgb_reader is None or depth_reader is None:
                # try alternative names if present
                alt_rgb = os.path.join(episode_path, "rgb.mp4")
                alt_depth = os.path.join(episode_path, "depth.mp4")
                rgb_reader = rgb_reader or _safe_reader(alt_rgb)
                depth_reader = depth_reader or _safe_reader(alt_depth)
            if rgb_reader is None or depth_reader is None:
                # cannot decode videos
                print(f"[WARN] Missing video reader for {episode_path}")
                return None

            # count frames
            T_rgb = _count_frames(rgb_reader)
            T_depth = _count_frames(depth_reader)

            T_state = len(ee_states)
            if not (T_rgb == T_depth == T_state):
                # lengths must match
                rgb_reader.close()
                depth_reader.close()
                print(f"[WARN] length mismatch: rgb={T_rgb}, depth={T_depth}, state={T_state} @ {episode_path}")
                return None

            # compute language embedding once
            lang_emb = self._embed([task_desc])[0].numpy() if task_desc else np.zeros((512,), dtype=np.float32)

            steps = []
            for t in range(T_rgb):
                # read frames
                try:
                    rgb = rgb_reader.get_data(t)           # (H,W,3) uint8
                except Exception:
                    print(f"[WARN] failed read rgb frame {t} @ {episode_path}")
                    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

                try:
                    depth = depth_reader.get_data(t)
                    # ensure single channel
                    if depth.ndim == 3 and depth.shape[2] > 1:
                        depth = depth[:, :, :1]
                    elif depth.ndim == 2:
                        depth = depth[:, :, None]
                except Exception:
                    print(f"[WARN] failed read depth frame {t} @ {episode_path}")
                    depth = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8)

                state_t = np.array(ee_states[t], dtype=np.float32)
                # Calculate delta action (next_state - current_state)
                if t + 1 < T_state:
                    next_state = np.array(ee_states[t + 1], dtype=np.float32)
                    action_t = next_state - state_t  # delta action
                else:
                    action_t = np.zeros_like(state_t)  # zero action for last step

                step = {
                    "observation": {
                        "image": rgb,
                        "depth_image": depth,
                        "state": state_t,
                    },
                    "action": action_t,
                    "discount": 1.0,
                    "reward": float(t == (T_rgb - 1)),
                    "is_first": t == 0,
                    "is_last": t == (T_rgb - 1),
                    "is_terminal": t == (T_rgb - 1),
                    "language_instruction": task_desc,
                    "language_embedding": lang_emb,
                }
                steps.append(step)

            # close readers
            rgb_reader.close()
            depth_reader.close()

            sample = {
                "steps": steps,
                "episode_metadata": {
                    "file_path": episode_path,
                    "depth_min": np.array(depth_min, dtype=np.float32).tolist(),
                    "depth_max": np.array(depth_max, dtype=np.float32).tolist(),
                }
            }
            return episode_path, sample

        # main loop
        root_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        episode_paths = _get_episode_paths(root_path)

        for episode_path in episode_paths:
            parsed = _parse_example(episode_path)
            if parsed is None:
                continue
            yield parsed

        # parallel Beam variant (commented)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #     beam.Create(episode_paths)
        #     | beam.Map(_parse_example)
        # )
