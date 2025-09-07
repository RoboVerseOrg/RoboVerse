from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import json

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

class RoboVerseDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, qpos or RTX version: consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, qpos or RTX version: consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='demo/'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _get_episode_paths(path):
            episode_paths = []
            tasks = os.listdir(path)
            for task in tasks:
                task_path = os.path.join(path, task)
                if not os.path.isdir(task_path):
                    continue
                robots = os.listdir(task_path)
                for robot in robots:
                    robot_path = os.path.join(task_path, robot)
                    episodes = os.listdir(robot_path)
                    for episode in episodes:
                        episode_path = os.path.join(robot_path, episode)
                        if not os.path.isdir(episode_path):
                            continue
                        episode_paths.append(episode_path)
            return episode_paths

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset

            metadata_path = os.path.join(episode_path, 'metadata.json')
            camera_param_path = os.path.join(episode_path, 'cam_info.json') # if exists

            assert os.path.isfile(metadata_path), f"metadata.json not found in {episode_path}"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            ee_states = metadata.get('robot_ee_state', [])
            task_desc = metadata.get('task_desc', [])
            depth_min = metadata.get('depth_min', [])
            depth_max = metadata.get('depth_max', [])

            images = sorted([img for img in os.listdir(episode_path) if img.startswith('rgb')],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            depth_images = sorted([img for img in os.listdir(episode_path) if img.startswith('depth')],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            if len(images) != len(depth_images):
                return None

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(images):
                # compute Kona language embedding
                language_embedding = self._embed([task_desc[0]])[0].numpy()

                episode.append({
                    'observation': {
                        'image': os.path.join(episode_path, images[i]),
                        'depth_image': os.path.join(episode_path, depth_images[i]),
                        'state': ee_states[i],
                    },
                    'action': ee_states[i + 1] if i + 1 < len(ee_states) else ee_states[i],
                    'discount': 1.0,
                    'reward': float(i == (len(images) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(images) - 1),
                    'is_terminal': i == (len(images) - 1),
                    'language_instruction': task_desc[0],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample


        episode_paths = _get_episode_paths(path)
        for episode_path in episode_paths:
            yield _parse_example(episode_path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
