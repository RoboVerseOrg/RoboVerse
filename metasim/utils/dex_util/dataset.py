# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset

"""DexYCB dataset."""

import logging
from pathlib import Path

import numpy as np
import yaml

_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

_MANO_JOINT_CONNECT = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


class DexYCBVideoDataset:
    """DexYCB video dataset for hand-object interaction data.

    Args:
        data_dir: Directory containing the DexYCB dataset.
        hand_type: Type of hand ("left" or "right").
        filter_objects: List of object names to filter for.
    """

    def __init__(self, data_dir, hand_type="right", filter_objects=None):
        self._data_dir = Path(data_dir)
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"

        # Filter
        if filter_objects is None:
            filter_objects = []
        self.use_filter = len(filter_objects) > 0
        inverse_ycb_class = {"_".join(value.split("_")[1:]): key for key, value in YCB_CLASSES.items()}
        ycb_object_names = list(inverse_ycb_class.keys())
        filter_ids = []
        for obj in filter_objects:
            if obj not in ycb_object_names:
                logging.warning(f"Filter object name {obj} is not a valid YCB name")
            else:
                filter_ids.append(inverse_ycb_class[obj])

        # Camera and mano
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        self._mano_side = hand_type
        self._mano_parameters = self._load_mano()

        # Capture data
        self._subject_dirs = [sub for sub in self._data_dir.iterdir() if sub.stem in _SUBJECTS]
        self._capture_meta = {}
        self._capture_pose = {}
        self._capture_filter = {}
        self._captures = []
        for subject_dir in self._subject_dirs:
            for capture_dir in subject_dir.iterdir():
                meta_file = capture_dir / "meta.yml"
                with meta_file.open(mode="r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                if hand_type not in meta["mano_sides"]:
                    continue

                pose = np.load((capture_dir / "pose.npz").resolve().__str__())
                if self.use_filter:
                    ycb_ids = meta["ycb_ids"]
                    # Skip current capture if no desired object inside
                    if len(list(set(ycb_ids) & set(filter_ids))) < 1:
                        continue
                    capture_filter = [i for i in range(len(ycb_ids)) if ycb_ids[i] in filter_ids]
                    object_pose = pose["pose_y"]
                    frame_indices, filter_id = self._filter_object_motion_frame(capture_filter, object_pose)
                    if len(frame_indices) < 20:
                        continue
                    self._capture_filter[capture_dir.stem] = [filter_id]
                self._capture_meta[capture_dir.stem] = meta
                self._capture_pose[capture_dir.stem] = pose
                self._captures.append(capture_dir.stem)

    def __len__(self):
        return len(self._captures)

    def __iter__(self):
        """Iterate over the dataset.

        Returns:
            Iterator: Iterator over the dataset items.
        """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item: int):
        if item >= self.__len__():
            raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        hand_pose = pose["pose_m"]
        object_pose = pose["pose_y"]
        ycb_ids = meta["ycb_ids"]

        # Load extrinsic and mano parameters
        extrinsic_name = meta["extrinsics"]
        extrinsic_mat = np.array(self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]).reshape([3, 4])
        extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
        mano_name = meta["mano_calib"][0]
        mano_parameters = self._mano_parameters[mano_name]

        if self.use_filter:
            capture_filter = np.array(self._capture_filter[capture_name])
            frame_indices, _ = self._filter_object_motion_frame(capture_filter, object_pose)
            ycb_ids = [ycb_ids[valid_id] for valid_id in self._capture_filter[capture_name]]
            hand_pose = hand_pose[frame_indices]
            object_pose = object_pose[frame_indices][:, capture_filter, :]
        object_mesh_files = [self._object_mesh_file(ycb_id) for ycb_id in ycb_ids]

        ycb_data = dict(
            hand_pose=hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsic_mat,
            ycb_ids=ycb_ids,
            hand_shape=mano_parameters,
            object_mesh_file=object_mesh_files,
            capture_name=capture_name,
        )
        return ycb_data

    def _filter_object_motion_frame(self, capture_filter, object_pose, frame_margin=40):
        frames = np.arange(0)
        for filter_id in capture_filter:
            filter_object_pose = object_pose[:, filter_id, :]
            object_move_list = []
            for frame in range(filter_object_pose.shape[0] - 2):
                object_move_list.append(self.is_object_move(filter_object_pose[frame:, :]))
            if True not in object_move_list:
                continue
            first_frame = object_move_list.index(True)
            last_frame = len(object_move_list) - object_move_list[::-1].index(True) - 1
            start = max(0, first_frame - frame_margin)
            end = min(filter_object_pose.shape[0], last_frame + frame_margin)
            frames = np.arange(start, end)
            break
        return frames, filter_id

    @staticmethod
    def is_object_move(single_object_pose: np.ndarray):
        """Check if the object is moving.

        Args:
            single_object_pose (np.ndarray): The pose of the object.

        Returns:
            bool: True if the object is moving, False otherwise.
        """
        single_object_trans = single_object_pose[:, 4:]
        future_frame = min(single_object_trans.shape[0] - 1, 5)
        current_move = np.linalg.norm(single_object_trans[1] - single_object_trans[0]) > 2e-2
        future_move = np.linalg.norm(single_object_trans[future_frame] - single_object_trans[0]) > 5e-2
        return current_move or future_move

    def _object_mesh_file(self, object_id):
        obj_file = self._data_dir / "models" / YCB_CLASSES[object_id] / "textured_simple.obj"
        return str(obj_file.resolve())

    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("extrinsics"):
                continue
            extrinsic_file = cali_dir / "extrinsics.yml"
            name = cali_dir.stem[len("extrinsics_") :]
            with extrinsic_file.open(mode="r") as f:
                extrinsic = yaml.load(f, Loader=yaml.FullLoader)
            extrinsics[name] = extrinsic

        intrinsic_dir = self._calib_dir / "intrinsics"
        for intrinsic_file in intrinsic_dir.iterdir():
            with intrinsic_file.open(mode="r") as f:
                intrinsic = yaml.load(f, Loader=yaml.FullLoader)
            name = intrinsic_file.stem.split("_")[0]
            x = intrinsic["color"]
            camera_mat = np.array([[x["fx"], 0.0, x["ppx"]], [0.0, x["fy"], x["ppy"]], [0.0, 0.0, 1.0]])
            intrinsics[name] = camera_mat

        return intrinsics, extrinsics

    def _load_mano(self):
        mano_parameters = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("mano"):
                continue

            mano_file = cali_dir / "mano.yml"
            with mano_file.open(mode="r") as f:
                shape_parameters = yaml.load(f, Loader=yaml.FullLoader)
            mano_name = "_".join(cali_dir.stem.split("_")[1:])
            mano_parameters[mano_name] = np.array(shape_parameters["betas"])

        return mano_parameters


def main(dexycb_dir: str):
    """Main function for testing the DexYCBVideoDataset.

    Args:
        dexycb_dir: Directory containing the DexYCB dataset.
    """
    from collections import Counter

    dataset = DexYCBVideoDataset(dexycb_dir)
    logging.info(f"Number of captures: {len(dataset)}")

    ycb_names = []
    for i, data in enumerate(dataset):
        ycb_ids = int(data["ycb_ids"][0])
        ycb_names.append(YCB_CLASSES[ycb_ids])

    counter = Counter(ycb_names)
    logging.info(counter)

    sample = dataset[0]
    logging.info(sample.keys())


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
