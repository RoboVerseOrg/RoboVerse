"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-10-23 16:02:49

poses = {
    "objects": {
        "table": {
            "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
        },
        "banana": {
            "pos": torch.tensor([0.333304, -0.456427, 0.819528]),
            "rot": torch.tensor([-0.488764, 0.130237, -0.040990, -0.861666]),
        },
        "mug": {
            "pos": torch.tensor([0.844106, -0.266613, 0.043941]),
            "rot": torch.tensor([-0.307712, -0.369976, 0.582579, -0.655006]),
        },
        "book": {
            "pos": torch.tensor([0.188609, -0.302352, 0.816615]),
            "rot": torch.tensor([-0.999974, -0.000038, -0.003419, 0.006270]),
        },
        "lamp": {
            "pos": torch.tensor([0.699817, 0.100155, 1.054492]),
            "rot": torch.tensor([0.999734, -0.000041, 0.001581, 0.023004]),
        },
        "remote_control": {
            "pos": torch.tensor([0.839061, -0.641684, 0.010703]),
            "rot": torch.tensor([0.008498, 0.659394, 0.751749, -0.001231]),
        },
        "rubiks_cube": {
            "pos": torch.tensor([0.584011, -0.619940, 0.830712]),
            "rot": torch.tensor([0.958410, 0.000931, 0.000610, 0.285394]),
        },
        "vase": {
            "pos": torch.tensor([0.298688, 0.052866, 0.953117]),
            "rot": torch.tensor([0.999638, -0.000592, 0.000430, 0.026893]),
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.890000, -0.250001, 0.780000]),
            "rot": torch.tensor([-0.029191, -0.024987, 0.000730, -0.999261]),
            "dof_pos": {
                "panda_finger_joint1": 0.040000,
                "panda_finger_joint2": 0.040000,
                "panda_joint1": -0.000000,
                "panda_joint2": -0.785398,
                "panda_joint3": -0.000000,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.000000,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
            },
        },
    },
}
