"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-11-11 18:40:20

poses = {
    "objects": {
        "table": {
            "pos": torch.tensor([0.590000, -0.310000, 0.400000]),
            "rot": torch.tensor([0.900447, -0.000000, -0.000000, 0.434966]),
        },
        "banana": {
            "pos": torch.tensor([0.309824, -0.556204, 0.017262]),
            "rot": torch.tensor([0.980276, -0.047087, 0.130006, 0.141210]),
        },
        "mug": {
            "pos": torch.tensor([0.791811, -0.339044, 0.865300]),
            "rot": torch.tensor([1.000000, -0.000001, 0.000066, -0.000883]),
        },
        "book": {
            "pos": torch.tensor([0.371151, -0.327381, 0.821943]),
            "rot": torch.tensor([0.995850, -0.001929, 0.003827, -0.090906]),
        },
        "lamp": {
            "pos": torch.tensor([1.001164, 0.371955, 0.119492]),
            "rot": torch.tensor([0.761892, -0.568836, 0.286753, -0.117128]),
        },
        "remote_control": {
            "pos": torch.tensor([0.870649, -0.498104, 0.811890]),
            "rot": torch.tensor([0.973876, 0.000347, 0.001498, 0.227073]),
        },
        "rubiks_cube": {
            "pos": torch.tensor([0.827811, -0.545319, 0.832990]),
            "rot": torch.tensor([0.971273, -0.000209, 0.000055, 0.237966]),
        },
        "vase": {
            "pos": torch.tensor([0.322260, -0.022052, 0.078785]),
            "rot": torch.tensor([0.515820, 0.141548, -0.758338, 0.372582]),
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.960000, -0.800000, 0.780000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
            "dof_pos": {
                "panda_finger_joint1": 0.040000,
                "panda_finger_joint2": 0.040000,
                "panda_joint1": 1.239999,
                "panda_joint2": -0.245398,
                "panda_joint3": 0.880000,
                "panda_joint4": -1.476195,
                "panda_joint5": 0.960000,
                "panda_joint6": 3.230794,
                "panda_joint7": 0.785398,
            },
        },
    },
}
