"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-10-23 16:00:57

poses = {
    "objects": {
        "table": {
            "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
        },
        "banana": {
            "pos": torch.tensor([0.352536, -0.477441, 0.819529]),
            "rot": torch.tensor([0.235927, -0.136440, 0.004026, 0.962136]),
        },
        "mug": {
            "pos": torch.tensor([0.748443, -0.347645, 0.864654]),
            "rot": torch.tensor([0.748473, -0.000780, 0.000998, -0.663164]),
        },
        "book": {
            "pos": torch.tensor([0.268786, -0.316599, 0.818212]),
            "rot": torch.tensor([0.844761, 0.001680, 0.002561, -0.535134]),
        },
        "lamp": {
            "pos": torch.tensor([0.684974, 0.100071, 1.054534]),
            "rot": torch.tensor([0.999994, -0.000070, 0.001461, 0.003300]),
        },
        "remote_control": {
            "pos": torch.tensor([0.734845, -0.618761, 0.811628]),
            "rot": torch.tensor([0.703674, 0.005018, 0.004427, 0.710491]),
        },
        "rubiks_cube": {
            "pos": torch.tensor([0.575018, -0.623688, 0.830375]),
            "rot": torch.tensor([0.978458, 0.000122, 0.000443, 0.206447]),
        },
        "vase": {
            "pos": torch.tensor([0.299030, 0.051064, 0.953119]),
            "rot": torch.tensor([0.999986, -0.001200, -0.001866, 0.004861]),
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.800000, -0.800000, 0.780000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
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
