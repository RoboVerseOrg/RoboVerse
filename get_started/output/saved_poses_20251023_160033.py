"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-10-23 16:00:33

poses = {
    "objects": {
        "table": {
            "pos": torch.tensor([0.400000, -0.200000, 0.400000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
        },
        "banana": {
            "pos": torch.tensor([0.327516, -0.543494, 0.819349]),
            "rot": torch.tensor([0.940695, -0.068877, 0.117251, 0.310808]),
        },
        "mug": {
            "pos": torch.tensor([0.719082, -0.335252, 0.863782]),
            "rot": torch.tensor([0.873546, -0.000435, 0.001989, -0.486737]),
        },
        "book": {
            "pos": torch.tensor([0.286618, -0.297241, 0.817537]),
            "rot": torch.tensor([0.989636, 0.000553, 0.003487, -0.143558]),
        },
        "lamp": {
            "pos": torch.tensor([0.682159, 0.100024, 1.054530]),
            "rot": torch.tensor([0.999998, -0.000090, 0.001425, 0.001088]),
        },
        "remote_control": {
            "pos": torch.tensor([0.700050, -0.595647, 0.808003]),
            "rot": torch.tensor([0.953809, 0.001142, 0.001779, 0.300408]),
        },
        "rubiks_cube": {
            "pos": torch.tensor([0.548829, -0.606272, 0.830059]),
            "rot": torch.tensor([0.998727, -0.000612, 0.000403, 0.050429]),
        },
        "vase": {
            "pos": torch.tensor([0.299562, 0.050490, 0.953118]),
            "rot": torch.tensor([0.999999, -0.000792, -0.000850, 0.000634]),
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
