"""Compact ACT-style chunked behavior-cloning policy for LIBERO.

A real (small) IL baseline: two ResNet18 image encoders (agentview +
eye-in-hand) + a proprio MLP, fused and decoded into an action *chunk* (the core
action-chunking idea from ACT). Trained with L1 on demo actions; closed-loop
eval uses temporal ensembling over overlapping chunks. Shared verbatim between
the GPU trainer and the CPU closed-loop evaluator so weights load 1:1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

# Joint positions + gripper: an unambiguous proprio representation that is
# byte-identical between the demo HDF5 ("joint_states"/"gripper_states") and the
# native eval obs ("robot0_joint_pos"/"robot0_gripper_qpos") -- avoids the
# axis-angle-vs-quaternion mismatch that ee orientation would introduce.
PROPRIO_KEYS = ("joint_states", "gripper_states")  # 7 + 2 = 9
PROPRIO_DIM = 9
ACTION_DIM = 7
CHUNK = 16
IMG = 128


def _resnet18_encoder() -> nn.Module:
    # Pretrained ImageNet stems help a lot with few demos; fall back to scratch
    # (e.g. offline) so training still runs without network access.
    try:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()  # -> 512-d feature
    return m


class ChunkedBCPolicy(nn.Module):
    """images(agentview, eye_in_hand) + proprio -> next CHUNK actions."""

    def __init__(self, chunk: int = CHUNK, hidden: int = 512):
        super().__init__()
        self.chunk = chunk
        self.enc_agent = _resnet18_encoder()
        self.enc_hand = _resnet18_encoder()
        self.proprio = nn.Sequential(nn.Linear(PROPRIO_DIM, 128), nn.ReLU(), nn.Linear(128, 128))
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk * ACTION_DIM),
        )

    def forward(self, agent_img, hand_img, proprio):
        fa = self.enc_agent(agent_img)
        fh = self.enc_hand(hand_img)
        fp = self.proprio(proprio)
        z = torch.cat([fa, fh, fp], dim=-1)
        out = self.head(z)
        return out.view(-1, self.chunk, ACTION_DIM)


# ImageNet normalization for the (untrained-from-scratch) resnet stems.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep_image(img_uint8: np.ndarray, device) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,H,W) normalized float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8)).float().div(255.0)
    t = t.permute(2, 0, 1).unsqueeze(0)
    return ((t - _MEAN) / _STD).to(device)


def proprio_vec(obs: dict) -> np.ndarray:
    """Assemble the 9-d proprio vector from a LIBERO obs dict (native or demo).

    joint positions (7) + gripper (2). Native eval obs uses ``robot0_joint_pos``
    / ``robot0_gripper_qpos``; the demo HDF5 uses ``joint_states`` /
    ``gripper_states`` -- the same physical quantities.
    """
    joints = np.asarray(obs.get("robot0_joint_pos", obs.get("joint_states")))[:7]
    grip = np.asarray(obs.get("robot0_gripper_qpos", obs.get("gripper_states")))[:2]
    return np.concatenate([joints.ravel(), grip.ravel()]).astype(np.float32)
