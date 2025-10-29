"""
RoboVerse policy adapter for SmolVLA using LeRobot framework.

This module provides input/output transforms to adapt RoboVerse data format
to SmolVLA's expected format within the LeRobot training pipeline.
"""

import dataclasses
from typing import Dict, Any

import torch
import torch.nn as nn

__all__ = [
    "RoboVerseSmolVLAInputs",
    "RoboVerseSmolVLAOutputs",
    "RoboVerseSmolVLAConfig",
    "create_roboverse_transforms",
    "get_roboverse_data_config",
]


class RoboVerseSmolVLAInputs(nn.Module):
    """
    Transform RoboVerse observations to SmolVLA input format.

    Expected input keys from RoboVerse:
    - image: RGB image (B, H, W, 3) or (B, 3, H, W)
    - state: Robot state (joint positions) (B, state_dim)
    - prompt: Text instruction string

    SmolVLA expects:
    - image: RGB image normalized to [0, 1]
    - instruction: Text prompt
    """

    def __init__(self, image_size=(224, 224), normalize_image=True):
        super().__init__()
        self.image_size = image_size
        self.normalize_image = normalize_image

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform RoboVerse batch to SmolVLA format.

        Args:
            batch: Dictionary containing 'image', 'state', 'prompt', etc.

        Returns:
            Transformed batch compatible with SmolVLA
        """
        output = {}

        # Handle image
        if "image" in batch:
            image = batch["image"]

            # Ensure NCHW format
            if image.ndim == 3:
                image = image.unsqueeze(0)  # Add batch dim
            if image.shape[-1] == 3:  # HWC -> CHW
                image = image.permute(0, 3, 1, 2)

            # Normalize if needed
            if self.normalize_image and image.dtype == torch.uint8:
                image = image.float() / 255.0

            # Resize if needed
            if image.shape[-2:] != self.image_size:
                image = torch.nn.functional.interpolate(
                    image,
                    size=self.image_size,
                    mode='bilinear',
                    align_corners=False
                )

            output["image"] = image

        # Handle instruction/prompt
        if "prompt" in batch:
            output["instruction"] = batch["prompt"]
        elif "instruction" in batch:
            output["instruction"] = batch["instruction"]

        # Pass through state if present (may be used for certain model variants)
        if "state" in batch:
            output["state"] = batch["state"]

        # Pass through actions (for training)
        if "actions" in batch:
            output["actions"] = batch["actions"]

        return output


class RoboVerseSmolVLAOutputs(nn.Module):
    """
    Transform SmolVLA outputs to RoboVerse action format.

    SmolVLA outputs actions as (B, action_horizon, action_dim).
    For RoboVerse, we typically use the first action in the horizon.
    """

    def __init__(self, use_first_action=True):
        super().__init__()
        self.use_first_action = use_first_action

    def forward(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform SmolVLA output to RoboVerse format.

        Args:
            model_output: Dictionary containing 'actions' from SmolVLA

        Returns:
            Transformed output compatible with RoboVerse
        """
        output = {}

        if "actions" in model_output:
            actions = model_output["actions"]

            # If action horizon > 1, take first action
            if actions.ndim == 3 and self.use_first_action:
                actions = actions[:, 0, :]  # (B, H, D) -> (B, D)

            output["actions"] = actions

        return output


@dataclasses.dataclass
class RoboVerseSmolVLAConfig:
    """Configuration for RoboVerse-SmolVLA integration."""

    # Image preprocessing
    image_size: tuple = (224, 224)
    normalize_image: bool = True

    # Action processing
    use_first_action: bool = True
    action_dim: int = 7  # [dx, dy, dz, drx, dry, drz, gripper]

    # Delta vs absolute actions
    use_delta_actions: bool = True
    delta_action_mask: list = None  # Which action dims are deltas

    def __post_init__(self):
        if self.delta_action_mask is None:
            # By default, all dims except last (gripper) are deltas
            self.delta_action_mask = [True] * (self.action_dim - 1) + [False]


def create_roboverse_transforms(config: RoboVerseSmolVLAConfig = None):
    """
    Create input and output transforms for RoboVerse-SmolVLA integration.

    Args:
        config: Configuration object. If None, uses default config.

    Returns:
        Tuple of (input_transform, output_transform)
    """
    if config is None:
        config = RoboVerseSmolVLAConfig()

    input_transform = RoboVerseSmolVLAInputs(
        image_size=config.image_size,
        normalize_image=config.normalize_image
    )

    output_transform = RoboVerseSmolVLAOutputs(
        use_first_action=config.use_first_action
    )

    return input_transform, output_transform


def get_roboverse_data_config():
    """
    Example data configuration for LeRobot training.

    Add this to your LeRobot training config:

    ```python
    from roboverse_smolvla_policy import get_roboverse_data_config

    data_config = get_roboverse_data_config()
    ```
    """
    config = RoboVerseSmolVLAConfig()
    input_transform, output_transform = create_roboverse_transforms(config)

    return {
        "input_transforms": input_transform,
        "output_transforms": output_transform,
        "config": config,
    }


if __name__ == "__main__":
    # Test transforms
    print("Testing RoboVerse-SmolVLA transforms...")

    config = RoboVerseSmolVLAConfig()
    input_tf, output_tf = create_roboverse_transforms(config)

    # Test input transform
    dummy_batch = {
        "image": torch.randint(0, 255, (2, 256, 256, 3), dtype=torch.uint8),
        "state": torch.randn(2, 7),
        "prompt": ["pick up the butter", "grasp the object"],
        "actions": torch.randn(2, 7),
    }

    transformed = input_tf(dummy_batch)
    print("Input transform successful!")
    print(f"  Image shape: {transformed['image'].shape}")
    print(f"  Image dtype: {transformed['image'].dtype}")
    print(f"  Image range: [{transformed['image'].min():.3f}, {transformed['image'].max():.3f}]")
    print(f"  Instructions: {transformed['instruction']}")

    # Test output transform
    model_output = {
        "actions": torch.randn(2, 10, 7)  # (B, horizon, action_dim)
    }

    transformed_out = output_tf(model_output)
    print("\nOutput transform successful!")
    print(f"  Action shape: {transformed_out['actions'].shape}")

    print("\nAll transforms working correctly!")
