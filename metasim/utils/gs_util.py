# ruff: noqa: UP006, UP007
from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion


def quaternion_multiply(init_quat: List[float], rotate_quat: List[float]) -> List[float]:
    """Multiplies two quaternions.

    Args:
        init_quat: Initial quaternion.
        rotate_quat: Rotation quaternion.

    Returns:
        The product of the two quaternions.

    Raises:
        ValueError: If the input quaternions are not valid.
    """
    qx, qy, qz, qw = init_quat
    q1 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    qx, qy, qz, qw = rotate_quat
    q2 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    quat = q2 * q1

    return [quat.x, quat.y, quat.z, quat.w]


def alpha_blend_rgba(
    fg_image: Union[str, Image.Image, np.ndarray],
    bg_image: Union[str, Image.Image, np.ndarray],
    alpha: Union[str, Image.Image, np.ndarray] = None,
) -> Image.Image:
    """Alpha blends a foreground RGBA image over a background image.

    Args:
        fg_image: Foreground image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray (H, W, 3) or (H, W, 4).
        bg_image: Background image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray (H, W, 3) or (H, W, 4).
        alpha: Optional per-pixel alpha mask. If provided, can be file path (str),
            PIL Image (should be 'L' or '1' mode), or numpy ndarray (H, W), (H, W, 1).

    Returns:
        A PIL Image (RGBA) representing the alpha-blended result.
    """
    # Convert input images to PIL Images
    if isinstance(fg_image, str):
        fg_image = Image.open(fg_image)
    elif isinstance(fg_image, np.ndarray):
        fg_image = Image.fromarray(fg_image)

    if isinstance(bg_image, str):
        bg_image = Image.open(bg_image)
    elif isinstance(bg_image, np.ndarray):
        bg_image = Image.fromarray(bg_image)

    if fg_image.size != bg_image.size:
        raise ValueError(f"Image sizes not match {fg_image.size} v.s. {bg_image.size}.")

    if alpha is None:
        # Default: use foreground alpha channel for blending
        fg = fg_image.convert("RGBA")
        bg = bg_image.convert("RGBA")
        blended = Image.alpha_composite(bg, fg)
    else:
        # Use alpha channel for blending
        if alpha is not None and isinstance(alpha, str):
            alpha = Image.open(alpha)
        elif isinstance(alpha, np.ndarray):
            alpha = Image.fromarray(alpha)

        if fg_image.size != alpha.size:
            raise ValueError(f"Image sizes not match {fg_image.size} v.s. {alpha.size}.")
        blended = Image.composite(fg_image, bg_image, alpha)

    return blended


def alpha_blend_rgba_torch(
    fg_image: torch.Tensor,
    bg_image: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Alpha blends foreground over background using PyTorch with float tensors.

    Supports single image or batched inputs.

    Args:
        fg_image: (H, W, 3) or (B, H, W, 3); float32 in [0,1].
        bg_image: (H, W, 3) or (B, H, W, 3); float32 in [0,1].
        alpha:    (H, W), (H, W, 1), (B, H, W) or (B, H, W, 1); float32 in [0,1].

    Returns:
        Blended tensor with shape matching fg_image (3 channels, with batch if provided), float32 in [0,1].
    """
    device = fg_image.device

    # Move to same device
    bg_image = bg_image.to(device)
    alpha = alpha.to(device)

    # Track if input was batched
    batched = fg_image.ndim == 4

    # Normalize dimensions to (B, H, W, C)
    if fg_image.ndim == 3:
        fg_image = fg_image.unsqueeze(0)
    if bg_image.ndim == 3:
        bg_image = bg_image.unsqueeze(0)

    # Ensure alpha shape to (B, H, W, 1)
    if alpha.ndim == 2:  # (H, W)
        alpha = alpha.unsqueeze(0).unsqueeze(-1)
    elif alpha.ndim == 3:  # (B, H, W)
        alpha = alpha.unsqueeze(-1)
    elif alpha.ndim == 4 and alpha.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"Unsupported alpha shape: {alpha.shape}")

    # Type and range checks
    if (
        not torch.is_floating_point(fg_image)
        or not torch.is_floating_point(bg_image)
        or not torch.is_floating_point(alpha)
    ):
        raise TypeError("alpha_blend_rgba_torch expects float tensors in [0,1]")

    # Clamp to [0,1] to be safe
    fg_image = fg_image.clamp(0.0, 1.0)
    bg_image = bg_image.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)

    # Broadcast and blend
    blended = fg_image * alpha + bg_image * (1.0 - alpha)

    # Squeeze batch if original was unbatched
    if not batched:
        blended = blended.squeeze(0)

    return blended
