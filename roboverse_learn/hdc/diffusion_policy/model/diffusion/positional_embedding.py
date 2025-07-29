import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding Module.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimensionality of the positional embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T) where B is the batch size and T is the sequence length.

        Returns:
            torch.Tensor: Sinusoidal positional embeddings of shape (B, T, dim).
        """
        device = x.device
        B, T = x.shape
        half_dim = self.dim // 2

        # Compute scaling factors
        emb_factors = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -math.log(10000) / (half_dim - 1)
        )

        # Calculate positional embeddings
        emb = x[..., None] * emb_factors[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb
