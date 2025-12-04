from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from roboverse_learn.il.dp.models.diffusion.flow_net import FlowTransformer
from roboverse_learn.il.dp.models.vision.multi_image_obs_encoder import MultiImageObsEncoder
import torch

from roboverse_learn.il.dp.policies.ddpm_image_policy import DiffusionDenoisingImagePolicy


class DiffusionDiTImagePolicy(DiffusionDenoisingImagePolicy):

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        scheduler_step_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(obs_dict)

    def build_denoising_model(
        self,
        input_dim: int,
        global_cond_dim: Optional[int],
        diffusion_step_embed_dim: int,
    ):
        return FlowTransformer(
            input_dim=input_dim,
            condition_dim=global_cond_dim,
            hidden_dim=self.hidden_dim,
            output_dim=input_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            time_embed_dim=diffusion_step_embed_dim,
        )
