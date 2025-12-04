from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from roboverse_learn.il.dp.models.diffusion.conditional_unet1d import ConditionalUnet1D
from roboverse_learn.il.dp.models.vision.multi_image_obs_encoder import MultiImageObsEncoder
import torch

from roboverse_learn.il.dp.policies.ddpm_image_policy import DiffusionDenoisingImagePolicy


class DiffusionUnetImagePolicy(DiffusionDenoisingImagePolicy):

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
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        scheduler_step_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.down_dims = tuple(down_dims)
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale

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
        return ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_predict_scale=self.cond_predict_scale,
        )
