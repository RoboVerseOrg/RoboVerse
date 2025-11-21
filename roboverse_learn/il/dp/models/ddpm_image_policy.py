from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from einops import reduce

from roboverse_learn.il.utils.common.normalizer import LinearNormalizer
from roboverse_learn.il.utils.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionDenoisingImagePolicy(BaseImagePolicy):

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
        scheduler_step_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()

        # Parse action / observation sizes.
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError(f"Action must be 1D, got shape {action_shape}")
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Configure model inputs.
        input_dim = action_dim + obs_feature_dim
        global_cond_dim: Optional[int] = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # Instantiate denoising backbone.
        model = self.build_denoising_model(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.scheduler_step_kwargs = dict(scheduler_step_kwargs or {})

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(obs_dict)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Set diffusion steps.
        scheduler.set_timesteps(self.num_inference_steps)

        step_kwargs = dict(self.scheduler_step_kwargs)
        step_kwargs.update(kwargs)

        for t in scheduler.timesteps:
            # 1. Apply conditioning.
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. Predict model output.
            t = t.to(device=trajectory.device)
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. Compute previous sample x_t -> x_{t-1}.
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **step_kwargs,
            ).prev_sample

        # Enforce conditioning on the final sample.
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if "past_action" in obs_dict:
            raise NotImplementedError("Past actions are not supported yet.")

        # Normalize input observations.
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, _ = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # Condition through global feature.
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Condition through inpainting.
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask is not supported yet.")

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    def build_denoising_model(
        self,
        input_dim: int,
        global_cond_dim: Optional[int],
        diffusion_step_embed_dim: int,
    ):
        raise NotImplementedError("Subclasses must provide a denoising backbone.")
