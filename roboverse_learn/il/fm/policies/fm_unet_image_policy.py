from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce

from roboverse_learn.il.dp.models.diffusion.conditional_unet1d import ConditionalUnet1D
from roboverse_learn.il.dp.models.diffusion.mask_generator import LowdimMaskGenerator
from roboverse_learn.il.dp.models.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.base.base_image_policy import BaseImagePolicy


class FlowMatchingUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # Instantiate the Unet model
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
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
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps  # number of inference steps for sampling

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model

        # Sample noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        time_steps = torch.linspace(0, 1.0, self.num_inference_steps + 1)
        for i in range(self.num_inference_steps):

            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. compute next sample
            # Midpoint ODE Solver x_end = x_t + (t_end - t_start) * f(x_mid, t_mid)
            t_start = time_steps[i].view(1).expand(trajectory.shape[0])
            t_start = t_start.to(self.device)
            t_end = time_steps[i + 1].view(1).expand(trajectory.shape[0])
            t_end = t_end.to(self.device)
            t_start_expa = t_start.view(-1, 1, 1)
            t_end_expa = t_end.view(-1, 1, 1)
            t_mid = t_start + (t_end - t_start) / 2
            trajectory_mid = (
                trajectory
                + model(trajectory, t_start, local_cond=local_cond, global_cond=global_cond)
                * (t_end_expa - t_start_expa)
                / 2
            )
            trajectory = trajectory + (t_end_expa - t_start_expa) * model(
                trajectory_mid, t_mid, local_cond=local_cond, global_cond=global_cond
            )
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # print("!!obs_dict", obs_dict["head_cam"].shape)
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # print("!!nobs", nobs["head_cam"].shape)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None  # used in p(A_t/O_t) policy
        if self.obs_as_global_cond:  # p(A_t/O_t) policy
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:  # p(A_t, O_t)
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])  # batch_size * horizon * action_dim
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # compute loss mask
        loss_mask = ~condition_mask

        # Sample timesteps
        timesteps = torch.rand(trajectory.shape[0], device=trajectory.device)
        timesteps_exp = timesteps.view(-1, 1, 1)
        # Sample middle trajectory
        trajectory_inter = (1 - timesteps_exp) * noise + timesteps_exp * trajectory
        # Ture flow vector
        vertor_fm_true = trajectory - noise
        # Predict the noise residual
        vector_fm_pred = self.model(trajectory_inter, timesteps, local_cond=local_cond, global_cond=global_cond)
        # Compute the loss
        loss = F.mse_loss(vector_fm_pred, vertor_fm_true, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
