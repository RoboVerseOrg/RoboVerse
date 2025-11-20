from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from roboverse_learn.il.utils.common.normalizer import LinearNormalizer
from roboverse_learn.il.utils.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from diffusion_policy.model.diffusion.flow_net import SimpleFlowNet
from diffusion_policy.model.diffusion.action_ae import CNNActionEncoder, SimpleActionDecoder
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.flow_matchers import TorchFlowMatcher


class VITAImagePolicy(BaseImagePolicy):
    """
    Implementation of paper "VITA: Vision-to-action flow matching policy." arXiv preprint arXiv:2507.13231 (2025).
    VITA is noise-free and conditioning-free flow matching policy that directly flows from latent images to actions.
    VITA is highly performant and fast. The flow matching network and action decoder in VITA can be both simple MLP.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        # VITA specific params
        flow_net,
        flow_matcher: TorchFlowMatcher,
        decode_flow_latents=True,
        consistency_weight=1.0,
        enc_contrastive_weight=1e-4,
        flow_contrastive_weight=0.0,
        latent_dim=512,
        action_ae=None,
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps

        self.flow_matcher = flow_matcher
        self.action_ae = action_ae

        self.obs_encoder = obs_encoder
        self.obs_projector = nn.Linear(
            obs_feature_dim * n_obs_steps,
            latent_dim
        )

        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout
        )
        self.action_encoder = CNNActionEncoder(
            pred_horizon=horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, -1)
        obs_latents = self.obs_projector(nobs_features)

        # Encode actions
        action_latents = self.action_encoder(nactions)

        # Flow matching loss: obs_latents -> action_latents
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=action_latents,
            start=obs_latents,  # Use visual latents as the flow source
        )
        loss = flow_loss
        metrics['flow_loss'] = flow_loss.item()

        # Encoder contrastive loss
        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = action_latents.view(batch_size, -1)
            contrastive_loss = compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics['enc_contrastive_loss'] = contrastive_loss.item()

        # Flow latent decoding
        if self.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=obs_latents,  # Use visual latents as the flow source
                num_steps=self.num_sampling_steps
            )

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss += self.consistency_weight * consistency_loss
                metrics['consistency_loss'] = consistency_loss.item()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics['flow_contrastive_loss'] = contrastive_loss.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = F.l1_loss(actions_recon, nactions)
                metrics['flow_action_recon_loss'] = action_recon_loss.item()
                loss += self.action_ae["flow_recon_weight"] * action_recon_loss
        else:
            action_latents_pred = action_latents

        # Encoder reconstruction losses
        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(action_latents)
            action_recon_loss = F.l1_loss(actions_recon, nactions)
            metrics['enc_action_recon_loss'] = action_recon_loss.item()
            loss += self.action_ae["enc_recon_weight"] * action_recon_loss

        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, -1)
        obs_latents = self.obs_projector(nobs_features)

        # run sampling
        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=obs_latents,  # Use visual latents as the flow source
            return_traces=False
        )

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        # get action
        start = self.n_action_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


def compute_contrastive_loss(image_features, action_features, temperature=0.07):
    # Contrastive loss between image and action feautres (InfoNCE)
    # Can provide an additional boost on top of FLD and FLC

    # Normalize features
    batch_size = image_features.size(0)
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    # Compute similarity matrix
    logits = torch.matmul(image_features, action_features.T) / temperature

    # Symmetric contrastive loss (image-to-action + action-to-image)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2a = F.cross_entropy(logits, labels)
    loss_a2i = F.cross_entropy(logits.T, labels)

    return (loss_i2a + loss_a2i) / 2
