import torch
from functools import partial

from roboverse_learn.il.utils.flow.base_flow_matcher import BaseFlowMatcher


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss.
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


def dispersive_loss(z, tau=1.0):
    """
    Dispersive Loss.
    """
    if z.shape[0] <= 1:
        return 0.0
    dist_matrix = torch.cdist(z, z, p=2) ** 2
    # Normalize to prevent overflow/underflow
    dist_matrix = dist_matrix / (torch.max(dist_matrix).detach() + 1e-8)
    exp_term = torch.exp(-dist_matrix / tau)
    mean_exp = torch.mean(exp_term)
    loss = torch.log(mean_exp)
    return loss


class MeanFlowMatcher(BaseFlowMatcher):
    def __init__(
        self,
        flow_ratio=0.5,
        time_dist_mu=-0.4,
        time_dist_sigma=1.0,
        adaptive_loss_gamma=0.5,
        dispersive_loss_tau=1.0,
        dispersive_loss_weight=0.0,
        cfg_scale=0.5,
        use_imf=False,
        **kwargs,
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.adaptive_loss_gamma = adaptive_loss_gamma
        self.dispersive_loss_tau = dispersive_loss_tau
        self.dispersive_loss_weight = dispersive_loss_weight
        self.cfg_scale = cfg_scale
        self.use_imf = use_imf

    def sample_t_r(self, batch_size, device):
        """
        Samples t and r from a log-normal distribution.
        """
        # Log-normal distribution
        normal_samples = (
            torch.randn(batch_size, 2, device=device) * self.time_dist_sigma
            + self.time_dist_mu
        )
        samples = torch.sigmoid(normal_samples)

        # t = max, r = min
        t = torch.max(samples, dim=1)[0]
        r = torch.min(samples, dim=1)[0]

        # Set r=t for a portion of the batch
        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r[indices] = t[indices]

        return t, r

    def compute_loss(self, model, target, start=None, **kwargs):
        """
        Compute the MeanFlow + Dispersive loss.
        Assumes `model` returns (prediction, internal_features_list).
        """
        if start is None:
            raise ValueError("MeanFlowMatcher requires a 'start' (vision latent) tensor.")

        x1 = target
        x0 = start
        batch_size = x0.shape[0]
        device = x0.device

        # Sample t and r
        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(-1, *([1] * (x0.dim() - 1)))
        r_ = r.view(-1, *([1] * (x0.dim() - 1)))

        # Define path and sample z_t
        z_t = (1 - t_) * x1 + t_ * x0

        # Ground-truth instantaneous velocity v = dx/dt
        v = x0 - x1

        def pred_meanflow(z_in, t_in, r_in):
            return model(x=z_in, timestep=t_in, r=r_in, **kwargs)

        if self.use_imf:
            with torch.no_grad():
                v_net, _ = pred_meanflow(z_t, t, t)
            dz_tangent = v_net
        else:
            dz_tangent = v

        # JVP inputs
        primals = (z_t, t, r)
        tangents = (dz_tangent, torch.ones_like(t), torch.zeros_like(r))
        pred, dudt = torch.autograd.functional.jvp(pred_meanflow, primals, tangents, create_graph=True)
        predicted_mean_vel, internal_features = pred

        # dudt[0] is the JVP for the velocity output
        u_tgt = v - (t_ - r_) * dudt[0]

        # MeanFlow Loss
        error = predicted_mean_vel - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error, gamma=self.adaptive_loss_gamma)

        loss = meanflow_loss
        metrics = {'meanflow_loss': meanflow_loss.item()}

        # Dispersive Loss
        if self.dispersive_loss_weight > 0 and internal_features is not None:
            dis_loss_total = 0.0
            # internal_features is a list of tensors from the network's hidden layers
            for features in internal_features:
                dis_loss_total += dispersive_loss(features, tau=self.dispersive_loss_tau)

            metrics['dispersive_loss'] = dis_loss_total.item()
            loss += self.dispersive_loss_weight * dis_loss_total

        metrics['loss'] = loss.item()
        return loss, metrics

    def sample(self, model, shape, device, num_steps=None, return_traces=False, start=None, **kwargs):
        """
        Generate samples in 1-NFE using MeanFlow.
        """
        if start is None:
            raise ValueError("MeanFlowMatcher requires a 'start' (vision latent) tensor for sampling.")

        x_source = start
        batch_size = x_source.shape[0]

        t = torch.ones(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)

        # Model predicts u(x0, 1, 0) which equals (x0 - x1)
        mean_velocity, _ = model(x_source, t, r=r, **kwargs)

        # x1 = x0 - u(x0, 1, 0)
        x_target = x_source - mean_velocity

        if return_traces:
            traj_history = [x_source.detach().clone().cpu(), x_target.detach().clone().cpu()]
            vel_history = [torch.zeros_like(x_source).cpu(), mean_velocity.detach().clone().cpu()]
            return x_target, (traj_history, vel_history)

        return x_target
