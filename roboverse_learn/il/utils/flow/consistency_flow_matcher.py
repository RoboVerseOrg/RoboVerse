import torch
import numpy as np
import torchcfm.conditional_flow_matching as cfm

from roboverse_learn.il.utils.flow.base_flow_matcher import BaseFlowMatcher


class ConsistencyFlowMatcher(BaseFlowMatcher):
    def __init__(
        self,
        eps=1e-2,
        num_segments=2,
        boundary=1,
        delta=1e-3,
        alpha=1e-5,
        noise_scale=1.0,
        sigma_var=1.0,
        ode_tol=1e-5,
        num_sampling_steps=1,
    ):
        super().__init__()
        self.eps = eps
        self.num_segments = num_segments
        self.boundary = boundary
        self.delta = delta
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.sigma_var = sigma_var
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1. - t) * sigma_var
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, start=None, **kwargs):
        """Compute the CFM loss for training."""
        batch_size = target.shape[0]
        device = target.device

        if start is None:
            a0 = torch.randn_like(target)
        else:
            a0 = start
        t = torch.rand(batch_size, device=device) * (1 - self.eps) + self.eps
        r = torch.clamp(t + self.delta, max=1.0)

        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1 - t_expand) * a0
        xr = r_expand * target + (1 - r_expand) * a0

        segments = torch.linspace(0, 1, self.num_segments + 1, device=device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1 - segment_ends_expand) * a0

        vt = model(xt, t, **kwargs)
        vr = model(xr, r, **kwargs)
        vr = torch.nan_to_num(vr)

        ft = self._f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = self._threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, self.boundary, x_at_segment_ends)

        losses_f = torch.mean(torch.square(ft - fr).reshape(batch_size, -1), dim=-1)
        losses_v = self._masked_losses_v(vt, vr, self.boundary, segment_ends, t, batch_size)

        loss = torch.mean(losses_f + self.alpha * losses_v)
        return loss, {
            'loss': loss.item(),
            'flow_loss': torch.mean(losses_f).item(),
            'velocity_loss': torch.mean(losses_v).item()
        }

    def sample(self, model, shape, device, num_steps=None, return_traces=False, start=None, **kwargs):
        """Generate samples, optionally returning traces."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        if start is None:
            noise = torch.randn(shape, device=device)
        else:
            noise = start
        z = noise.detach().clone()
        dt = 1.0 / num_steps
        eps = self.eps

        if return_traces:
            traj_history = []
            vel_history = []

        for i in range(num_steps):
            num_t = i / num_steps * (1 - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            vt = model(z, t, **kwargs)
            sigma_t = self.sigma_t(num_t)
            if sigma_t > 0:
                pred_sigma = vt + (sigma_t**2) / (2 * (self.noise_scale**2) * ((1-num_t)**2)) * \
                    (0.5 * num_t * (1-num_t) * vt - 0.5 * (2-num_t) * z.detach().clone())
                z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma)
            else:
                z = z.detach().clone() + vt * dt

            if return_traces:
                traj_history.append(z.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return z, (traj_history, vel_history)
        return z

    def _f_euler(self, t_expand, segment_ends_expand, xt, vt):
        return xt + (segment_ends_expand - t_expand) * vt

    def _threshold_based_f_euler(self, t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
        if isinstance(threshold, int) and threshold == 0:
            return x_at_segment_ends
        less_than_threshold = t_expand < threshold
        return less_than_threshold * self._f_euler(t_expand, segment_ends_expand, xt, vt) + \
            (~less_than_threshold) * x_at_segment_ends

    def _masked_losses_v(self, vt, vr, threshold, segment_ends, t, batch_size):
        if isinstance(threshold, int) and threshold == 0:
            return torch.tensor(0.0, device=vt.device)
        t_expand = t.view(-1, 1, 1).repeat(1, vt.shape[1], vt.shape[2])
        less_than_threshold = t_expand < threshold
        far_from_segment_ends = (segment_ends - t) > 1.01 * self.delta
        far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, vt.shape[1], vt.shape[2])
        losses_v = torch.square(vt - vr)
        losses_v = less_than_threshold * far_from_segment_ends * losses_v
        return torch.mean(losses_v.reshape(batch_size, -1), dim=-1)
