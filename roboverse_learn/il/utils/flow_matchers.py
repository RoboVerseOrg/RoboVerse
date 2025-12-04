import torch
import numpy as np
import torchcfm.conditional_flow_matching as cfm


class BaseFlowMatcher():
    def compute_loss(self, model, target, **kwargs):
        raise NotImplementedError

    def sample(self, model, shape, device, num_steps, return_traces=False, **kwargs):
        raise NotImplementedError


class TorchFlowMatcher(BaseFlowMatcher):
    def __init__(self, fm, num_sampling_steps=6):
        """
        Flow matcher wrapper for torchcfm.
        """
        super().__init__()
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, start=None, **kwargs):
        """
        Compute the training loss using the flow matcher.

        Args:
            model: The flow network (e.g., ConditionalUnet1D or FlowTransformer).
            target: Target actions for training.

        Returns:
            Tuple of (loss tensor, dictionary of metrics).
        """
        if start is None:
            x0 = torch.randn_like(target)
        else:
            x0 = start
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep, **kwargs)
        loss = torch.mean((vt - ut) ** 2)
        # Use L1 loss
        # loss = torch.mean(torch.abs(vt - ut))
        return loss, {'loss': loss.item()}

    def sample(self, model, shape, device, num_steps=None, return_traces=False, start=None, **kwargs):
        """
        Generate samples using the flow network.

        Args:
            model: The flow network.
            shape: Shape of the output tensor (batch_size, pred_horizon, action_dim).
            return_traces: If True, return trajectory and velocity histories.
            num_steps: Number of sampling steps. If None, use self.num_sampling_steps.
            start [IMPORTANT]: Optional flow source. If None, start from standard normal noise.

        Returns:
            Sampled actions, or (actions, (traj_history, vel_history)) if return_traces is True.
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps
        if start is None:
            x = torch.randn(shape, device=device)
        else:
            x = start
        dt = 1.0 / num_steps

        if return_traces:
            # Add the initial state
            traj_history = [x]
            vel_history = [np.zeros_like(x.cpu())]

        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=x.device) * (t / num_steps)
            vt = model(x, timestep, **kwargs)
            x = x + vt * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


class ConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(cfm.ConditionalFlowMatcher(**kwargs), num_sampling_steps)


class TargetConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(cfm.TargetConditionalFlowMatcher(**kwargs), num_sampling_steps)


class SchrodingerBridgeConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(cfm.SchrodingerBridgeConditionalFlowMatcher(**kwargs), num_sampling_steps)


class ExactOptimalTransportConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(cfm.ExactOptimalTransportConditionalFlowMatcher(**kwargs), num_sampling_steps)
