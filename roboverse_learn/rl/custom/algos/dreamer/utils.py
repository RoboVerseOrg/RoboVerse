import collections

import torch
import torch.nn as nn


class AverageScalarMeter:
    """Meter for tracking the moving average of a scalar value."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = collections.deque(maxlen=window_size)
        self.count = 0
        self.sum = 0.0

    def update(self, value):
        """Update the meter with a new value."""
        self.values.append(value)
        self.count += 1
        self.sum += value

    def get_mean(self):
        """Get the moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class RunningMeanStd(nn.Module):
    """Maintains a running mean and std for observation normalization."""

    def __init__(self, shape, clip=5.0, epsilon=1e-8):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x):
        """Update the running statistics with a batch of data."""
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update running statistics
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def forward(self, x, update=False):
        """Normalize the input and optionally update statistics."""
        shape = x.shape
        if update:
            with torch.no_grad():
                self.update(x)
        # Normalize and clip
        result = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        if self.clip:
            result = torch.clamp(result, -self.clip, self.clip)
        return result


class FreezeParameters:
    """Context manager to freeze parameters during optimization."""

    def __init__(self, modules):
        self.modules = modules
        self.parameters = []

    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = False
                self.parameters.append(param)

    def __exit__(self, *args):
        for param in self.parameters:
            param.requires_grad = True


class RequiresGrad:
    """Context manager to temporarily enable gradients."""

    def __init__(self, model):
        self._model = model
        self._prev_states = {}

    def __enter__(self):
        for module in self._model.modules():
            self._prev_states[module] = module.training
            module.train(True)

    def __exit__(self, *args):
        for module, state in self._prev_states.items():
            module.train(state)


def symlog(x):
    """Symmetric logarithm for stable training."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """Symmetric exponential, inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class Optimizer:
    """Wrapper for optimizer with gradient clipping."""

    def __init__(self, name, parameters, lr, eps, grad_clip, weight_decay=0.0, opt="adam", use_amp=False):
        self.name = name
        self.parameters = parameters
        self.lr = lr
        self.eps = eps
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.opt = opt
        self.use_amp = use_amp

        if opt == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        elif opt == "adamw":
            self.optimizer = torch.optim.AdamW(parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        elif opt == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {opt} not supported")

        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, parameters=None, retain_graph=False):
        metrics = {}
        parameters = parameters or self.parameters

        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass with optional AMP
        if self.use_amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward(retain_graph=retain_graph)

        # Compute gradient norm for logging
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        metrics[f"{self.name}_grad_norm"] = grad_norm.item()

        # Update with optional AMP
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        metrics[f"{self.name}_loss"] = loss.item()
        return metrics

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict() if self.use_amp else None}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.use_amp and state_dict["scaler"] is not None:
            self.scaler.load_state_dict(state_dict["scaler"])
