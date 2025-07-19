import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, dim, demean=True, destd=True, clip=5.0):
        super().__init__()
        self.dim = dim
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.zeros(dim))
        self.register_buffer("std", torch.zeros(dim))

    def update(self, x):
        var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False)
        m = x.shape[0]
        w = self.n.to(x.dtype) / (m + self.n).to(x.dtype)
        self.var[:] = (
            w * self.var + (1 - w) * var_x + w * (1 - w) * (mean_x - self.mean).pow(2)
        )
        self.mean[:] = w * self.mean + (1 - w) * mean_x
        self.std[:] = torch.sqrt(self.var)
        self.n += m

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.update(x)
        if self.n > 0:
            if self.demean:
                x = x - self.mean
            if self.destd:
                x = x / (self.std + 1e-8)
            if self.clip:
                x = torch.clamp(x, -self.clip, self.clip)
        return x
