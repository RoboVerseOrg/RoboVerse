import torch
import torch.nn as nn

class AR1Prior(nn.Module):
    def __init__(self):
        super(AR1Prior, self).__init__()
        # Initializing phi as a learnable parameter
        # self.phi = nn.Parameter(torch.tensor(0.5))
        self.phi = 0.95

    def forward(self, series):
        # Calculate the likelihood of the series given phi
        # Ignoring the first term since it doesn't have a previous term
        error = series[1:] - self.phi * series[:-1]
        log_likelihood = -0.5 * torch.sum(error**2)
        return log_likelihood
