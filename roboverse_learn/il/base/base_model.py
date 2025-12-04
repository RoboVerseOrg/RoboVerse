import torch


class BaseModel:
    def __init__(self):
        pass

    def forward(self, obs):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_loss(self, batch):
        raise NotImplementedError("Subclasses should implement this method.")
