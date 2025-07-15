import jax.numpy as jnp
import numpy as np
import torch


class TorchModel(torch.nn.Module):
    def __init__(self, inputs, num_classes=1):
        super(TorchModel, self).__init__()
        self.dense1 = torch.nn.Linear(inputs, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.dense3 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.dense1(x))
        x = torch.nn.functional.tanh(self.dense2(x))
        x = self.dense3(x)
        return x


class TorchPolicy:
    def __init__(self, model):
        self.model = model
        self.mean = None
        self.var = None

    def step(self, obs):
        if self.mean is not None and self.var is not None:
            obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        obs = torch.from_numpy(obs).float()
        action = self.model(obs).detach().numpy()
        return action

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, mean=None, var=None):
        self.model.load_state_dict(torch.load(path))
        if mean is not None and var is not None:
            self.mean = np.load(mean)[0]
            self.var = np.load(var)[0]

    def __call__(self, obs):
        return self.step(obs)

    def __str__(self):
        return "TorchPolicy"

    def __repr__(self):
        return "TorchPolicy"


def flax_to_torch(flax_model, torch_model):
    """Copy flax model weights to pytorch model."""
    for param_torch, param_flax in zip(torch_model.named_children(), flax_model.params["params"].keys()):
        if param_flax.startswith("Dense"):
            param_torch[1].weight.data = torch.from_numpy(
                np.array(jnp.transpose(flax_model.params["params"][param_flax]["kernel"]))
            )
            param_torch[1].bias.data = torch.from_numpy(np.array(flax_model.params["params"][param_flax]["bias"]))
    return torch_model


def load_from_flax_ckpt(flax_ckpt_folder, torch_model):
    from flax.training import checkpoints

    ckpt = checkpoints.restore_checkpoint(
        flax_ckpt_folder,
        target=None,
    )
    flax_params = ckpt["params"]

    for param_torch, param_flax in zip(torch_model.named_children(), flax_params.keys()):
        if param_flax.startswith("Dense"):
            param_torch[1].weight.data = torch.from_numpy(np.array(jnp.transpose(flax_params[param_flax]["kernel"])))
            param_torch[1].bias.data = torch.from_numpy(np.array(flax_params[param_flax]["bias"]))
    return torch_model
