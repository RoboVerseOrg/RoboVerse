import numbers
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from torch.func import functional_call, stack_module_state

from roboverse_learn.dexbench_rvrl.algos.tdmpc2 import math


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)


class Ensemble(nn.Module):
    def __init__(self, modules, device="cpu"):
        super().__init__()
        self.device = device
        self.params, self.buffers = stack_module_state(modules)
        for k, v in self.params.items():
            self.params[k] = nn.Parameter(v.to(device))
        for k, v in self.buffers.items():
            self.params[k] = v.to(device)
        self.module = deepcopy(modules[0]).to(device)
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, buffers, *args, **kwargs):
        return functional_call(self.module, (params, buffers), args, kwargs)

    def forward(self, *args, **kwargs):
        params = self.params.to_dict() if isinstance(self.params, TensorDictParams) else self.params
        return torch.vmap(self._call, (0, 0, None), randomness="different")(params, self.buffers, *args, **kwargs)

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr

    def safe_copy(self):
        new = self.__class__.__new__(self.__class__)
        nn.Module.__init__(new)

        new.device = self.device
        new.module = deepcopy(self.module).to(self.device)

        new._repr = getattr(self, "_repr", str(self.module))
        new._n = getattr(self, "_n", None)

        new.params = {k: v.detach().clone().to(self.device) for k, v in getattr(self, "params", {}).items()}
        new.buffers = {k: v.detach().clone().to(self.device) for k, v in getattr(self, "buffers", {}).items()}

        return new


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - 0.5


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0, act=None, device="cpu"):
        super().__init__(in_features, out_features, bias=bias, device=device)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0), act=nn.Mish(inplace=False)))
    mlp.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def rgb_enc(in_shape, model_cfg, img_h=None, img_w=None, act=True):
    encoder_type = model_cfg.get("encoder_type", "resnet")
    visual_feature_dim = model_cfg.get("visual_feature_dim", 512)
    img_h = img_h if img_h is not None else 256
    img_w = img_w if img_w is not None else 256

    if encoder_type == "resnet":
        encoder = torchvision.models.resnet18(pretrained=True)
        visual_feature_dim = encoder.fc.in_features
        del encoder.fc  # delete the original fully connected layer
        encoder.fc = nn.Identity()
        print("=> using resnet18 as visual encoder")
        return encoder, visual_feature_dim
    elif encoder_type == "cnn":
        cnn_cfg = model_cfg.get("cnn", {})
        stages = cnn_cfg.get("stages", 5)
        input_dim = in_shape[0]

        kernel_size = cnn_cfg.get("kernel_size", [4])
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * stages
        elif isinstance(kernel_size, list):
            if len(kernel_size) == 1:
                kernel_size = kernel_size * stages
            else:
                assert len(kernel_size) == stages, "kernel_size should be an int or list of length stages"

        stride = cnn_cfg.get("stride", [2])
        if isinstance(stride, int):
            stride = [stride] * stages
        elif isinstance(stride, list):
            if len(stride) == 1:
                stride = stride * stages
            else:
                assert len(stride) == stages, "stride should be an int or list of length stages"

        depth = cnn_cfg.get("depth", [32])
        if isinstance(depth, int):
            depth = [depth] * stages
        elif isinstance(depth, list):
            if len(depth) == 1:
                depth = depth * stages
            else:
                assert len(depth) == stages, "depth should be an int or list of length stages"

        visual_encoder = []
        visual_encoder.append(ShiftAug(pad=6))
        visual_encoder.append(PixelPreprocess())
        for i in range(stages):
            padding = (kernel_size[i] - 1) // stride[i]
            visual_encoder.append(
                nn.Conv2d(
                    input_dim,
                    depth[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding,
                    bias=False,
                )
            )
            visual_encoder.append(nn.ReLU())
            input_dim = depth[i]

        visual_encoder.append(nn.Flatten())
        visual_encoder = nn.Sequential(*visual_encoder)

        with torch.no_grad():
            test_data = torch.zeros(1, *in_shape)
            visual_feature_dim = visual_encoder(test_data).shape[1]
            out_dim = visual_encoder(test_data).shape[1]
            visual_encoder.add_module("out", NormedLinear(out_dim, visual_feature_dim))
            if act:
                visual_encoder.add_module("act", nn.Mish(inplace=False))
        print("=> using custom cnn as visual encoder")
        return visual_encoder, visual_feature_dim
    else:
        raise NotImplementedError


def enc(obs_shape, model_cfg, img_h=64, img_w=64, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    hidden_dim = model_cfg.get("hidden_dim", [256, 256, 256])
    feature_dim = 0
    latent_dim = model_cfg.get("latent_dim", 512)
    for k in obs_shape.keys():
        if "state" in k:
            out[k] = mlp(
                obs_shape[k][0] + model_cfg.get("task_dim", 96),
                hidden_dim,
                latent_dim,
                act=nn.Mish(inplace=False),
            )
            feature_dim += latent_dim
        elif "rgb" in k:
            out[k], visual_feature_dim = rgb_enc(obs_shape[k], model_cfg, img_h, img_w)
            feature_dim += visual_feature_dim
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out), feature_dim


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ["weight", "bias", "ln.weight", "ln.bias"]
    new_state_dict = dict()

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith("_Qs."):
            num = key[len("_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_Qs.params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val
            new_total_key = "_detach_Qs_params." + new_key
            new_state_dict[new_total_key] = val
        elif key.startswith("_target_Qs."):
            num = key[len("_target_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_target_Qs_params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
        for key in ("__batch_size", "__device"):
            new_key = prefix + "params." + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict.keys():
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if "Qs" in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert "Qs" not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
    new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]
    if "_action_masks" in target_state_dict:
        new_state_dict["_action_masks"] = target_state_dict["_action_masks"]

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(
        self, obs_shape, model_cfg, tau, episodic, multitask, tasks, action_dims, img_h=64, img_w=64, device="cpu"
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.model_cfg = model_cfg
        self.tau = tau
        self.episodic = episodic
        self.multitask = multitask
        self.tasks = tasks
        self.action_dims = action_dims
        self.img_h = img_h
        self.img_w = img_w
        self.num_q = model_cfg.get("num_q", 5)
        self.num_bins = model_cfg.get("num_bins", 101)
        self.vmin = model_cfg.get("vmin", -10)
        self.vmax = model_cfg.get("vmax", 10)
        self.bin_size = (
            model_cfg.get("bin_size", (self.vmax - self.vmin) / (self.num_bins - 1)) if self.num_bins > 1 else 0
        )
        if isinstance(action_dims, numbers.Integral):
            self.action_dim = action_dims
        else:
            assert len(action_dims) == len(tasks)
            self.action_dim = max(action_dims)
        self.task_dim = model_cfg.get("task_dim", 96)
        if multitask:
            self._task_emb = nn.Embedding(len(tasks), self.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(tasks), self.action_dim))
            for i in range(len(tasks)):
                self._action_masks[i, : action_dims[i]] = 1.0
        self._encoder, feature_dim = enc(self.obs_shape, model_cfg, img_h, img_w)
        self.latent_dim = model_cfg.get("latent_dim", 512)
        self._linear = mlp(
            feature_dim,
            model_cfg.get("feature_dim", [256, 256]),
            self.latent_dim,
            act=nn.Mish(inplace=False),
        )
        self._dynamics = mlp(
            self.latent_dim + self.action_dim + self.task_dim,
            model_cfg.get("dynamics_dim", [256, 256]),
            self.latent_dim,
            act=nn.Mish(inplace=False),
        )
        self._reward = mlp(
            self.latent_dim + self.action_dim + self.task_dim,
            model_cfg.get("reward_dim", [256, 256]),
            self.num_bins,
            act=nn.Mish(inplace=False),
        )
        self._termination = (
            mlp(
                self.latent_dim + self.task_dim,
                model_cfg.get("termination_dim", [256, 256]),
                1,
                act=nn.Mish(inplace=False),
            )
            if episodic
            else None
        )
        self._pi = mlp(
            self.latent_dim + self.task_dim,
            model_cfg.get("actor_dim", [256, 256]),
            2 * self.action_dim,
            act=nn.Mish(inplace=False),
        )
        self._Qs = Ensemble(
            [
                mlp(
                    self.latent_dim + self.action_dim + self.task_dim,
                    model_cfg.get("critic_dim", [256, 256]),
                    self.num_bins,
                    dropout=model_cfg.get("dropout", 0.0),
                    act=nn.Mish(inplace=False),
                )
                for _ in range(self.num_q)
            ],
            device,
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight, self._Qs.params["2.weight"]])

        self.register_buffer("log_std_min", torch.tensor(model_cfg.get("log_std_min", -10)))
        self.register_buffer("log_std_dif", torch.tensor(model_cfg.get("log_std_max", 2.0)) - self.log_std_min)
        self.init()

    def init(self):
        self._detach_Qs_params = TensorDictParams(
            TensorDict({k: v.detach() for k, v in self._Qs.params.items()}, batch_size=[self.num_q]), no_convert=True
        )
        self._target_Qs_params = TensorDictParams(
            TensorDict({k: v.detach().clone() for k, v in self._Qs.params.items()}, batch_size=[self.num_q]),
            no_convert=True,
        )

        self._detach_Qs = self._Qs.safe_copy()
        self._target_Qs = self._Qs.safe_copy()

        self._detach_Qs.params = self._detach_Qs_params
        self._target_Qs.params = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = ["Encoder", "Feature Mlp", "Dynamics", "Reward", "Termination", "Policy prior", "Q-functions"]
        for i, m in enumerate([
            self._encoder,
            self._linear,
            self._dynamics,
            self._reward,
            self._termination,
            self._pi,
            self._Qs,
        ]):
            if m == self._termination and not self.episodic:
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += f"Learnable parameters: {self.total_params:,}"
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        embeddings = []
        for key, value in obs.items():
            assert key in self._encoder, f"Encoder for observation type {key} not found."
            if "rgb" in key and value.ndim == 5:
                T, B, C, H, W = value.shape
                value = value.reshape(B * T, C, H, W)
                embeddings.append(self._encoder[key](value).reshape(T, B, -1))
            else:
                if self.multitask:
                    task_value = self.task_emb(value, task)
                else:
                    task_value = value
                embeddings.append(self._encoder[key](task_value))
        feature = torch.cat(embeddings, dim=-1)
        latent = self._linear(feature)
        return latent

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, task, unnormalized=False):
        """
        Predicts termination signal.
        """
        assert task is None
        if self.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.multitask:  # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.0,
            "entropy": -log_prob,
            "scaled_entropy": -scaled_log_prob,
        })
        return action, info

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        if self.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.num_bins, self.vmin, self.vmax)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
