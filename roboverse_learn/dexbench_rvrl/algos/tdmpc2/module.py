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
from functorch import combine_state_for_ensemble

from roboverse_learn.dexbench_rvrl.algos.tdmpc2 import math

class LayerNormChannelLast(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

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
    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)
    
    def __repr__(self):
        return f"Ensemble of: {self._repr}"


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

    if encoder_type == "cnn":
        cnn_cfg = model_cfg.get("cnn", {})
        stages = cnn_cfg.get("stages", 5)
        input_dim = in_shape[0]
        # visual_feature_dim = model_cfg.get("visual_feature_dim", 1024)
        visual_feature_dim = model_cfg.get("latent_dim", 512)

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
            conv_shapes = []
            test_data = torch.zeros(1, *in_shape)
            conv_shapes.append(test_data[0].shape)
            for idx, layer in enumerate(visual_encoder):
                test_data = layer(test_data)
                if isinstance(layer, nn.Conv2d):
                    conv_shapes.append(test_data[0].shape)
            out_dim = test_data.shape[1]
            if act:
                visual_encoder.add_module("act", nn.Mish(inplace=False))
            visual_encoder.add_module("out", NormedLinear(out_dim, visual_feature_dim))
        print("=> using custom cnn as visual encoder")
        return visual_encoder, visual_feature_dim, conv_shapes
    else:
        raise NotImplementedError


def enc(obs_shape, model_cfg, img_h=64, img_w=64, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    hidden_dim = model_cfg.get("hidden_dim", [256, 256, 256])
    feature_dim = 0
    latent_dim = model_cfg.get("latent_dim", 512)
    feature_dim_dict = {}
    conv_shapes = None
    for k in obs_shape.keys():
        if "state" in k:
            out[k] = mlp(
                obs_shape[k][0] + model_cfg.get("task_dim", 96),
                hidden_dim,
                latent_dim,
                act=nn.Mish(inplace=False),
            )
            feature_dim += latent_dim
            feature_dim_dict[k] = latent_dim
        elif "rgb" in k:
            out[k], visual_feature_dim, conv_shapes = rgb_enc(obs_shape[k], model_cfg, img_h, img_w)
            feature_dim += visual_feature_dim
            feature_dim_dict[k] = visual_feature_dim
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out), feature_dim, feature_dim_dict, conv_shapes

class Decoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_dim_dict,
        model_cfg,
        decode_shapes=None,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_key = list(obs_shape.keys())
        self.state_key = [key for key in obs_shape.keys() if "state" in key]
        assert len(self.state_key) < 2, "only support one state observation"
        self.state_shape = sum([sum(obs_shape[key]) for key in self.state_key])
        self.state_feature_dim = sum([feature_dim_dict[key] for key in self.state_key])
        self.feature_dim_dict = feature_dim_dict
        if "rgb" in obs_shape.keys():
            assert decode_shapes is not None, "decode_shapes must be provided for rgb observation"
            decode_shapes.reverse()
            self.decode_shapes = decode_shapes
            self.img_h = self.decode_shapes[-1][1]
            self.img_w = self.decode_shapes[-1][2]
            self.img_key = [key for key in obs_shape.keys() if "rgb" in key]
            assert len(self.img_key) == 1, "only support one rgb observation, shape 3xhxw"
            self.num_channel = obs_shape[self.img_key[0]][0]
            self.visual_feature_dim = feature_dim_dict[self.img_key[0]]
            self.input_latent_dim = self.decode_shapes[0][0] * self.decode_shapes[0][1] * self.decode_shapes[0][2]
            self.linear_layer = nn.Linear(self.visual_feature_dim, self.input_latent_dim)
            self.num_img = len(self.img_key)

            cnn_cfg = model_cfg.get("cnn", {})
            stages = cnn_cfg.get("stages", 5)
            kernel_size = cnn_cfg.get("kernel_size", 4)
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size] * stages
            elif isinstance(kernel_size, list):
                if len(kernel_size) == 1:
                    kernel_size = kernel_size * stages
                else:
                    assert len(kernel_size) == stages, "kernel_size should be an int or list of length stages"
            kernel_size.reverse()

            stride = cnn_cfg.get("stride", 2)
            if isinstance(stride, int):
                stride = [stride] * stages
            elif isinstance(stride, list):
                if len(stride) == 1:
                    stride = stride * stages
                else:
                    assert len(stride) == stages, "stride should be an int or list of length stages"
            stride.reverse()

            depth = cnn_cfg.get("depth", [32])
            if isinstance(depth, int):
                depth = depth * stages
            elif isinstance(depth, list):
                if len(depth) == 1:
                    depth = depth * stages
                else:
                    assert len(depth) == stages, "depth should be an int or list of length stages"
            depth.reverse()

            input_dim = depth[0]
            self.visual_decoder = []
            for i in range(stages - 1):
                pad_h, outpad_h = self.calc_same_pad(
                    k=kernel_size[i], s=stride[i], d=1, in_=self.decode_shapes[i][1], out_=self.decode_shapes[i + 1][1]
                )
                pad_w, outpad_w = self.calc_same_pad(
                    k=kernel_size[i], s=stride[i], d=1, in_=self.decode_shapes[i][2], out_=self.decode_shapes[i + 1][2]
                )
                self.visual_decoder.append(
                    nn.ConvTranspose2d(
                        input_dim,
                        depth[i + 1],
                        kernel_size[i],
                        stride[i],
                        padding=(pad_h, pad_w),
                        output_padding=(outpad_h, outpad_w),
                        bias=False,
                    )
                )
                self.visual_decoder.append(LayerNormChannelLast(depth[i + 1], eps=1e-3))
                self.visual_decoder.append(nn.SiLU())
                input_dim = depth[i + 1]
            pad_h, outpad_h = self.calc_same_pad(
                k=kernel_size[stages - 1],
                s=stride[stages - 1],
                d=1,
                in_=self.decode_shapes[-2][1],
                out_=self.decode_shapes[-1][1],
            )
            pad_w, outpad_w = self.calc_same_pad(
                k=kernel_size[stages - 1],
                s=stride[stages - 1],
                d=1,
                in_=self.decode_shapes[-2][2],
                out_=self.decode_shapes[-1][2],
            )
            self.visual_decoder.append(
                nn.ConvTranspose2d(
                    depth[-1],
                    self.num_channel,
                    kernel_size[stages - 1],
                    stride[stages - 1],
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=True,
                )
            )
            self.visual_decoder = nn.Sequential(*self.visual_decoder)
        if model_cfg is None:
            hidden_dim = [256, 256, 256]
        else:
            hidden_dim = model_cfg.get("hidden_dim")
        self.mlp_decoder = []
        input_dim = self.state_feature_dim
        for hdim in hidden_dim:
            self.mlp_decoder.append(nn.Linear(input_dim, hdim, bias=False))
            self.mlp_decoder.append(nn.LayerNorm(hdim, eps=1e-3))
            self.mlp_decoder.append(nn.SiLU())
            input_dim = hdim
        self.mlp_decoder.append(nn.Linear(input_dim, np.prod(self.state_shape), bias=True))
        self.mlp_decoder = nn.Sequential(*self.mlp_decoder)

    def calc_same_pad(self, k, s, d, in_, out_):
        pad = (k - 1) // s
        val = (in_ - 1) * s - 2 * pad + d * (k - 1) + 1
        outpad = out_ - val
        return pad, outpad

    def forward(self, embedding):
        reconstructed_obs = {}
        start_idx = 0
        for key in self.obs_key:
            feature_dim = self.feature_dim_dict[key]
            feature = embedding[key]
            input_shape = feature.shape
            if key in self.state_key:
                if len(input_shape) > 2:
                    feature = feature.view(-1, feature_dim)
                    reconstructed_vector_obs = self.mlp_decoder(feature).unflatten(0, input_shape[:2])
                reconstructed_obs[key] = reconstructed_vector_obs
            elif key in self.img_key:
                if len(input_shape) > 2:
                    feature = feature.view(-1, feature_dim)
                N = feature.shape[0]
                feature = self.linear_layer(feature)
                feature = feature.view(
                    N,
                    -1,
                    self.decode_shapes[0][1],
                    self.decode_shapes[0][2],
                )
                reconstructed_visual_obs = self.visual_decoder(feature)
                reconstructed_visual_obs = reconstructed_visual_obs.unflatten(0, input_shape[:2]) + 0.5
                reconstructed_obs[key] = reconstructed_visual_obs
            start_idx += feature_dim

        return reconstructed_obs

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
        self._encoder, feature_dim, self.feature_dim_dict, conv_shapes = enc(self.obs_shape, model_cfg, img_h, img_w)
        self.use_decode = model_cfg.get("decode", False)
        if self.use_decode:
            self._decoder = Decoder(
                self.obs_shape,
                self.feature_dim_dict,
                model_cfg,
                conv_shapes,
            )
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
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

        self.log_std_min = torch.tensor(model_cfg.get("log_std_min", -10))
        self.log_std_dif = torch.tensor(model_cfg.get("log_std_max", 2.0)) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs) 
        return self
    
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

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self
    
    def track_q_grad(self, mode=True):
        """
        Enable or disable gradient tracking for the Q-networks used for computing target values.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
         for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
             p_target.data.lerp_(p.data, self.tau)

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
        feature_dict = {}
        for key, value in obs.items():
            assert key in self._encoder, f"Encoder for observation type {key} not found."
            if "rgb" in key and value.ndim == 5:
                T, B, C, H, W = value.shape
                value = value.reshape(B * T, C, H, W)
                feature = self._encoder[key](value).reshape(T, B, -1)
                embeddings.append(feature)
                feature_dict[key] = feature.clone()
            else:
                if self.multitask:
                    task_value = self.task_emb(value, task)
                else:
                    task_value = value
                feature = self._encoder[key](task_value)
                embeddings.append(feature)
                feature_dict[key] = feature.clone()
        feature = torch.stack(embeddings, dim=0).mean(dim=0)
        # feature = torch.cat(embeddings, dim=-1)
        # latent = self._linear(feature)
        latent = feature
        return latent, feature_dict

    def decode(self, feature):
        """
        Decodes latent representation back to observation space.
        """
        assert self.decode, "Decoder not initialized. Set decode=True in model_cfg to enable decoding."
        return self._decoder(feature)
        

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

    def Q(self, z, a, task, return_type="min", target=False):
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
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.num_bins, self.vmin, self.vmax), math.two_hot_inv(Q2, self.num_bins, self.vmin, self.vmax)
        return torch.min(Q1, Q2) if return_type == "min" else 0.5 * (Q1 + Q2)
