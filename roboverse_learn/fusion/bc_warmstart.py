"""IL <-> RL weight bridge for state-based MLP policies.

The natural overlap between RoboVerse's IL and RL halves is a *state* (proprio)
policy that is a plain MLP: rsl-rl's actor is an ``MLPModel`` whose trunk lives
under ``mlp.*`` (a stack of ``nn.Linear`` + activation), and a behaviour-cloning
policy on the same observation/action space is the same shape. This module moves
weights between the two so an IL-pretrained actor can be RL-fine-tuned, and an
RL actor can seed BC.

It works purely on *ordered ``nn.Linear`` layers* (matched by shape, in order),
so it is robust to differences in attribute/state-dict naming between the BC and
RL implementations. Image/diffusion/ACT policies are out of scope here (no
weight-compatible MLP trunk); use the demo-collection bridge for those.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def _ordered_linear_keys(state_dict: OrderedDict[str, torch.Tensor]) -> list[str]:
    """Return ``*.weight`` keys of Linear layers, in declaration order.

    A key ``k.weight`` is treated as a Linear layer iff its tensor is 2-D and a
    matching ``k.bias`` (1-D) exists. Order follows the state-dict insertion
    order, which for a sequential MLP is the forward order.
    """
    keys = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.dim() == 2:
            bias = k[: -len(".weight")] + ".bias"
            if bias in state_dict and state_dict[bias].dim() == 1:
                keys.append(k)
    return keys


def _as_state_dict(obj) -> OrderedDict[str, torch.Tensor]:
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    return OrderedDict(sd)


def align_linear_layers(
    src: OrderedDict[str, torch.Tensor] | torch.nn.Module,
    dst: OrderedDict[str, torch.Tensor] | torch.nn.Module,
) -> list[tuple[str, str]]:
    """Pair src->dst Linear layers in order, asserting matching shapes.

    Returns a list of ``(src_weight_key, dst_weight_key)``. Raises ``ValueError``
    if the two have a different number of Linear layers or any layer's shape
    disagrees — failing fast rather than silently producing a broken policy.
    """
    src_sd, dst_sd = _as_state_dict(src), _as_state_dict(dst)
    src_keys, dst_keys = _ordered_linear_keys(src_sd), _ordered_linear_keys(dst_sd)
    if len(src_keys) != len(dst_keys):
        raise ValueError(
            f"Linear-layer count mismatch: src has {len(src_keys)} {src_keys}, "
            f"dst has {len(dst_keys)} {dst_keys}. The BC and RL actors must share "
            f"the same MLP depth (and obs/action dims) to warm-start."
        )
    pairs = []
    for sk, dk in zip(src_keys, dst_keys):
        if src_sd[sk].shape != dst_sd[dk].shape:
            raise ValueError(
                f"Shape mismatch for layer {sk} {tuple(src_sd[sk].shape)} -> "
                f"{dk} {tuple(dst_sd[dk].shape)}. Check obs_dim / action_dim / hidden_dims."
            )
        pairs.append((sk, dk))
    return pairs


def load_bc_into_actor_critic(
    actor_critic: torch.nn.Module,
    bc_source: OrderedDict[str, torch.Tensor] | torch.nn.Module,
    *,
    actor_attr: str = "actor",
    strict: bool = True,
) -> int:
    """Copy a BC MLP's Linear weights into an rsl-rl ``ActorCritic`` actor.

    Args:
        actor_critic: an rsl-rl ``ActorCritic`` (or any module exposing the actor
            under ``actor_attr``; falls back to the whole module if absent).
        bc_source: the BC policy module or its ``state_dict`` (a plain MLP).
        actor_attr: attribute holding the actor sub-module (default ``"actor"``).
        strict: if True, require equal layer counts/shapes (recommended). If
            False, copy only the leading layers that match and stop at the first
            mismatch (useful when the BC net shares only an encoder trunk).

    Returns:
        Number of Linear layers copied. The critic is left untouched (RL learns
        its own value function from the warm-started actor).
    """
    actor = getattr(actor_critic, actor_attr, actor_critic)
    actor_sd = actor.state_dict()
    bc_sd = _as_state_dict(bc_source)

    if strict:
        pairs = align_linear_layers(bc_sd, actor_sd)
    else:
        bc_keys, actor_keys = _ordered_linear_keys(bc_sd), _ordered_linear_keys(actor_sd)
        pairs = []
        for sk, dk in zip(bc_keys, actor_keys):
            if bc_sd[sk].shape != actor_sd[dk].shape:
                break
            pairs.append((sk, dk))

    new_sd = OrderedDict(actor_sd)
    for sk, dk in pairs:
        new_sd[dk] = bc_sd[sk].clone()
        bias_s, bias_d = sk[: -len(".weight")] + ".bias", dk[: -len(".weight")] + ".bias"
        new_sd[bias_d] = bc_sd[bias_s].clone()
    actor.load_state_dict(new_sd)
    return len(pairs)


def extract_actor_mlp_state_dict(
    actor_source: OrderedDict[str, torch.Tensor] | torch.nn.Module,
    *,
    actor_attr: str = "actor",
) -> OrderedDict[str, torch.Tensor]:
    """Pull the actor's MLP Linear layers out as a standalone, BC-loadable dict.

    Keys are kept *relative to the actor and unchanged* (e.g. ``mlp.0.weight``,
    ``mlp.2.weight`` for a ``Linear, activation, Linear, ...`` ``nn.Sequential``),
    so the dict loads cleanly into a freshly-built MLP of the *same* structure —
    the natural way to seed a BC policy from a trained RL actor. Only the ordered
    ``nn.Linear`` weight/bias tensors are included (no normalizer / distribution
    buffers).
    """
    src = getattr(actor_source, actor_attr, actor_source) if hasattr(actor_source, actor_attr) else actor_source
    sd = _as_state_dict(src)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for wk in _ordered_linear_keys(sd):
        bk = wk[: -len(".weight")] + ".bias"
        out[wk] = sd[wk].clone()
        out[bk] = sd[bk].clone()
    return out
