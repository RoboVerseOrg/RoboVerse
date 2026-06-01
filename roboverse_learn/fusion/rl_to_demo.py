"""RL -> IL bridge: turn a trained rsl-rl policy into RoboVerse demonstrations.

Rolls out a policy in a RoboVerse task and writes each successful episode in the
canonical demo layout that the imitation-learning pipeline already ingests::

    <out_dir>/success/demo_0000/{metadata.json, metadata.pkl, rgb.mp4, status.txt}

so the standard follow-up is unchanged::

    python roboverse_learn/il/data2zarr_dp.py --task_name <name> \
        --metadata_dir <out_dir>/success --expert_data_num <N>
    python roboverse_learn/il/train.py  ...

The rollout mirrors the proven ``scripts/advanced/collect_demo.py`` loop
(``state_tensor_to_nested`` -> per-env dict states -> :func:`save_demo`) but
swaps trajectory *replay* for *policy inference*. Episode segmentation accounts
for ``RLTaskEnv.step`` auto-resetting done envs in-place: on a done step the
freshly-read state already belongs to the *next* episode, so it is split off as
that episode's first frame (see :func:`_finalise_episode`).
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

from loguru import logger as log


def _build_env(task: str, robot: str, sim: str, num_envs: int, device, camera_cfg):
    """Build a RoboVerse task env with a single RGB(+depth) camera for rendering.

    Reuses the same task-discovery + scenario path as
    ``roboverse_learn/rl/rsl_rl/ppo.py:make_roboverse_env``, but attaches a
    camera (RL training runs camera-less) so ``save_demo`` can write ``rgb.mp4``.
    """
    from metasim.task.registry import _discover_task_modules, get_task_class

    _discover_task_modules()
    task_cls = get_task_class(task)
    cameras = [camera_cfg] if camera_cfg is not None else []
    scenario = task_cls.scenario.update(
        robots=[robot], simulator=sim, num_envs=num_envs, headless=True, cameras=cameras
    )
    return task_cls(scenario=scenario, device=device)


def _default_camera():
    """Front 3/4 PinholeCamera matching collect_demo's default (rgb+depth)."""
    from metasim.scenario.cameras import PinholeCameraCfg

    return PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))


def _mlp_from_linear_state_dict(sd, device, activation=None):
    """Rebuild a deterministic actor MLP from an ordered ``mlp.*`` state dict.

    Matches rsl-rl's ``MLPModel.mlp`` (``Linear, ELU, Linear, ELU, ..., Linear``)
    and returns it in eval mode. Only the ``mlp.{i}.weight/bias`` Linear tensors
    are used; distribution/normalizer buffers are ignored (deterministic mean
    action, which is what we want for clean demos).
    """
    import re

    import torch
    import torch.nn as nn

    activation = activation or nn.ELU
    idxs = sorted({int(m.group(1)) for k in sd if (m := re.match(r"mlp\.(\d+)\.weight$", k))})
    if not idxs:
        return None
    linears = [nn.Linear(sd[f"mlp.{i}.weight"].shape[1], sd[f"mlp.{i}.weight"].shape[0]) for i in idxs]
    layers: list[nn.Module] = []
    for j, lin in enumerate(linears):
        layers.append(lin)
        if j < len(linears) - 1:
            layers.append(activation())
    mlp = nn.Sequential(*layers)
    # state_dict keys for the Sequential are 0.weight, 2.weight, ... (ELU has none)
    remap = {}
    seq_linear_pos = [p for p, m in enumerate(layers) if isinstance(m, nn.Linear)]
    for pos, i in zip(seq_linear_pos, idxs):
        remap[f"{pos}.weight"] = sd[f"mlp.{i}.weight"]
        remap[f"{pos}.bias"] = sd[f"mlp.{i}.bias"]
    mlp.load_state_dict(remap)
    mlp.to(device).eval()
    return mlp


def _load_inference_policy(env_wrapper, train_cfg, checkpoint: str, device):
    """Return a callable ``policy(actor_obs) -> actions`` from a checkpoint.

    Handles every checkpoint format used in this repo:
    * a TorchScript export (``policy.pt``) -> ``torch.jit.load`` (standalone);
    * a raw actor/MLP state dict — either ``{"actor_state_dict": {...}}`` (the
      cross-sim-eval save format) or a bare ``mlp.*`` dict -> rebuild a
      deterministic MLP (no runner / train_cfg needed);
    * an rsl-rl runner checkpoint (``model_*.pt`` with ``model_state_dict``) ->
      rebuild ``OnPolicyRunner`` and ``runner.load``.
    """
    import torch

    # 1) TorchScript module: directly callable.
    try:
        jit = torch.jit.load(checkpoint, map_location=device)
        jit.eval()
        log.info(f"[rl_to_demo] loaded TorchScript policy from {checkpoint}")
        return lambda obs: jit(obs)
    except (RuntimeError, ValueError):
        pass

    # 2) Raw actor / MLP state dict.
    blob = torch.load(checkpoint, map_location=device)
    actor_sd = None
    if isinstance(blob, dict):
        if "actor_state_dict" in blob:
            actor_sd = blob["actor_state_dict"]
        elif any(k.startswith("mlp.") for k in blob):
            actor_sd = blob
    if actor_sd is not None:
        mlp = _mlp_from_linear_state_dict(actor_sd, device)
        if mlp is not None:
            log.info(f"[rl_to_demo] loaded actor MLP ({sum(1 for k in actor_sd if k.endswith('.weight'))} layers) from {checkpoint}")
            return lambda obs: mlp(obs if isinstance(obs, torch.Tensor) else obs["policy"])

    # 3) rsl-rl runner checkpoint.
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env=env_wrapper, train_cfg=train_cfg, log_dir=None, device=str(device))
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=str(device))
    log.info(f"[rl_to_demo] loaded rsl-rl runner checkpoint from {checkpoint}")
    return policy


def _finalise_episode(cache: list, reset_frame) -> tuple[list, list]:
    """Split a done env's cache into (completed_episode, next_episode_seed).

    ``reset_frame`` is the post-auto-reset state read on the done step; it is the
    first frame of the next episode, so it seeds the new cache while the prior
    frames form the completed episode.
    """
    return cache, [reset_frame]


def collect_demos_from_policy(
    *,
    task: str,
    robot: str,
    checkpoint: str,
    out_dir: str,
    sim: str = "mujoco",
    num_demos: int = 100,
    num_envs: int = 1,
    max_steps: int = 1000,
    device: str = "cuda:0",
    train_cfg=None,
    camera_cfg=None,
    success_fn: Callable[..., "Sequence[bool]"] | None = None,
    task_desc: str = "",
    deterministic: bool = True,
) -> int:
    """Collect ``num_demos`` successful demos by rolling out ``checkpoint``.

    Args:
        task / robot / sim: RoboVerse task + robot + backend (as in RL training).
        checkpoint: path to a TorchScript ``policy.pt`` or an rsl-rl ``model_*.pt``.
        out_dir: demos are written under ``<out_dir>/success/demo_XXXX``.
        num_demos: target number of successful episodes.
        num_envs: parallel rollout envs (more = faster collection).
        max_steps: hard cap on rollout steps (safety bound for infinite-horizon
            locomotion tasks).
        train_cfg: rsl-rl train cfg (only needed to rebuild a runner for
            ``model_*.pt`` checkpoints; ``None`` is fine for TorchScript).
        success_fn: ``(terminated, episode_cache, info) -> bool`` per env.
            Default: an episode is "successful" when it terminated via the task's
            termination/checker signal (``terminated`` True) rather than timing
            out — matching checker-based manipulation tasks. For continuous
            locomotion (no success checker) pass ``lambda *a, **k: True`` to keep
            every completed episode.

    Returns:
        Number of successful demos written.
    """
    import torch

    from metasim.utils.save_util import save_demo
    from metasim.utils.state import state_tensor_to_nested
    from roboverse_learn.rl.rsl_rl.env_wrapper import RslRlEnvWrapper

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    camera_cfg = camera_cfg if camera_cfg is not None else _default_camera()
    env = _build_env(task, robot, sim, num_envs, dev, camera_cfg)
    handler = env.handler
    robot_cfg = env.robots[0] if getattr(env, "robots", None) else None

    env_wrapper = RslRlEnvWrapper(env, train_cfg=train_cfg)
    policy = _load_inference_policy(env_wrapper, train_cfg, checkpoint, dev)

    if success_fn is None:
        def success_fn(terminated, cache, info):  # noqa: ANN001 - default checker-based
            return bool(terminated)

    os.makedirs(os.path.join(out_dir, "success"), exist_ok=True)

    # Per-env trajectory caches (lists of single-env DictEnvState).
    env.reset()
    tensor_state = handler.get_states(mode="tensor")
    if not tensor_state.robots and not tensor_state.objects:
        raise RuntimeError(
            f"Task {task!r} exposes no RobotCfg-registered robot/object in its state "
            f"(its handler.get_states().robots is empty). This is typical of scene-MJCF "
            f"manager tasks (e.g. mjlab cartpole / velocity), whose articulation lives in "
            f"the scene MJCF rather than as a MetaSim RobotCfg. The RL->IL demo bridge "
            f"targets standard manipulation tasks (robot + objects + camera), which is also "
            f"what il/data2zarr_dp.py expects. Use a manipulation-task checkpoint."
        )
    init_nested = state_tensor_to_nested(handler, tensor_state)
    caches: list[list] = [[init_nested[i]] for i in range(num_envs)]

    collected = 0
    step = 0
    while collected < num_demos and step < max_steps:
        with torch.no_grad():
            actions = policy(env.obs_buf)
            if isinstance(actions, tuple):  # some policies return (action, ...)
                actions = actions[0]
        _, _reward, terminated, time_out, info = env.step(actions)
        nested = state_tensor_to_nested(handler, handler.get_states(mode="tensor"))
        done = (terminated | time_out).tolist()
        term_list = terminated.tolist()
        step += 1

        for i in range(num_envs):
            caches[i].append(nested[i])
            if not done[i]:
                continue
            completed, seed = _finalise_episode(caches[i][:-1], caches[i][-1])
            caches[i] = seed
            if collected >= num_demos:
                continue
            if len(completed) < 2:
                continue  # too short to be a useful demo
            if success_fn(term_list[i], completed, info):
                save_dir = os.path.join(out_dir, "success", f"demo_{collected:04d}")
                save_demo(save_dir, completed, robot_cfg, task_desc)
                collected += 1
                log.info(f"[rl_to_demo] saved demo {collected}/{num_demos} ({len(completed)} steps) -> {save_dir}")

    log.info(f"[rl_to_demo] done: {collected} successful demos in {step} steps under {out_dir}/success")
    return collected
