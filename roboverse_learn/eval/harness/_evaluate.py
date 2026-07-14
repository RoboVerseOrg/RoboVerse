"""``evaluate`` — one entry point; single sim or multi-sim parity.

Wires: build task -> infer embodiment -> derive typed specs -> negotiate (policy card)
-> EnvAdapter -> VecEvalRunner. Passing multiple simulators runs the SAME policy across
backends and returns a :class:`ParityReport`, which is how a policy's cross-engine
robustness is measured rather than assumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from loguru import logger

from metasim.task.registry import get_task_class

from .embodiment import infer_embodiment
from .env_adapter import EnvAdapter
from .policy import Policy
from .runner import VecEvalRunner
from .spec import derive_action_spec, derive_obs_spec


@dataclass
class EvalResult:
    task: str
    simulator: str
    episodes: int
    successes: int
    steps_mean: float = 0.0
    per_episode_success: tuple[bool, ...] = ()

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.episodes, 1)


@dataclass
class ParityReport:
    """One policy across multiple backends — did success generalize across physics engines?"""

    task: str
    results: dict[str, EvalResult] = field(default_factory=dict)

    def success_rate_spread(self) -> float:
        rates = [r.success_rate for r in self.results.values()]
        return (max(rates) - min(rates)) if rates else 0.0

    def divergent(self, tol: float = 0.05) -> bool:
        return self.success_rate_spread() > tol


def _camera_shapes(scenario):
    out = []
    for cam in getattr(scenario, "cameras", None) or []:
        h = getattr(cam, "height", None)
        w = getattr(cam, "width", None)
        name = getattr(cam, "name", None)
        if not (name and h and w):  # don't silently vanish a malformed camera — the policy would KeyError
            raise ValueError(f"scenario camera has no name/height/width: {name=} {h=} {w=}")
        if "rgb" not in (getattr(cam, "data_types", None) or ["rgb"]):
            raise ValueError(
                f"camera {name!r} does not render 'rgb' (data_types={cam.data_types}); the harness "
                "derives an <cam>.rgb obs field for every scenario camera."
            )
        out.append((name, (int(h), int(w))))
    return out


def _run_one(
    task_name: str,
    policy: Policy,
    *,
    simulator: str,
    episodes: int,
    num_envs: int,
    control: str,
    chunk: str,
    max_steps: int | None,
    headless: bool,
    seed: int | None,
    include_ee_pose: bool,
    cameras,
) -> EvalResult:
    cls = get_task_class(task_name)
    # `cls.scenario` is a CLASS attribute and `update()` mutates in place, so configure a copy:
    # otherwise this run's simulator/num_envs/cameras leak into every later use of the task class
    # in the same process (a camera added here would silently force rendering on the next run).
    scenario = copy.deepcopy(cls.scenario)
    scenario.update(simulator=simulator, num_envs=num_envs, headless=headless)
    if cameras is not None:
        scenario.update(cameras=list(cameras))
    env = cls(scenario)
    env.task_name = task_name
    try:
        # RLTaskEnv.step auto-resets terminated envs, which breaks the wave-based runner's
        # per-env episode accounting (a reset env's terminated would refer to a new episode).
        # Fail fast rather than silently mis-measure — this runner targets BaseTaskEnv checker tasks.
        from metasim.task.rl_task import RLTaskEnv

        if isinstance(env, RLTaskEnv):
            raise NotImplementedError(
                f"{type(env).__name__} is an RLTaskEnv (auto-resets in step); the harness runner "
                "targets BaseTaskEnv checker tasks. Use roboverse_learn.rl for RL rollouts."
            )
        emb = infer_embodiment(list(scenario.robots))
        obs_spec = derive_obs_spec(
            emb,
            cameras=_camera_shapes(scenario),
            include_ee_pose=include_ee_pose,
            include_language=callable(getattr(env, "get_language_instruction", None)),
        )
        card = policy.describe()
        # Connect-time contract check: if the policy declares the obs fields it needs, verify the
        # env actually produces them — fail fast with a typed error naming the missing field,
        # instead of a KeyError deep inside act() mid-rollout. (Bundled/scripted policies advertise
        # an empty ObsSpec and learn the concrete spec via bind(), so the check is skipped for them.)
        if card.needs_obs.fields:
            obs_spec.compatible_with(card.needs_obs).raise_if_bad()
        action_spec = derive_action_spec(emb, control=control, chunk_len=card.produces_action.chunk_len)
        # ...and the same in the action direction: a policy that advertises concrete action fields
        # (or a control the env is not driving) must not silently get a different spec.
        action_spec.compatible_with(card.produces_action).raise_if_bad()
        adapter = EnvAdapter(env, emb, obs_spec, action_spec)
        # let the policy learn the concrete derived specs (negotiation / setup hook)
        if not hasattr(policy, "bind"):
            raise TypeError(
                f"policy {type(policy).__name__} has no bind(obs_spec, action_spec); it is part of the "
                "Policy protocol (the harness hands every policy the derived specs before the rollout)."
            )
        policy.bind(obs_spec, action_spec)
        runner = VecEvalRunner(adapter, policy, chunk=chunk)
        ms = max_steps or getattr(cls, "max_episode_steps", 300)
        roll = runner.run(episodes=episodes, max_steps=ms, seed=seed)
        return EvalResult(
            task=task_name,
            simulator=simulator,
            episodes=roll.episodes,
            successes=roll.successes,
            steps_mean=roll.steps_mean,
            per_episode_success=roll.per_episode_success,
        )
    finally:
        try:
            env.close()
        except Exception as e:  # cleanup best-effort, but surface the failure (AGENTS: no silent swallow)
            logger.warning(f"[harness] env.close() failed for {task_name} on {simulator}: {e}")


def evaluate(
    task_name: str,
    policy: Policy,
    *,
    simulators: str | list[str] = "mujoco",
    episodes: int = 10,
    num_envs: int = 1,
    control: str = "joint_pos",
    chunk: str = "auto",
    max_steps: int | None = None,
    headless: bool = True,
    seed: int | None = None,
    include_ee_pose: bool = True,
    cameras: list | None = None,
) -> EvalResult | ParityReport:
    """Evaluate ``policy`` on ``task_name``. A list of ``simulators`` -> ``ParityReport``.

    Args:
        task_name: registered task id, e.g. ``"maniskill.pick_cube"``.
        policy: any object satisfying the :class:`~.policy.Policy` protocol (or a
            :class:`~.transport.base.PolicyHandle` wrapping a remote one).
        simulators: one backend, or a list -> a :class:`ParityReport` across backends.
        episodes: a **floor** on the number of episodes (every env of the final wave counts).
        num_envs: envs stepped in parallel.
        control: action space of the arms; see :data:`~.spec.SUPPORTED_CONTROLS`.
        chunk: ``"auto"`` | ``"none"`` | ``"scheduler"`` | ``"ensemble"``.
        max_steps: episode cap; defaults to the task's ``max_episode_steps``.
        headless: run the backend without a viewer.
        seed: base seed; wave *i* uses ``seed + i``.
        include_ee_pose: emit ``<arm>.ee_pose`` obs fields (arms with an ``ee_body_name`` only).
        cameras: ``CameraCfg`` list to render (e.g.
            ``[PinholeCameraCfg(name="camera0", data_types=["rgb"], width=256, height=256,
            pos=(1.0, 0.0, 0.75), look_at=(0.0, 0.0, 0.0))]``). Registered tasks ship no camera,
            so a vision/VLA policy that declares a ``<cam>.rgb`` obs field **must** be given one
            here — otherwise the obs spec has no camera field and negotiation fails at connect.
    """
    sims = [simulators] if isinstance(simulators, str) else list(simulators)
    try:
        results = {
            sim: _run_one(
                task_name,
                policy,
                simulator=sim,
                episodes=episodes,
                num_envs=num_envs,
                control=control,
                chunk=chunk,
                max_steps=max_steps,
                headless=headless,
                seed=seed,
                include_ee_pose=include_ee_pose,
                cameras=cameras,
            )
            for sim in sims
        }
    finally:
        # a remote policy holds a socket + event loop; a local one may hold a model. Release it
        # once per evaluate() (not per simulator — the same policy object is reused across them).
        closer = getattr(policy, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as e:  # best-effort, but never hide it (AGENTS: no silent swallow)
                logger.warning(f"[harness] policy.close() failed: {e}")
    if len(sims) == 1:
        return results[sims[0]]
    return ParityReport(task=task_name, results=results)
