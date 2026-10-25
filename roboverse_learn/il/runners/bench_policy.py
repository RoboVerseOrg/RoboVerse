"""An imitation-learning eval runner as a ``metasim.bench`` policy.

The IL stack's :class:`~roboverse_learn.il.runners.base_eval_runner.BaseEvalRunner` turns the camera
image and the joint positions into the policy's inputs and its predicted chunk into joint targets.
This adapter hands it the observation the benchmark harness provides and returns its actions, so an
IL checkpoint is scored by ``python -m metasim bench run`` like any other policy::

    python -m metasim bench run --task stack_cube --sim mujoco --robot franka \\
        --policy roboverse_learn.il.runners.bench_policy:ILPolicy --checkpoint outputs/.../latest.ckpt \\
        --policy-arg camera=camera0
"""

from __future__ import annotations

from typing import Any

import torch

from metasim.bench import PolicyContext, TaskInstance


class ILPolicy:
    """Wrap an IL eval runner (``get_action(obs_dict)`` / ``reset()``) as a bench policy.

    Pass a ready ``runner`` (any :class:`BaseEvalRunner`), or a ``checkpoint`` and let the adapter build
    the diffusion-policy :class:`DefaultEvalRunner` from the training config stored in it on the first
    ``reset`` (the runner needs the scenario, which the harness supplies through the context). The
    harness observation is a ``TensorState``; the runner sees ``{"rgb": <camera>.rgb, "joint_qpos":
    robots[<robot>].joint_pos}``, exactly what ``DefaultRunner.eval`` feeds it.
    """

    name = "il"

    def __init__(
        self,
        runner: Any | None = None,
        checkpoint: str | None = None,
        camera: str = "camera0",
        robot: str | None = None,
        device: str | None = None,
        task_name: str | None = None,
    ) -> None:
        if runner is None and checkpoint is None:
            raise ValueError("ILPolicy needs a runner or a checkpoint")
        self.runner = runner
        self.checkpoint = checkpoint
        self.camera = camera
        self.robot = robot
        self.device = device
        self.task_name = task_name

    def _build_runner(self, ctx: PolicyContext) -> Any:
        import dill

        from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner
        from roboverse_learn.il.runners.default_runner import DefaultRunner

        scenario = ctx.extras.get("scenario")
        if scenario is None:
            raise RuntimeError("ILPolicy needs the scenario in the policy context (ctx.extras['scenario'])")
        with open(self.checkpoint, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location="cpu", weights_only=False)
        workspace = DefaultRunner(payload["cfg"])
        device = self.device or ("cuda:0" if ctx.device.type == "cuda" else str(ctx.device))
        return DefaultEvalRunner(
            workspace,
            scenario=scenario,
            num_envs=ctx.num_envs,
            checkpoint_path=self.checkpoint,
            device=device,
            task_name=self.task_name or ctx.task,
        )

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Build the runner on first use and clear its action cache / observation history."""
        if self.runner is None:
            self.runner = self._build_runner(ctx)
        if self.robot is None:
            self.robot = ctx.robots[0]
        self.runner.reset()

    def act(self, obs: Any, ctx: PolicyContext) -> Any:
        """The runner's joint-target action for the harness observation."""
        if self.runner is None:
            raise RuntimeError("ILPolicy.act called before reset")
        cameras = getattr(obs, "cameras", None) or {}
        if self.camera not in cameras:
            raise KeyError(
                f"ILPolicy needs camera {self.camera!r} in the observation (have {sorted(cameras)}); add it to the "
                "scenario or pass --policy-arg camera=<name>"
            )
        robot_name = self.robot or ctx.robots[0]
        return self.runner.get_action({
            "rgb": cameras[self.camera].rgb,
            "joint_qpos": obs.robots[robot_name].joint_pos,
        })
