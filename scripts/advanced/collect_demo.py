"""Demo collection script with domain randomization support.

Collects demonstration data by replaying trajectories with optional domain randomization.

Randomization Levels:
- Level 0: No randomization
- Level 1: Scene + Material randomization
- Level 2: Level 1 + Lighting randomization
- Level 3: Level 2 + Camera randomization

Scene Modes:
- Mode 0: Manual geometry
- Mode 1: USD Table + Manual environment
- Mode 2: USD Scene (Kujiale) + USD Table
- Mode 3: Full USD (Scene + Table + Desktop objects)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import tyro
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.render import RenderCfg


@dataclass
class Args:
    render: RenderCfg = field(default_factory=RenderCfg)
    """Renderer options"""
    task: str = "pick_butter"
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "isaacsim", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    num_demo_success: int | None = None
    """Target number of successful demos to collect"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    custom_save_dir: str | None = None
    """Custom base path for saving demos. If None, use default structure."""
    traj_filepath: str | None = None
    """Demo file to replay; defaults to the task's ``traj_filepath``."""
    episode_sidecar: bool = True
    """Also write each demo as a self-describing ``episode.npz`` (full state, provenance) next to its metadata."""
    scene: str | None = None
    """Scene name"""
    run_all: bool = True
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""
    renderer: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"

    # Domain randomization options
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"""
    scene_mode: Literal[0, 1, 2, 3] = 0
    """Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    def __post_init__(self):
        assert self.run_all or self.run_unfinished or self.run_failed, (
            "At least one of run_all, run_unfinished, or run_failed must be True"
        )
        if self.num_demo_success is None:
            self.num_demo_success = 100
        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        log.info(f"Args: {self}")

        # Log randomization settings
        if self.level > 0:
            mode_names = {0: "Manual", 1: "USD Table", 2: "USD Scene", 3: "Full USD"}
            log.info("=" * 60)
            log.info("DOMAIN RANDOMIZATION CONFIGURATION")
            log.info(f"  Level: {self.level}")
            log.info(f"  Scene Mode: {self.scene_mode} ({mode_names[self.scene_mode]})")
            log.info("  Randomization:")
            log.info("    Level 1+: Scene + Material")
            log.info("    Level 2+: + Lighting")
            log.info("    Level 3+: + Camera")
            log.info(f"  Seed: {self.randomization_seed if self.randomization_seed else 'Random'}")
            log.info("=" * 60)


args = tyro.cli(Args)

import multiprocessing as mp
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
from tqdm.rich import tqdm_rich as tqdm

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler
from metasim.task.registry import get_task_class
from metasim.types import TensorState
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.utils.state import select_envs, state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu
from metasim.utils.trajectory import episode_from_states, provenance_from_handler, save_episode

rootutils.setup_root(__file__, pythonpath=True)
from roboverse_learn.il.utils.clean_state import ensure_clean_state

# Import randomization components
try:
    from metasim.randomization import DomainRandomizationManager, DRConfig

    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Randomization components not available: {e}")
    RANDOMIZATION_AVAILABLE = False


def get_actions(all_actions, env, demo_idxs: list[int], robot: RobotCfg):
    action_idxs = env._episode_steps

    actions = []
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs, strict=False)):
        if action_idx < len(all_actions[demo_idx]):
            action = all_actions[demo_idx][action_idx]
        else:
            action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int]) -> list[bool]:
    action_idxs = env._episode_steps
    run_out = [
        action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs, strict=False)
    ]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def force_reset_to_state(env, state, env_id, collector=None, demo_idxs=None, finished=None, terminal=None):
    """Force reset one env to ``state`` and settle it.

    Settling steps the whole batch, so every other in-flight env physically advances too; those steps
    are appended to their demos (state + the joint target they were holding), otherwise the recorded
    episodes would skip physics that happened and no longer replay.
    """
    env.reset(states=[state], env_ids=[env_id])

    def _keep_other_envs(tensor_state):
        if collector is None:
            return
        # only envs that keep stepping after this iteration receive the settle physics
        others = [
            other
            for other in range(env.handler.num_envs)
            if other != env_id
            and not (finished is not None and finished[other])
            and not (terminal is not None and other in terminal)
            and demo_idxs[other] in collector.cache
        ]
        if not others:
            return
        nested = state_tensor_to_nested(env.handler, select_envs(tensor_state, others))
        for local, other in enumerate(others):
            collector.add(demo_idxs[other], nested[local], tensor_state=tensor_state, env_id=other)

    ensure_clean_state(env.handler, expected_state=state, env_id=env_id, on_step=_keep_other_envs)
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


def _physics_slice(tensor_state, env_id: int):
    """Env ``env_id``'s physics state without camera images (they live in the legacy demo, not the record)."""
    physics_only = TensorState(objects=tensor_state.objects, robots=tensor_state.robots, cameras={}, extras={})
    return select_envs(physics_only, [env_id])


class DemoCollector:
    def __init__(self, handler, robot_cfg, task_desc="", demo_start_idx=0):
        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.robot_cfg = robot_cfg
        self.task_desc = task_desc
        self.cache: dict[int, list[dict]] = {}
        self.episode_states: dict[int, list] = {}
        self.episode_actions: dict[int, list] = {}
        # run-invariant (backend, versions, asset hashes, git): computed once, stamped per episode
        self._provenance = (
            provenance_from_handler(handler, seed=args.randomization_seed, num_envs=1) if args.episode_sidecar else None
        )
        self._warned_no_targets = False
        self.save_request_queue = mp.Queue()
        self.save_proc = mp.Process(target=save_demo_mp, args=(self.save_request_queue, robot_cfg, task_desc))
        self.save_proc.start()

        TaskName = args.task
        if args.custom_save_dir:
            self.base_save_dir = args.custom_save_dir
        else:
            additional_str = f"-{args.cust_name}" if args.cust_name else ""
            self.base_save_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    def _get_max_demo_index(self, status: str) -> int:
        status_dir = os.path.join(self.base_save_dir, status)
        if not os.path.exists(status_dir):
            return 0

        max_idx = -1
        for item in os.listdir(status_dir):
            if item.startswith("demo_") and os.path.isdir(os.path.join(status_dir, item)):
                try:
                    idx = int(item.split("_")[1])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    continue

        return max_idx + 1

    def create(self, demo_idx: int, data_dict: dict, tensor_state=None, env_id: int | None = None):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]
        # the replayable episode: this env's full state before every action, and the joint targets applied
        self.episode_states[demo_idx] = (
            [_physics_slice(tensor_state, env_id)] if tensor_state is not None and args.episode_sidecar else []
        )
        self.episode_actions[demo_idx] = []

    def add(self, demo_idx: int, data_dict: dict, tensor_state=None, env_id: int | None = None):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))
        if tensor_state is not None and self.episode_states.get(demo_idx):
            env_state = _physics_slice(tensor_state, env_id)
            target = env_state.robots[self.robot_cfg.name].joint_pos_target
            if target is None:
                self.episode_states[demo_idx] = []  # this backend does not report targets: no episode file
                if not self._warned_no_targets:
                    self._warned_no_targets = True
                    log.warning(
                        f"{type(self.handler).__name__} reports no joint_pos_target in get_states; no episode.npz "
                        "sidecar can be written for this run (the legacy demo files are unaffected)."
                    )
                return
            self.episode_actions[demo_idx].append(target.detach().cpu().clone())
            self.episode_states[demo_idx].append(env_state)

    def save(self, demo_idx: int, status: str):
        assert demo_idx in self.cache
        assert status in ["success", "failed"], f"Invalid status: {status}"

        continuous_idx = demo_idx

        save_dir = os.path.join(self.base_save_dir, status, f"demo_{continuous_idx:04d}")
        if os.path.exists(os.path.join(save_dir, "status.txt")):
            os.remove(os.path.join(save_dir, "status.txt"))

        os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving demo {demo_idx} as {continuous_idx:04d} to {save_dir}")

        from metasim.utils.save_util import save_demo

        save_demo(save_dir, self.cache[demo_idx], self.robot_cfg, self.task_desc)

        if status == "failed":
            with open(os.path.join(save_dir, "status.txt"), "w") as f:
                f.write(status)

        states = self.episode_states.get(demo_idx) or []
        if args.episode_sidecar and states and len(states) == len(self.episode_actions[demo_idx]) + 1:
            episode = episode_from_states(
                self.handler,
                states,
                self.episode_actions[demo_idx],
                seed=args.randomization_seed,
                num_envs=1,
                provenance=self._provenance,
                info={
                    "task": args.task,
                    "robot": args.robot,
                    "status": status,
                    "task_desc": self.task_desc,
                    "demo_idx": demo_idx,
                },
            )
            save_episode(episode, os.path.join(save_dir, "episode.npz"))

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]
        self.episode_states.pop(demo_idx, None)
        self.episode_actions.pop(demo_idx, None)

    def final(self):
        self.save_request_queue.put(None)  # signal to save_demo_mp to exit
        self.save_proc.join()
        assert self.cache == {}


def should_skip(log_dir: str, demo_idx: int):
    demo_name = f"demo_{demo_idx:04d}"
    success_path = os.path.join(log_dir, "success", demo_name, "status.txt")
    failed_path = os.path.join(log_dir, "failed", demo_name, "status.txt")

    if args.run_unfinished:
        if not os.path.exists(success_path) and not os.path.exists(failed_path):
            return False
        return True

    if args.run_all:
        return False

    if args.run_failed:
        if os.path.exists(success_path):
            return is_status_success(log_dir, demo_idx)
        return False

    return True


def is_status_success(log_dir: str, demo_idx: int) -> bool:
    demo_name = f"demo_{demo_idx:04d}"
    status_path = os.path.join(log_dir, "success", demo_name, "status.txt")

    if os.path.exists(status_path):
        return open(status_path).read().strip() == "success"
    return False


class DemoIndexer:
    def __init__(self, save_root_dir: str, start_idx: int, end_idx: int, pbar: tqdm):
        self.save_root_dir = save_root_dir
        self._next_idx = start_idx
        self.end_idx = end_idx
        self.pbar = pbar
        self._skip_if_should()

    @property
    def next_idx(self):
        return self._next_idx

    def _skip_if_should(self):
        while should_skip(self.save_root_dir, self._next_idx):
            global global_step, tot_success, tot_give_up
            if is_status_success(self.save_root_dir, self._next_idx):
                tot_success += 1
            else:
                tot_give_up += 1
            self.pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
            self.pbar.update(1)
            log.info(f"Demo {self._next_idx} already exists, skipping...")
            self._next_idx += 1

    def move_on(self):
        self._next_idx += 1
        self._skip_if_should()


def main():
    global global_step, tot_success, tot_give_up
    task_cls = get_task_class(args.task)

    # NOTE (multi-agent limitation): this collection script is single-robot by
    # construction -- it builds the scenario with ``robots=[args.robot]`` and the
    # DR / save pipeline below assumes one robot. Multi-agent (bimanual)
    # *replay* is supported via scripts/advanced/replay_demo.py, but collecting
    # fresh multi-agent demos here is not yet wired up. Fail loud rather than
    # silently collecting only one arm of a multi-agent task.
    _task_robots = task_cls.scenario.robots
    if _task_robots is not None and len(_task_robots) > 1:
        raise NotImplementedError(
            f"Task '{args.task}' declares {len(_task_robots)} robots (multi-agent). collect_demo.py only "
            "supports single-robot collection; use scripts/advanced/replay_demo.py to replay multi-agent "
            "trajectories, or extend this script before collecting multi-agent demos."
        )

    if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
        dp_camera = True
    else:
        dp_camera = args.task != "close_box"

    is_libero_dataset = "libero_90" in args.task

    if is_libero_dataset:
        dp_pos = (2.0, 0.0, 2)
    elif dp_camera:
        dp_pos = (1.0, 0.0, 0.75)
    else:
        dp_pos = (1.5, 0.0, 1.5)

    camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=dp_pos, look_at=(0.0, 0.0, 0.0))

    # Lighting setup
    if args.render.mode == "pathtracing":
        ceiling_main = 18000.0
        ceiling_corners = 8000.0
    else:
        ceiling_main = 12000.0
        ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
        ),
    ]

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=lights,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    robot = get_robot(args.robot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.traj_filepath:
        # the task base downloads and loads its own ``traj_filepath`` in __init__; point it at the override first
        task_cls.traj_filepath = args.traj_filepath
    env = task_cls(scenario, device=device)

    ## Data
    traj_filepath = args.traj_filepath or getattr(env, "traj_filepath", None)
    if not traj_filepath:
        raise ValueError(
            f"task {args.task!r} declares no traj_filepath (demo-free task); pass --traj_filepath <file> to collect from a demo file"
        )
    if not os.path.exists(traj_filepath):
        raise FileNotFoundError(f"Trajectory file does not exist: {traj_filepath}")
    init_states, all_actions, all_states = get_traj(traj_filepath, robot, env.handler)

    # Initialize domain randomization manager
    randomization_manager = DomainRandomizationManager(
        config=DRConfig(
            level=args.level,
            scene_mode=args.scene_mode,
            randomization_seed=args.randomization_seed,
        ),
        scenario=scenario,
        handler=env.handler,
        init_states=init_states,
        render_cfg=args.render,
    )

    tot_demo = len(all_actions)
    if args.split == "train":
        init_states = init_states[: int(tot_demo * 0.9)]
        all_actions = all_actions[: int(tot_demo * 0.9)]
        all_states = all_states[: int(tot_demo * 0.9)]
    elif args.split == "val" or args.split == "test":
        init_states = init_states[int(tot_demo * 0.9) :]
        all_actions = all_actions[int(tot_demo * 0.9) :]
        all_states = all_states[int(tot_demo * 0.9) :]

    n_demo = len(all_actions)
    log.info(f"Collecting from {args.split} split, {n_demo} out of {tot_demo} demos")

    ########################################################
    ## Main
    ########################################################
    max_demo = n_demo
    try_num = args.retry_num + 1

    ## Demo collection state machine:
    ## CollectingDemo -> Success -> FinalizeDemo -> NextDemo
    ## CollectingDemo -> Timeout -> Retry/GiveUp -> NextDemo

    ## Setup
    task_desc = getattr(env, "task_desc", "")
    # collector = DemoCollector(env.handler, robot, task_desc)
    # pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")
    collector = DemoCollector(env.handler, robot, task_desc, demo_start_idx=args.demo_start_idx)
    pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")

    ## State variables
    failure_count = [0] * env.handler.num_envs
    steps_after_success = [0] * env.handler.num_envs
    finished = [False] * env.handler.num_envs
    TaskName = args.task

    if args.cust_name is not None:
        additional_str = f"-{args.cust_name}"
    else:
        additional_str = ""

    if args.custom_save_dir:
        save_root_dir = args.custom_save_dir
    else:
        save_root_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    demo_indexer = DemoIndexer(
        save_root_dir=save_root_dir,
        start_idx=args.demo_start_idx,
        end_idx=max_demo,
        pbar=pbar,
    )
    demo_idxs = []
    for demo_idx in range(env.handler.num_envs):
        demo_idxs.append(demo_indexer.next_idx)
        demo_indexer.move_on()
    log.info(f"Initialize with demo idxs: {demo_idxs}")

    ## Apply initial randomization (create scene and update positions)
    for env_id, demo_idx in enumerate(demo_idxs):
        randomization_manager.apply_randomization(demo_idx, is_initial=True)
        randomization_manager.update_positions_to_table(demo_idx, env_id)
        randomization_manager.update_camera_look_at(env_id)
        randomization_manager.apply_camera_randomization()  # Apply camera randomization after baseline adjustment

    ## Reset to initial states (after position adjustment)
    obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])

    ## Wait for environment to stabilize after reset
    ensure_clean_state(env.handler)

    ## Reset episode step counters after stabilization
    if hasattr(env, "_episode_steps"):
        for env_id in range(env.handler.num_envs):
            env._episode_steps[env_id] = 0

    ## Record the clean, stabilized initial state
    tensor_obs = env.handler.get_states(mode="tensor")
    obs = state_tensor_to_nested(env.handler, tensor_obs)

    for env_id, demo_idx in enumerate(demo_idxs):
        log.info(f"Starting Demo {demo_idx} in Env {env_id}")
        collector.create(demo_idx, obs[env_id], tensor_state=tensor_obs, env_id=env_id)

    ## Main Loop
    stop_flag = False

    while not all(finished):
        if stop_flag:
            pass

        if tot_success >= args.num_demo_success:
            log.info(f"Reached target number of successful demos ({args.num_demo_success}).")
            stop_flag = True

        if demo_indexer.next_idx >= max_demo:
            if not stop_flag:
                log.warning(f"Reached maximum demo index ({max_demo}), finishing in-flight demos.")
            stop_flag = True

        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
        actions = get_actions(all_actions, env, demo_idxs, robot)
        tensor_obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, tensor_obs)
        run_out = get_run_out(all_actions, env, demo_idxs)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id], tensor_state=tensor_obs, env_id=env_id)

        # envs whose demo ends this iteration (saved as success, timed out, or ran out of actions): their
        # legacy demos and episodes are closed below, so they must not absorb another env's settle frames
        done_mask = time_out | torch.tensor(run_out, device=time_out.device)
        terminal = set(done_mask.nonzero().squeeze(-1).tolist())
        for env_id in success.nonzero().squeeze(-1).tolist():
            if not finished[env_id] and (
                run_out[env_id] or steps_after_success[env_id] >= args.tot_steps_after_success
            ):
                terminal.add(env_id)
        for env_id in success.nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            if steps_after_success[env_id] == 0:
                log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                tot_success += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

            if not run_out[env_id] and steps_after_success[env_id] < args.tot_steps_after_success:
                steps_after_success[env_id] += 1
            else:
                steps_after_success[env_id] = 0
                collector.save(demo_idx, status="success")
                collector.delete(demo_idx)

                if (not stop_flag) and (demo_indexer.next_idx < max_demo):
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    log.info(f"Transitioning Env {env_id}: Demo {demo_idx} to Demo {new_demo_idx}")

                    randomization_manager.apply_randomization(new_demo_idx, is_initial=False)
                    randomization_manager.update_positions_to_table(new_demo_idx, env_id)
                    randomization_manager.update_camera_look_at(env_id)
                    randomization_manager.apply_camera_randomization()  # Apply camera randomization
                    force_reset_to_state(
                        env,
                        init_states[new_demo_idx],
                        env_id,
                        collector=collector,
                        demo_idxs=demo_idxs,
                        finished=finished,
                        terminal=terminal,
                    )

                    tensor_obs = env.handler.get_states(mode="tensor")
                    obs = state_tensor_to_nested(env.handler, tensor_obs)
                    collector.create(new_demo_idx, obs[env_id], tensor_state=tensor_obs, env_id=env_id)
                    terminal.discard(env_id)  # reset and recording again: it keeps stepping
                    demo_indexer.move_on()
                    run_out[env_id] = False
                else:
                    finished[env_id] = True

        for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
            collector.save(demo_idx, status="failed")
            collector.delete(demo_idx)
            failure_count[env_id] += 1

            if failure_count[env_id] < try_num:
                log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                randomization_manager.apply_randomization(demo_idx, is_initial=False)
                randomization_manager.update_positions_to_table(demo_idx, env_id)
                randomization_manager.update_camera_look_at(env_id)
                randomization_manager.apply_camera_randomization()  # Apply camera randomization
                force_reset_to_state(
                    env,
                    init_states[demo_idx],
                    env_id,
                    collector=collector,
                    demo_idxs=demo_idxs,
                    finished=finished,
                    terminal=terminal,
                )

                tensor_obs = env.handler.get_states(mode="tensor")
                obs = state_tensor_to_nested(env.handler, tensor_obs)
                collector.create(demo_idx, obs[env_id], tensor_state=tensor_obs, env_id=env_id)
                terminal.discard(env_id)  # reset and recording again: it keeps stepping
            else:
                log.error(f"Demo {demo_idx} failed too many times, giving up")
                failure_count[env_id] = 0
                tot_give_up += 1
                # pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                if demo_indexer.next_idx < max_demo:
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    randomization_manager.apply_randomization(new_demo_idx, is_initial=False)
                    randomization_manager.update_positions_to_table(new_demo_idx, env_id)
                    randomization_manager.update_camera_look_at(env_id)
                    randomization_manager.apply_camera_randomization()  # Apply camera randomization
                    force_reset_to_state(
                        env,
                        init_states[new_demo_idx],
                        env_id,
                        collector=collector,
                        demo_idxs=demo_idxs,
                        finished=finished,
                        terminal=terminal,
                    )

                    tensor_obs = env.handler.get_states(mode="tensor")
                    obs = state_tensor_to_nested(env.handler, tensor_obs)
                    collector.create(new_demo_idx, obs[env_id], tensor_state=tensor_obs, env_id=env_id)
                    terminal.discard(env_id)  # reset and recording again: it keeps stepping
                    demo_indexer.move_on()
                else:
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
