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
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu

rootutils.setup_root(__file__, pythonpath=True)

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
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs)):
        if action_idx < len(all_actions[demo_idx]):
            action = all_actions[demo_idx][action_idx]
        else:
            action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int]) -> list[bool]:
    action_idxs = env._episode_steps
    run_out = [action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs)]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def force_reset_to_state(env, state, env_id):
    """Force reset environment to specific state with validation."""
    env.reset(states=[state], env_ids=[env_id])
    ensure_clean_state(env.handler, expected_state=state)
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


class DemoCollector:
    def __init__(self, handler, robot_cfg, task_desc="", demo_start_idx=0):
        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.robot_cfg = robot_cfg
        self.task_desc = task_desc
        self.cache: dict[int, list[dict]] = {}
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

    def create(self, demo_idx: int, data_dict: dict):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]

    def add(self, demo_idx: int, data_dict: dict):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))

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

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]

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
    env = task_cls(scenario, device=device)

    ## Data
    assert os.path.exists(env.traj_filepath), f"Trajectory file does not exist: {env.traj_filepath}"
    init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)

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
    obs = env.handler.get_states()
    obs = state_tensor_to_nested(env.handler, obs)

    for env_id, demo_idx in enumerate(demo_idxs):
        log.info(f"Starting Demo {demo_idx} in Env {env_id}")
        collector.create(demo_idx, obs[env_id])

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
        obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, obs)
        run_out = get_run_out(all_actions, env, demo_idxs)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id])

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
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
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
                force_reset_to_state(env, init_states[demo_idx], env_id)

                obs = env.handler.get_states()
                obs = state_tensor_to_nested(env.handler, obs)
                collector.create(demo_idx, obs[env_id])
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
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                else:
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
