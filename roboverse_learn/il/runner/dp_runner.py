import copy
import datetime
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf

import hydra
import imageio.v2 as iio
import numpy as np
import torch
import tqdm
import wandb
from roboverse_learn.il.utils.ema_model import EMAModel
from loguru import logger as log
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from roboverse_learn.il.base.base_eval_runner import BaseEvalRunner
from roboverse_learn.il.base.base_runner import BaseRunner
from roboverse_learn.il.utils.eval_args import Args
from roboverse_learn.il.utils.eval_runner_getter import get_runner
from roboverse_learn.il.utils.json_logger import JsonLogger
from roboverse_learn.il.utils.lr_scheduler import get_scheduler
from roboverse_learn.il.utils.pytorch_util import dict_apply, optimizer_to
from torch.utils.data import DataLoader

from metasim.task.registry import get_task_class

from metasim.randomization import DomainRandomizationManager, DRConfig

RANDOMIZATION_AVAILABLE = True


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

class DPRunner(BaseRunner):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.model_config)

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.eval_args = hydra.utils.instantiate(cfg.eval_config.eval_args)

    def train(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.train_config.training_params.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.dataset_config)
        train_dataloader = create_dataloader(dataset, **cfg.train_config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(
            val_dataset, **cfg.train_config.val_dataloader
        )

        self.model.set_normalizer(normalizer)
        if cfg.train_config.training_params.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.train_config.training_params.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.train_config.training_params.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.train_config.training_params.num_epochs
            )
            // cfg.train_config.training_params.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.train_config.training_params.use_ema:
            ema = hydra.utils.instantiate(cfg.train_config.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None
        wandb_run = None

        # configure logging
        if cfg.logging.mode == "online":
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # device transfer
        device = torch.device(cfg.train_config.training_params.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.train_config.training_params.debug:
            cfg.train_config.training_params.num_epochs = 2
            cfg.train_config.training_params.max_train_steps = 3
            cfg.train_config.training_params.max_val_steps = 3
            cfg.train_config.training_params.rollout_every = 1
            cfg.train_config.training_params.checkpoint_every = 1
            cfg.train_config.training_params.val_every = 1
            cfg.train_config.training_params.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.train_config.training_params.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.train_config.training_params.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # print("obs_dict:", batch)
                        # print("dict_keys:", batch.keys())
                        # print("dict_items:", batch.items())
                        # print()
                        # from pprint import pprint

                        # pprint(batch)
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = (
                            raw_loss
                            / cfg.train_config.training_params.gradient_accumulate_every
                        )
                        loss.backward()

                        # step optimizer
                        if (
                            self.global_step
                            % cfg.train_config.training_params.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.train_config.training_params.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.train_config.training_params.max_train_steps is not None
                        ) and batch_idx >= (
                            cfg.train_config.training_params.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.train_config.training_params.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.train_config.training_params.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.train_config.training_params.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.train_config.training_params.max_val_steps
                                    is not None
                                ) and batch_idx >= (
                                    cfg.train_config.training_params.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.train_config.training_params.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        # print("obs_dict:", obs_dict)
                        # print("dict_keys:", obs_dict.keys())
                        # print("dict_items:", obs_dict.items())
                        # print()
                        # from pprint import pprint
                        # pprint(obs_dict)
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (
                    (self.epoch + 1) % cfg.train_config.training_params.checkpoint_every
                ) == 0 or self.epoch + 1 >= cfg.train_config.training_params.num_epochs:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.dataset_config.zarr_path).stem
                    self.save_checkpoint(
                        cfg.checkpoint.save_root_dir
                        + f"/checkpoints/{self.epoch + 1}.ckpt"
                    )  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

    def evaluate(self, ckpt_path=None):
        args = self.eval_args

        num_envs: int = args.num_envs
        log.info(f"Using GPU device: {args.gpu_id}")
        task_cls = get_task_class(args.task)

        # Camera configuration
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

        camera = PinholeCameraCfg(
            name="camera0",
            data_types=["rgb", "depth"],
            width=256,
            height=256,
            pos=dp_pos,
            look_at=(0.0, 0.0, 0.0),
        )

        # Lighting setup
        render_mode = getattr(args, 'render_mode', 'raytracing')
        if render_mode == "pathtracing":
            ceiling_main = 18000.0
            ceiling_corners = 8000.0
        else:
            ceiling_main = 12000.0
            ceiling_corners = 5000.0

        from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
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
            simulator=args.sim,
            num_envs=args.num_envs,
            headless=args.headless,
            lights=lights,
            cameras=[camera]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tic = time.time()
        env = task_cls(scenario, device=device)
        robot = get_robot(args.robot)

        # Domain Randomization configuration
        dr_level = getattr(args, 'level', 0)
        dr_scene_mode = getattr(args, 'scene_mode', 0)
        dr_seed = getattr(args, 'randomization_seed', None)

        if not RANDOMIZATION_AVAILABLE:
            if dr_level > 0:
                log.warning("Domain randomization requested but not available!")
            randomization_manager = None
        else:
            from dataclasses import dataclass as dc
            @dc
            class SimpleRenderCfg:
                mode: str = render_mode

            randomization_manager = DomainRandomizationManager(
                config=DRConfig(
                    level=dr_level,
                    scene_mode=dr_scene_mode,
                    randomization_seed=dr_seed,
                ),
                scenario=scenario,
                handler=env.handler,
                init_states=None,
                render_cfg=SimpleRenderCfg(mode=render_mode)
            )
            if dr_level > 0:
                log.info(f"Domain Randomization enabled: level={dr_level}, scene_mode={dr_scene_mode}, seed={dr_seed}")
            else:
                log.info("Domain Randomization disabled (level=0)")

        toc = time.time()
        log.trace(f"Time to launch: {toc - tic:.2f}s")

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = self.get_checkpoint_path()
        checkpoint = ckpt_path if ckpt_path is not None else checkpoint
        if checkpoint is None:
            raise ValueError(
                "No checkpoint found, please provide a valid checkpoint path."
            )
        args.checkpoint_path = checkpoint
        ckpt_name = args.checkpoint_path.name + "_" + time_str
        ckpt_name = f"{args.task}/{args.algo}/{args.robot}/{ckpt_name}"
        runnerCls = get_runner(args.algo)
        policyRunner: BaseEvalRunner = runnerCls(
            self,
            scenario=scenario,
            num_envs=num_envs,
            checkpoint_path=args.checkpoint_path,
            device=f"cuda:{args.gpu_id}",
            task_name=args.task,
            subset=args.subset,
        )

        action_set_steps = (
            2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
        )
        ## Data
        tic = time.time()
        assert os.path.exists(env.traj_filepath), (
            f"Trajectory file: {env.traj_filepath} does not exist."
        )
        init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
        num_demos = len(init_states)
        toc = time.time()
        log.trace(f"Time to load data: {toc - tic:.2f}s")

        # Update DR manager with init_states
        if randomization_manager is not None:
            randomization_manager.init_states = init_states
            randomization_manager.original_positions = {}
            for demo_idx, init_state in enumerate(init_states):
                demo_key = f"demo_{demo_idx}"
                randomization_manager.original_positions[demo_key] = {}

                if "objects" in init_state:
                    for obj_name, obj_state in init_state["objects"].items():
                        randomization_manager.original_positions[demo_key][f"obj_{obj_name}"] = {
                            "x": float(obj_state["pos"][0]),
                            "y": float(obj_state["pos"][1]),
                            "z": float(obj_state["pos"][2]),
                        }

                if "robots" in init_state:
                    for robot_name, robot_state in init_state["robots"].items():
                        randomization_manager.original_positions[demo_key][f"robot_{robot_name}"] = {
                            "x": float(robot_state["pos"][0]),
                            "y": float(robot_state["pos"][1]),
                            "z": float(robot_state["pos"][2]),
                        }

        total_success = 0
        total_completed = 0
        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)


        for demo_start_idx in range(
            args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        ):
            demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            current_demo_idxs = list(range(demo_start_idx, demo_end_idx))

            # Apply domain randomization before reset
            if randomization_manager is not None and dr_level > 0:
                for env_id, demo_idx in enumerate(current_demo_idxs):
                    log.info(f"[DP Eval] Episode {demo_idx}: Applying DR")
                    randomization_manager.apply_randomization(demo_idx=demo_idx, is_initial=(demo_start_idx == args.task_id_range_low))
                    randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=env_id)
                    randomization_manager.update_camera_look_at(env_id=env_id)
                    randomization_manager.apply_camera_randomization()

            tic = time.time()
            obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
            toc = time.time()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            # Ensure environment stabilizes after reset
            if randomization_manager is not None and dr_level > 0:
                ensure_clean_state(env.handler)

                if hasattr(env, "_episode_steps"):
                    for env_id in range(num_envs):
                        env._episode_steps[env_id] = 0

            policyRunner.reset()

            step = 0
            MaxStep = args.max_step
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            images_list = []
            print(policyRunner.policy_cfg)

            while step < MaxStep:
                log.debug(f"Step {step}")

                new_obs = {
                    "rgb": obs.cameras["camera0"].rgb,
                    "joint_qpos": obs.robots[args.robot].joint_pos,
                }

                images_list.append(np.array(new_obs["rgb"].cpu()))
                action = policyRunner.get_action(new_obs)

                for round_i in range(action_set_steps):
                    obs, reward, success, time_out, extras = env.step(action)

                # eval
                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                step += 1
                if all(SuccessOnce):
                    break

            SuccessEnd = success.tolist()
            total_success += SuccessOnce.count(True)
            total_completed += len(SuccessOnce)
            os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
            for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
                demo_idx_str = str(demo_idx).zfill(4)
                if i % args.save_video_freq == 0:
                    iio.mimwrite(
                        f"tmp/{ckpt_name}/{demo_idx}.mp4",
                        [images[i] for images in images_list],
                    )
                with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                    f.write(f"Demo Index: {demo_idx}\n")
                    f.write(f"Num Envs: {num_envs}\n")
                    f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                    f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                    f.write(f"TimeOut: {TimeOut[i]}\n")
                    f.write(f"Domain Randomization Level: {dr_level}\n")
                    f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
                    f.write(f"Domain Randomization Seed: {dr_seed}\n")
                    f.write(
                        f"Cumulative Average Success Rate: {total_success / total_completed:.4f}\n"
                    )
            log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
            log.info("Num Envs: ", num_envs)
            log.info(f"SuccessOnce: {SuccessOnce}")
            log.info(f"SuccessEnd: {SuccessEnd}")
            log.info(f"TimeOut: {TimeOut}")
        log.info(f"FINAL RESULTS: Average Success Rate = {total_success / total_completed:.4f}")
        with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
            f.write(f"Total Success: {total_success}\n")
            f.write(f"Total Completed: {total_completed}\n")
            f.write(f"Average Success Rate: {total_success / total_completed:.4f}\n")
            f.write(f"Domain Randomization Level: {dr_level}\n")
            f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
            f.write(f"Domain Randomization Seed: {dr_seed}\n")
        env.close()

    def run(
        self,
        train=None,
        eval=None,
        ckpt_path=None,
    ):
        train = self.cfg.train_enable
        eval = self.cfg.eval_enable
        if not train:
            ckpt_path = self.cfg.eval_path
        if train:
            self.train()
        if eval:
            self.evaluate(ckpt_path=ckpt_path)


class BatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    # print("create_dataloader_batch_size", batch_size)
    batch_sampler = BatchSampler(
        len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = DPRunner(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
