if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
import pickle
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
#from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy
from diffusion_policy.policy.tedi_transformer_lowdim_policy import TEDiTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
#from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
OmegaConf.register_new_resolver("eval", eval, replace=True)

class HumanoidWorkspaceTransformer(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: TEDiTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: TEDiTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        accelerator = Accelerator(log_with='wandb')#,mixed_precision='fp16')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        wandb_cfg["dir"] = str(self.output_dir)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg,"resume": "never","id": wandb.util.generate_id(),"allow_val_change": True}
        )
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = cfg.training.resume_ckpt
            lastest_ckpt_path = pathlib.Path(lastest_ckpt_path)
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        #normalizer = dataset.get_normalizer()

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
#        env_runner: BaseLowdimRunner
#        env_runner = hydra.utils.instantiate(
#            cfg.task.env_runner,
#            output_dir=self.output_dir)
#        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        #wandb_run = wandb.init(
        #    dir=str(self.output_dir),
        #    config=OmegaConf.to_container(cfg, resolve=True),
        #    **cfg.logging
        #)
        #wandb.config.update(
        #    {
        #        "output_dir": self.output_dir,
        #    }
        #)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        #device = torch.device(cfg.training.device)
        #self.model.to(device)
        #if self.ema_model is not None:
        #    self.ema_model.to(device)
        #optimizer_to(self.optimizer, device)


        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler)
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        #loss.backward()
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # # run rollout
                # TODO: need to write a Humanoid env_runner and validation! now I comment them all
                # # if (self.epoch % cfg.training.rollout_every) == 0:
                # #     policy.reset_buffer()
                # #     runner_log = env_runner.run(policy)
                # #     # log all
                # #     step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 2  and accelerator.is_main_process:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss
                do_validation = (self.epoch % cfg.training.val_every) == 0
                if do_validation:
                    # 1. 每个进程都跑验证循环
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model(batch)
                            val_losses.append(loss)

                    # 2. gather 或者 all_reduce 所有进程的 val_losses
                    avg_val_loss = accelerator.gather_for_metrics(torch.mean(torch.stack(val_losses)))
                    # 3. 只有 rank=0 输出日志
                    if accelerator.is_main_process:
                        val_loss = avg_val_loss.item()
                        step_log['val_loss'] = val_loss
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']

                        policy.reset_buffer()
                        result = policy.predict_action(obs_dict)
                        # In contrast to Diffusion Policy, we are only interested in the validation loss on the predicted actions
                        # since the next actions are noisy
                        pred_action = result['action']

                        # Cut GT to be (B, Ta, Da)
                        start = cfg.n_obs_steps - 1
                        end = start + cfg.n_action_steps
                        gt_action = gt_action[:,start:end]

                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # checkpointing
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(tag=f'epoch_{self.epoch}_step_{self.global_step}')
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # TODO Perkins: shoul save top k ckpts in the future !

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                #policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                #wandb_run.log(step_log, step=self.global_step)
                #json_logger.log(step_log)
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TEDiTransformerLowdimPolicy(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
