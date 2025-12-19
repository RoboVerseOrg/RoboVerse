import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
import json
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf

from .utils import load_data  # data functions
from .utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from .policy import ACTPolicy, CNNMLPPolicy
from roboverse_learn.il.runners.base_runner import BaseRunner

import IPython
e = IPython.embed
from datetime import datetime


class ACTRunner(BaseRunner):
    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Determine device
        self.device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set seed
        set_seed(cfg['seed'])

        # Policy configuration
        self.policy_class = cfg['policy_class']
        self.policy_config = cfg['policy_config']

        # Initialize policy and optimizer
        self.policy = self._make_policy()
        self.policy.to(self.device)
        self.optimizer = self._make_optimizer()

    def _make_policy(self):
        if self.policy_class == 'ACT':
            policy = ACTPolicy(self.policy_config)
        elif self.policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(self.policy_config)
        else:
            raise NotImplementedError
        return policy

    def _make_optimizer(self):
        if self.policy_class == 'ACT':
            optimizer = self.policy.configure_optimizers()
        elif self.policy_class == 'CNNMLP':
            optimizer = self.policy.configure_optimizers()
        else:
            raise NotImplementedError
        return optimizer

    def forward_pass(self, data):
        image_data, qpos_data, action_data, is_pad = data
        image_data = image_data.to(self.device)
        qpos_data = qpos_data.to(self.device)
        action_data = action_data.to(self.device)
        is_pad = is_pad.to(self.device)
        return self.policy(qpos_data, image_data, action_data, is_pad)

    def train(self):
        cfg = self.cfg
        num_epochs = cfg['num_epochs']
        ckpt_dir = self.output_dir
        seed = cfg['seed']

        # Load data
        train_dataloader, val_dataloader, stats, _ = load_data(
            cfg['dataset_dir'],
            cfg['num_episodes'],
            cfg['camera_names'],
            cfg['batch_size'],
            cfg['batch_size']
        )

        # Save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        # Save config to cfg.yaml
        config_path = os.path.join(ckpt_dir, 'cfg.yaml')
        with open(config_path, 'w') as f:
            # Convert OmegaConf to dict if needed, or just dump the dict
            if isinstance(cfg, (dict, list)):
                yaml.dump(cfg, f, default_flow_style=False)
            else:
                yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None

        for epoch in tqdm(range(num_epochs)):
            print(f'\nEpoch {epoch}')
            # Validation
            with torch.inference_mode():
                self.policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = self.forward_pass(data)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(self.policy.state_dict()))

            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            # Training
            self.policy.train()
            self.optimizer.zero_grad()
            epoch_train_dicts = []
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = self.forward_pass(data)
                # Backward
                loss = forward_dict['loss']
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_train_dicts.append(forward_dict)

            epoch_train_summary = compute_dict_mean(epoch_train_dicts)
            train_history.append(epoch_train_summary)
            epoch_train_loss = epoch_train_summary['loss']
            print(f'Train loss: {epoch_train_loss:.5f}')

            summary_string = ''
            for k, v in epoch_train_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            if epoch % 100 == 0:
                ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
                torch.save(self.policy.state_dict(), ckpt_path)
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(self.policy.state_dict(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

        # Save training curves
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

        file_path = os.path.join("./roboverse_learn/il/policies/act", "ckpt_dir_path.txt")
        with open(file_path, 'w') as f:
            f.write(ckpt_dir)

        return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def main(args):
    # Prepare configuration
    policy_class = args['policy_class']
    task_name = args['task_name']
    camera_names = args['camera_names']

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': args['state_dim']
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError

    level = args.get('level', 0)
    ckpt_dir = f"il_outputs/act/{task_name}/ckpt/level{level}"

    # Load metadata from dataset directory
    dataset_dir = args['dataset_dir']
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    dataset_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            dataset_metadata = json.load(f)

    config = {
        'num_epochs': args['num_epochs'],
        'episode_len': args['episode_len'],
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': True,
        'data': dataset_metadata,
        'dataset_dir': dataset_dir,
        'num_episodes': args['num_episodes'],
        'batch_size': args['batch_size'],
        'device': args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Convert to OmegaConf for compatibility with BaseRunner if needed,
    # but BaseRunner doesn't strictly enforce OmegaConf type for simple access.
    # However, it's better practice to wrapping it if we were fully migrating.
    # For now, passing the dict is fine as long as we access with [] or .get()

    runner = ACTRunner(config, output_dir=ckpt_dir)
    runner.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--camera_names', nargs='+', type=str, default=['head_camera'], help='camera names')
    parser.add_argument('--episode_len', action='store', type=int, default=400, help='episode length')

    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', required=False, default=9)
    parser.add_argument('--level', action='store', type=str, help='level', required=False, default=0)
    parser.add_argument('--device', action='store', type=str, help='device', required=False, default='cuda')

    main(vars(parser.parse_args()))
