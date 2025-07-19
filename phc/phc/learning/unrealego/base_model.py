from operator import contains
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import util


class BaseModel(nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.log_dir, opt.experiment_name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.visual_pose_names = []
        self.image_paths = []
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    
    def set_input(self, input):
        self.input = input

    # update learning rate
    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


    # return training loss
    def get_current_errors(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    # return visualization images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)

                if "heatmap" in name:
                    is_heatmap = True
                else:
                    is_heatmap = False

                visual_ret[name] = util.tensor2im(value.data, is_heatmap=is_heatmap)

                # if isinstance(value, list):
                #     visual_ret[name] = util.tensor2im(value[-1].data, is_heatmap)
                # else:
                #     visual_ret[name] = util.tensor2im(value.data, is_heatmap)

        return visual_ret

    # save models
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    # load models
    def load_networks(self, which_epoch=None, net=None, path_to_trained_weights=None):
        if which_epoch is not None:
            for name in self.model_names:
                print(name)
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (which_epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_'+name)
                    state_dict = torch.load(save_path)
                    net.load_state_dict(state_dict)
                    # net.load_state_dict(self.fix_model_state_dict(state_dict))
                    if not self.isTrain:
                        net.eval()
        else:
            state_dict = torch.load(path_to_trained_weights)
            if self.opt.distributed:
                net.load_state_dict(self.fix_model_state_dict(state_dict))
            else:
                net.load_state_dict(state_dict)
            print('Loaded pre_trained {}'.format(os.path.basename(path_to_trained_weights)))

    def fix_model_state_dict(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict