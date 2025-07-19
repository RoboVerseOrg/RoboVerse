from cProfile import run
from enum import auto
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss

import itertools
from .base_model import BaseModel
from . import network
from utils.loss import LossFuncLimb, LossFuncCosSim, LossFuncMPJPE
from utils.util import batch_compute_similarity_transform_torch


class UnrealEgoHeatmapSharedModel(BaseModel):
    def name(self):
        return 'UnrealEgo Heatmap Shared model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = [
            'heatmap_left', 'heatmap_right', 
        ]

        self.visual_names = [
            'input_rgb_left', 'input_rgb_right',
            'pred_heatmap_left', 'pred_heatmap_right',
            'gt_heatmap_left', 'gt_heatmap_right',
        ]

        self.visual_pose_names = [
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap']
        else:
            self.model_names = ['HeatMap']

        self.eval_key = "mse_heatmap"
        self.cm2mm = 10


        # define the transform network
        print(opt.model)
        self.net_HeatMap = network.define_HeatMap(opt, model=opt.model)

        if self.isTrain:
            # define loss functions
            self.lossfunc_MSE = MSELoss()

            # initialize optimizers
            self.optimizer_HeatMap = torch.optim.Adam(
                params=self.net_HeatMap.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_HeatMap)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)

    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            # estimate stereo heatmaps
            pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
            self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)

    def backward_HeatMap(self):
        with autocast(enabled=self.opt.use_amp):
            loss_heatmap_left = self.lossfunc_MSE(
                self.pred_heatmap_left, self.gt_heatmap_left
            )
            loss_heatmap_right = self.lossfunc_MSE(
                self.pred_heatmap_right, self.gt_heatmap_right
            )
            
            self.loss_heatmap_left = loss_heatmap_left * self.opt.lambda_heatmap
            self.loss_heatmap_right = loss_heatmap_right * self.opt.lambda_heatmap
            
            loss_total = self.loss_heatmap_left + self.loss_heatmap_right

        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_HeatMap.train()
        
        # set optimizer.zero_grad()
        self.optimizer_HeatMap.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_HeatMap()

        # optimizer step
        self.scaler.step(self.optimizer_HeatMap)

        self.scaler.update()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.net_HeatMap.eval()

        # forward pass
        pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
        self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)
        
        # compute metrics
        for id in range(self.pred_heatmap_left.size()[0]):  # batch size
            # calculate mse loss for heatmap
            loss_heatmap_left_id = self.lossfunc_MSE(
                self.pred_heatmap_left[id], self.gt_heatmap_left[id]
            )
            loss_heatmap_right_id = self.lossfunc_MSE(
                self.pred_heatmap_right[id], self.gt_heatmap_right[id]
            )
            
            mse_heatmap = loss_heatmap_left_id + loss_heatmap_right_id

            # update metrics dict
            runnning_average_dict.update(dict(
                mse_heatmap=mse_heatmap
                )
            )

        return runnning_average_dict