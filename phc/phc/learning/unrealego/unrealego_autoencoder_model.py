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


class UnrealEgoAutoEncoderModel(BaseModel):
    def name(self):
        return 'UnrealEgo AutoEncoder model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = [
            'heatmap_left_rec', 'heatmap_right_rec', 
            'pose', 'cos_sim', 
        ]

        if self.isTrain:
            self.visual_names = [
                'input_rgb_left', 'input_rgb_right',
                'pred_heatmap_left', 'pred_heatmap_right',
                'gt_heatmap_left', 'gt_heatmap_right',
                'pred_heatmap_left_rec', 'pred_heatmap_right_rec'
            ]
        else:
            self.visual_names = [
                # 'input_rgb_left', 'input_rgb_right',
                'pred_heatmap_left', 'pred_heatmap_right',
                'gt_heatmap_left', 'gt_heatmap_right',
            ]

        self.visual_pose_names = [
            "pred_pose", "gt_pose"
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap', 'AutoEncoder']
        else:
            self.model_names = ['HeatMap', 'AutoEncoder']

        self.eval_key = "mpjpe"
        self.cm2mm = 10


        # define the transform network
        self.net_HeatMap = network.define_HeatMap(opt, model=opt.model)
        self.net_AutoEncoder = network.define_AutoEncoder(opt, model=opt.model)

        self.load_networks(
            net=self.net_HeatMap, 
            path_to_trained_weights=opt.path_to_trained_heatmap
            )
        network._freeze(self.net_HeatMap)

        # define loss functions
        self.lossfunc_MSE = MSELoss()
        self.lossfunc_limb = LossFuncLimb()
        self.lossfunc_cos_sim = LossFuncCosSim()
        self.lossfunc_MPJPE = LossFuncMPJPE()

        if self.isTrain:
            # initialize optimizers
            self.optimizer_AutoEncoder = torch.optim.Adam(
                params=self.net_AutoEncoder.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_AutoEncoder)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)
        self.gt_pose = data['gt_local_pose'].cuda(self.device)

    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            # estimate stereo heatmaps
            with torch.no_grad():
                pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
                self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)

            # estimate pose and reconstruct stereo heatmaps
            self.pred_pose, pred_heatmap_rec_cat = self.net_AutoEncoder(pred_heatmap_cat)
            self.pred_heatmap_left_rec, self.pred_heatmap_right_rec = torch.chunk(pred_heatmap_rec_cat, 2, dim=1)
    
    def backward_AutoEncoder(self):
        with autocast(enabled=self.opt.use_amp):
            loss_pose = self.lossfunc_MPJPE(self.pred_pose, self.gt_pose)
            loss_cos_sim = self.lossfunc_cos_sim(self.pred_pose, self.gt_pose)
            loss_heatmap_left_rec = self.lossfunc_MSE(
                self.pred_heatmap_left_rec, self.pred_heatmap_left.detach()
            )
            loss_heatmap_right_rec = self.lossfunc_MSE(
                self.pred_heatmap_right_rec, self.pred_heatmap_right.detach()
            )

            self.loss_pose = loss_pose * self.opt.lambda_mpjpe
            self.loss_cos_sim = loss_cos_sim * self.opt.lambda_cos_sim * self.opt.lambda_mpjpe
            self.loss_heatmap_left_rec = loss_heatmap_left_rec * self.opt.lambda_heatmap_rec
            self.loss_heatmap_right_rec = loss_heatmap_right_rec * self.opt.lambda_heatmap_rec

            loss_total = self.loss_pose + self.loss_cos_sim + \
                self.loss_heatmap_left_rec + self.loss_heatmap_right_rec

        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_AutoEncoder.train()
        
        # set optimizer.zero_grad()
        self.optimizer_AutoEncoder.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_AutoEncoder()

        # optimizer step
        self.scaler.step(self.optimizer_AutoEncoder)

        self.scaler.update()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.net_HeatMap.eval()
        self.net_AutoEncoder.eval()

        # forward pass
        pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
        self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)
        self.pred_pose = self.net_AutoEncoder.predict_pose(pred_heatmap_cat)

        S1_hat = batch_compute_similarity_transform_torch(self.pred_pose, self.gt_pose)

        # compute metrics
        for id in range(self.pred_pose.size()[0]):  # batch size
            # calculate mpjpe and p_mpjpe   # cm to mm
            mpjpe = self.lossfunc_MPJPE(self.pred_pose[id], self.gt_pose[id]) * self.cm2mm
            pa_mpjpe = self.lossfunc_MPJPE(S1_hat[id], self.gt_pose[id]) * self.cm2mm

            # update metrics dict
            runnning_average_dict.update(dict(
                mpjpe=mpjpe, 
                pa_mpjpe=pa_mpjpe)
            )

        return runnning_average_dict


