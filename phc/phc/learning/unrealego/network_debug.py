from re import X
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import weight_norm
import functools
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict
import math


######################################################################################
# Functions
######################################################################################
def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch+1+1+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            lr_l = 1.0 - max(0, epoch+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters_step, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, gain, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network_param(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('total number of parameters of {}: {:.3f} M'.format(name, num_params / 1e6))


def init_net(net, init_type='normal', gpu_ids=[], init_ImageNet=True):

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()

    if init_ImageNet is False:
        init_weights(net, init_type)
    else:
        init_weights(net.after_backbone, init_type)
        print('   ... also using ImageNet initialization for the backbone')

    return net


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False


######################################################################################
# Define networks
######################################################################################

def define_HeatMap(opt, model):

    if model == 'egoglass':
        net = HeatMap_EgoGlass(opt)
    elif model == "unrealego_heatmap_shared":
        net = HeatMap_UnrealEgo_Shared(opt)
    elif model == "unrealego_autoencoder":
        net = HeatMap_UnrealEgo_Shared(opt)

    print_network_param(net, 'HeatMap_Estimator for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, opt.init_ImageNet)

def define_AutoEncoder(opt, model):

    if model == 'egoglass':
        net = AutoEncoder(opt, input_channel_scale=2)
    elif model == "unrealego_autoencoder":
        net = AutoEncoder(opt, input_channel_scale=2)

    print_network_param(net, 'AutoEncoder for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, False)


######################################################################################
# Basic Operation
######################################################################################


def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding)
    # torch.nn.init.xavier_normal_(conv.weight)
    # conv = weight_norm(conv)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_deconv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
    # torch.nn.init.xavier_normal_(conv.weight)
    # conv = weight_norm(conv)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_fc_layer(in_feature, out_feature, with_relu=True, with_bn=True):
    modules = OrderedDict()
    fc = torch.nn.Linear(in_feature, out_feature)
    # torch.nn.init.xavier_normal_(fc.weight)
    # fc = weight_norm(fc)
    modules['fc'] = fc
    bn = torch.nn.BatchNorm1d(num_features=out_feature)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)

    if with_bn is True:
        modules['bn'] = bn
    else:
        print('no bn')

    if with_relu is True:
        modules['relu'] = relu
    else:
        print('no pose relu')

    return torch.nn.Sequential(modules)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

######################################################################################
# Network structure
######################################################################################


############################## EgoGlass ##############################


class HeatMap_EgoGlass(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_EgoGlass, self).__init__()

        self.backbone = HeatMap_EgoGlass_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_EgoGlass_AfterBackbone(opt)

    def forward(self, input):
        
        x = self.backbone(input)
        output = self.after_backbone(x)

        return output


class HeatMap_EgoGlass_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_EgoGlass_Backbone, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output


class HeatMap_EgoGlass_AfterBackbone(nn.Module):
    def __init__(self, opt):
        super(HeatMap_EgoGlass_AfterBackbone, self).__init__()

        self.num_heatmap = opt.num_heatmap

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_heatmap = nn.Conv2d(256, self.num_heatmap, 1)


    def forward(self, list_input):

        input = list_input[0]
        layer0 = list_input[1]
        layer1 = list_input[2]
        layer2 = list_input[3]
        layer3 = list_input[4]
        layer4 = list_input[5]

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        output = self.conv_heatmap(x)

        return output


############################## UnrealEgo ##############################

class HeatMap_UnrealEgo_Shared(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared, self).__init__()

        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name)

    def forward(self, input_left, input_right):

        x_left, x_right = self.backbone(input_left, input_right)
        output = self.after_backbone(x_left, x_right)

        return output


class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()

        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, input_left, input_right):
        output_left = self.backbone(input_left)
        output_right = self.backbone(input_right)

        return output_left, output_right

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
            self.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output


class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    def __init__(self, opt, model_name="resnet18"):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        if model_name == 'resnet18':
            feature_scale = 1
        elif model_name == "resnet34":
            feature_scale = 1
        elif model_name == "resnet50":
            feature_scale = 4
        elif model_name == "resnet101":
            feature_scale = 4
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.num_heatmap = opt.num_heatmap

        # self.layer0_1x1 = convrelu(128, 128, 1, 0)
        self.layer1_1x1 = convrelu(128 * feature_scale, 128 * feature_scale, 1, 0)
        self.layer2_1x1 = convrelu(256 * feature_scale, 256 * feature_scale, 1, 0)
        self.layer3_1x1 = convrelu(512 * feature_scale, 516 * feature_scale, 1, 0)
        self.layer4_1x1 = convrelu(1024 * feature_scale, 1024 * feature_scale, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(516 * feature_scale + 1024 * feature_scale, 1024 * feature_scale, 3, 1)
        self.conv_up2 = convrelu(256 * feature_scale + 1024 * feature_scale, 512 * feature_scale, 3, 1)
        self.conv_up1 = convrelu(128 * feature_scale + 512 * feature_scale, 512 * feature_scale, 3, 1)

        self.conv_heatmap = nn.Conv2d(512 * feature_scale, self.num_heatmap * 2, 1)

    def forward(self, list_input_left, list_input_right):
        list_stereo_feature = [
            torch.cat([list_input_left[id], list_input_right[id]], dim=1) for id in range(len(list_input_left))
        ]

        input = list_stereo_feature[0]
        layer0 = list_stereo_feature[1]
        layer1 = list_stereo_feature[2]
        layer2 = list_stereo_feature[3]
        layer3 = list_stereo_feature[4]
        layer4 = list_stereo_feature[5]

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        output = self.conv_heatmap(x)

        return output


############################## AutoEncoder ##############################


class AutoEncoder(nn.Module):

    def __init__(self, opt, input_channel_scale=1, fc_dim=16384):
        super(AutoEncoder, self).__init__()

        self.hidden_size = opt.ae_hidden_size
        self.with_bn = True
        self.with_pose_relu = True

        self.num_heatmap = opt.num_heatmap
        self.channels_heatmap = self.num_heatmap * input_channel_scale
        self.fc_dim = fc_dim

        self.conv1 = make_conv_layer(in_channels=self.channels_heatmap, out_channels=64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2 = make_conv_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3 = make_conv_layer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        # self.fc1 = make_fc_layer(in_feature=18432, out_feature=2048, with_bn=self.with_bn)
        self.fc1 = make_fc_layer(in_feature=self.fc_dim, out_feature=2048, with_bn=self.with_bn)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=self.with_bn)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=self.hidden_size, with_bn=self.with_bn)

        ## pose decoder
        self.pose_fc1 = make_fc_layer(self.hidden_size, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc2 = make_fc_layer(32, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc3 = torch.nn.Linear(32, (self.num_heatmap + 1) * 3)

        # heatmap decoder
        self.heatmap_fc1 = make_fc_layer(self.hidden_size, 512, with_bn=self.with_bn)
        self.heatmap_fc2 = make_fc_layer(512, 2048, with_bn=self.with_bn)
        # self.heatmap_fc3 = make_fc_layer(2048, 18432, with_bn=self.with_bn)
        self.heatmap_fc3 = make_fc_layer(2048, self.fc_dim, with_bn=self.with_bn)
        self.WH = int(math.sqrt(self.fc_dim/256))  

        self.deconv1 = make_deconv_layer(256, 128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv2 = make_deconv_layer(128, 64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv3 = make_deconv_layer(64, self.channels_heatmap, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

    def predict_pose(self, input):
        batch_size = input.size()[0]    

        # encode heatmap
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)

        # decode pose
        x_pose = self.pose_fc1(z)
        x_pose = self.pose_fc2(x_pose)
        output_pose = self.pose_fc3(x_pose)

        return output_pose.view(batch_size, self.num_heatmap + 1, 3)


    def forward(self, input):
        batch_size = input.size()[0]

        # encode heatmap
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)

        # decode pose
        x_pose = self.pose_fc1(z)
        x_pose = self.pose_fc2(x_pose)
        output_pose = self.pose_fc3(x_pose)

        # decode heatmap
        x_hm = self.heatmap_fc1(z)
        x_hm = self.heatmap_fc2(x_hm)
        x_hm = self.heatmap_fc3(x_hm)
        x_hm = x_hm.view(batch_size, 256, self.WH, self.WH)
        x_hm = self.deconv1(x_hm)
        x_hm = self.deconv2(x_hm)
        output_hm = self.deconv3(x_hm)

        return output_pose.view(batch_size, self.num_heatmap + 1, 3), output_hm


if __name__ == "__main__":
    
    model = HeatMap_UnrealEgo_Shared(opt=None, model_name='resnet50')

    input = torch.rand(3, 3, 256, 256)
    outputs = model(input, input)
    pred_heatmap_left, pred_heatmap_right = torch.chunk(outputs, 2, dim=1)

    print(pred_heatmap_left.size())
    print(pred_heatmap_right.size())
    
