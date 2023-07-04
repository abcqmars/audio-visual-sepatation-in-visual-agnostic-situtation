import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .synthesizer_net import InnerProd, Bias
from .audio_net import Unet
from .vision_net import ResnetFC, ResnetDilated
from .criterion import BCELoss, L1Loss, L2Loss, PitWrapper

# Lib for loading motion net.
import mmcv
from mmcv import Config
from mmaction.models import build_backbone

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet5', fc_dim=64, weights='', fusion_type = "con_motion", att_type="cos"):
        # 2D models
        if arch == 'unet5':
            net_sound = Unet(fc_dim=fc_dim, num_downs=5, fusion_type=fusion_type, att_type=att_type)
        elif arch == 'unet6':
            net_sound = Unet(fc_dim=fc_dim, num_downs=6, fusion_type=fusion_type, att_type=att_type)
        elif arch == 'unet7':
            net_sound = Unet(fc_dim=fc_dim, num_downs=7, fusion_type=fusion_type, att_type=att_type)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))
        return net_sound

    # builder for vision
    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool',
                    weights=''):
        pretrained=True
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetFC(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_synthesizer(self, arch, fc_dim=64, weights=''):
        if arch == 'linear':
            net = InnerProd(fc_dim=fc_dim)
        elif arch == 'bias':
            net = Bias()
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net

    def build_motion(self):
        backbone_dict=dict(
            type='ResNet3dFastOnly',
            pretrained=None,
            resample_rate=8,  # tau
            speed_ratio=8,  # alpha
            channel_ratio=8,  # beta_inv
            fast_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=False))
        model = build_backbone(backbone_dict)
        ckp_pth = "/mnt/data2/he/src/mmaction2/checkpoints/slowfast_r50_4x16x1_256e_kinetics400_rgb_20210722-04e43ed4.pth"
        ckp = torch.load(ckp_pth)
        new_dict = {k:v for k, v in ckp['state_dict'].items() if "cls_head" not in k and "slow_path" not in k}
        # Change the name of key.
        new_dict = {k.replace("backbone.", ""):v for k, v in new_dict.items()}
        model.load_state_dict(new_dict, strict=True)
        return model
    

    def build_criterion(self, arch, use_pit = False):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        if use_pit:
            net = PitWrapper(F.binary_cross_entropy)
        return net
