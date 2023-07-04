from glob import glob
from nis import match
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import fusion_net


class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False, extra_size = 32):
        super(Unet, self).__init__()

        # construct unet structure
        # unet_block = UnetBlock(
        #     ngf * 8, ngf * 8, input_nc=None,
        #     submodule=None, innermost=True)

        unet_block = InnerUnetBlock(ngf * 8, ngf * 8, extra_size= extra_size)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block

    def forward(self, x, v=None):
        x = self.bn0(x)
        x, meta = self.unet_block(x, v)
        return x, meta


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.down_forward = nn.Sequential(*down)
        self.mid_forward = submodule
        self.up_forward = nn.Sequential(*up)
        # self.model = nn.Sequential(*model)

    def forward(self, x, v=None):
        if self.outermost or self.noskip:
            x = self.down_forward(x)
            x, meta = self.mid_forward(x, v)
            x = self.up_forward(x)
            return x, meta
        else:
            xd = self.down_forward(x)
            xm, meta = self.mid_forward(xd, v)
            xu = self.up_forward(xm)
            return torch.cat([x, xu], 1), meta # torch.cat([x, self.model(x, v)], 1)


class Attention(nn.Module):
    def __init__(self, in_cn, out_cn):
        super(Attention, self).__init__()
        self.proj_q = nn.Linear(in_cn, out_cn)
        self.proj_k = nn.Linear(in_cn, out_cn)
        self.proj_v = nn.Linear(in_cn, out_cn)

        def cal_co(q, k):
            # inner dotproduct
            # B, N, D
            # B, D, N
            return q.bmm(k.permute(0, 2, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.co_map = cal_co
    
    def forward(self, q, k, v):
        # q, k, v: B, K, D
        proj_q = q # self.proj_q(q)
        proj_k = self.proj_k(k)
        proj_v = v #
        # proj_v = self.proj_k(v)
        att = self.softmax(self.co_map(proj_q, proj_k) )
        return att.bmm(proj_v)



class InnerUnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None, use_dropout=False, inner_output_nc=None, noskip=False, extra_size = 32):
        super(InnerUnetBlock, self).__init__()
        # input_nc * inner_input_nc 
        # inner_output_nc * outer_nc

        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        inner_output_nc = inner_input_nc
        
        C = 2
        inner_input_nc += extra_size * C
        self.extra_size = extra_size
        self.inner_out_size = inner_output_nc
        
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(False)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        downconv = nn.Conv2d(
            input_nc, inner_input_nc, kernel_size=4,
            stride=2, padding=1, bias=use_bias)
        upconv = nn.Conv2d(
            inner_output_nc, outer_nc, kernel_size=3,
            padding=1, bias=use_bias)

        down = [downrelu, downconv]
        up = [uprelu, upsample, upconv, upnorm]

        self.down_forward = nn.Sequential(*down)
        # self.freq_att = Attention(inner_input_nc + 512, inner_input_nc)
        self.up_forward = nn.Sequential(*up)

    def forward(self, x, v = None):
        # x: B, D, F, T.
        # v: B, D
        xin = x
        x = self.down_forward(x)
        # x: B x (D+extra_size*2) x F x T
        x_blocks = torch.split(x, [self.extra_size*2, self.inner_out_size], dim=1)
        x = x_blocks[-1]
        x = self.up_forward(x)
        return torch.cat([xin, x], 1), (x_blocks[0], )
