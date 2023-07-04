from glob import glob
from nis import match
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import fusion_net


class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False, fusion_type = "con_motion", att_type="cos"):
        super(Unet, self).__init__()

        # construct unet structure
        # unet_block = UnetBlock(
        #     ngf * 8, ngf * 8, input_nc=None,
        #     submodule=None, innermost=True)

        unet_block = InnerUnetBlock(ngf * 8, ngf * 8, fusion_type = fusion_type, att_type=att_type)
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
    def __init__(self, outer_nc, inner_input_nc, input_nc=None, use_dropout=False, inner_output_nc=None, noskip=False, fusion_type = "con_motion", att_type='cos'):
        super(InnerUnetBlock, self).__init__()
        # input_nc * inner_input_nc 
        # inner_output_nc * outer_nc

        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        inner_output_nc = inner_input_nc
        if fusion_type == "con_motion" or fusion_type == "hidsep" or fusion_type == "share" or fusion_type=="CoLoc_Sel" or fusion_type=="MixVis" or fusion_type=="CoLoc_ClipAtt":
            inner_output_nc *= 2
        elif fusion_type == "CoLoc_Double":
            inner_input_nc *= 2
            inner_output_nc *= 2
            # print(inner_input_nc)
        elif fusion_type == "vis2in":
            inner_output_nc = int( inner_output_nc * 3 / 2 ) 
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
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
        AV_fusion = fusion_net.get_fusion_net(fusion_type)
        self.fusion = AV_fusion(att_type=att_type)
        # self.model = nn.Sequential(*model)

    def forward(self, x, v = None):
        # x: B, D, F, T.
        # v: B, D
        xin = x
        x = self.down_forward(x)

        x, meta = self.fusion(x, v)
        
        x = self.up_forward(x)
        return torch.cat([xin, x], 1), meta

# class AV_fusion(nn.Module):
#     def __init__(self, mtype, **kwargs):
#         super(AV_fusion, self).__init__()
#         self.mtype = mtype
#         if mtype == "con":
#             self.init_concate(**kwargs)
#         elif mtype == "con3":
#             self.init_concate3(**kwargs)
#         elif mtype == "vis2in":
#             self.init_vis2in(**kwargs)
#         elif mtype == "a2v_att":
#             self.init_a2v_att(**kwargs)
#         elif mtype == "hidsep":
#             self.init_hidsep(**kwargs)
#         elif mtype == "share":
#             self.init_share(**kwargs)
    
#     def init_share(self, **kwargs):
#         self.max_pool = torch.nn.AdaptiveMaxPool3d((None, 1, 1))
    
#     def init_a2v_att(self, **kwargs):
#         # D = kwargs['input_D']
#         # self.att = nn.MultiheadAttention(D, 4, batch_first = True)
#         # None
#         self.max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

#     def init_concate3(self, **kwargs):
#         self.max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
#         # None
    
#     def init_vis2in(self, **kwargs):
#         D = kwargs['input_D']
#         self.att = nn.MultiheadAttention(int(D/2), 4, batch_first = True)
    
#     def init_hidsep(self, **kwargs):
#         self.avg_pool = torch.nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.max_pool = torch.nn.AdaptiveMaxPool3d((None, 1, 1))
#         # self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        
#     def av_att_forward(self, x, v):
#         # x: B, D, F, T
#         # v: B, D
#         B, D, F, T = x.shape
#         x_ctx = None
#         if v is None:
#             x_ctx = torch.mean(torch.mean(x, dim=-1), dim=-1) # B, D
#             v = x_ctx
#         x_in = x.view(B, D, -1).permute(0, 2, 1) # B, FT, D
#         v = v.unsqueeze(1) # B, 1, D
#         x_out, _ = self.att(v, x_in, x_in)# , weights
#         x_out = (x_out + v).squeeze()
#         x_out = x_out.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape)
#         return torch.cat((x_out, x), dim = 1), x_ctx
    
#     # Baseline model.
#     def concate3_forward(self, x, v):
#         # x: B, D, F, T.
#         # v: B, D
#         # return:
#         # x: B, 2D, F, T.
#         B, D, F, T = x.shape
#         x_ctx = None
#         if v is None:
#             x_ctx = self.max_pool(x).squeeze()
#             v = x_ctx
#         v = v.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape)
#         return torch.cat((v, x), dim = 1), x_ctx

#     def hidsep_forward(self, x, v_ls):
#         # x:     B, D,   F, T      # Audioencoder ouput
#         # v_ls: [B, D/C, H, W] * C # C visual inputs.
#         B, D, F, T = x.shape
#         # Extract global audio vector.
#         C = 2
#         x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
#         # x_blocks = torch.split(self.max_pool(x), int(D/C), dim=1) # [ B, D/C, 1, 1] *C
#         if v_ls and isinstance(v_ls[1], bool):
#             # x:     B, D,   F, T      # Audio encoder ouput
#             # v_ls: [B, D/C, H, W*C] * 1 # C visual inputs.
#             B, D, F, T = x.shape
#             v = v_ls[0] # B, D/C, H, W *C
#             C = 2
#             x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1) # [ B, D/C, 1, 1] *C
            
#             # 1. Compute Similarity.
#             x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
#             maps =  torch.nn.functional.cosine_similarity(x_cat, v.unsqueeze(1), dim=2) # B, C, H, W*C
            
#             # 2. Compute attention.
#             attended_v = self.max_pool(v.unsqueeze(1) * maps.unsqueeze(2)).squeeze(-1).squeeze(-1) # B, C, D/C
            
#             # 3. Get scores and matching loss.
#             scores = self.max_pool(maps).squeeze(-1).squeeze(-1) # B, C
#             # print(scores)
#             indices = scores < 0.5 # B, C
#             scores[scores>0] *= -1 if v_ls[1] is True else 1
#             match_loss = scores.sum(-1).mean().reshape(1)
            
#             # 4. replace audio vectors, and random reorder.
#             attended_v[indices] = x_cat.squeeze(-1).squeeze(-1)[indices] # B, C, D/C
#             indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
#             attended_v = torch.gather(attended_v, 1, indices[:, :, None].broadcast_to(attended_v.shape)) # B, C, D/C
#             v_ls = [attended_v[:,i,:].unsqueeze(-1).unsqueeze(-1).broadcast_to((B, int(D/C), F, T)) for i in range(C)]
#             feat_sources = v_ls
#             att_maps = maps
#             return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)
            
#         if v_ls is not None:
            
#             # Audio Visual separation.
            
#             ### 1. Get indices: ###
#             # Get all permutations of x: Below is a special case for C = 2
#             x_p1 = torch.stack(x_blocks, dim=1)       # B, C, D/C, 1, 1
#             x_p2 = torch.stack(x_blocks[::-1], dim=1) # B, C, D/C, 1, 1
#             x_t = torch.stack((x_p1, x_p2), dim=1)    # B, P, C, D/C, 1, 1
            
#             # Get attention maps with all permutation.
#             v_cat = torch.stack(v_ls, dim=1) # B, C, D/C, H, W
#             maps = torch.nn.functional.cosine_similarity(x_t, v_cat.unsqueeze(1), dim=3) # B, P, C, H, W
#             # maps = torch.sigmoid(torch.sum(x_t * v_cat.unsqueeze(1) / (D)**0.5, dim=3))
            
#             # Get scores of attention maps.
#             scores = self.max_pool(maps).squeeze(-1).squeeze(-1).sum(-1) # B, P
#             # Get indeices for reordering.
#             sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # B, P
            
#             ### 2. Compute Match loss ###
#             # maximize the highest score, minimize the rest. 
#             match_loss = -1 * sorted_scores[:, 0] + 1 * sorted_scores[:, 1:].sum(-1)
#             match_loss = match_loss.mean(0)
            
#             ### 3. Reorder ###
#             maps = torch.gather(maps, 1, indices[:,:1][:,:,None,None,None].broadcast_to(maps.shape)) # B, P, C, H, W
#             att_maps = maps[:, 0] # B, C, H, W
#             # x_cat = torch.gather(x_t, 1, indices[:,0][:,None,None,None,None,None].broadcast_to(x_cat.shape)).squeeze(1) #, B, C, D/C, 1, 1
            
#             ### 4. Attend on imgs ###
#             v_cat = self.max_pool(v_cat * att_maps.unsqueeze(2)) # B, C, D/C, 1, 1
#             feat_sources = torch.tensor_split(v_cat, C, dim=1)# [B, 1, D/C, 1, 1] * C
#         else:
#             # Audio Only separation.
#             indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
#             x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
#             x_cat = torch.gather(x_cat, 1, indices[:, :, None, None, None].broadcast_to(x_cat.shape)) # B, C, D/C, 1, 1
#             x_blocks = torch.tensor_split(x_cat, C, dim=1)
#             match_loss = None
#             att_maps = None
#             feat_sources = x_blocks
            
#         feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
#         return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

#     def vis2in_forward(self, x, v):
#         # x: B, D, F, T
#         # v: B, D
#         # return:
#         # out: B, 3D/2, F, T
#         B, D, _, _ = x.shape
#         x_blocks = torch.split(x, int(D/2), dim=1)
#         x_0 = x_blocks[0]
#         x_1 = x_blocks[1]
#         mix_x = (x_0 + x_1)/2 # B, D/2, F, T
#         # => B, F, T, D/2 => B, F * T, D/2 => B, F * T, D/2 => B, F, T, D/2 => # B, D/2, F, T
#         # atted_x, _ = self.att(mix_x, mix_x, mix_x) 
#         # mix_x += atted_x# audio mixture.
#         if v is None:
#             v_0 = torch.mean(torch.mean(x_0, dim=-1), dim=-1)# B, D/2, source 1
#             v_1 = torch.mean(torch.mean(x_1, dim=-1), dim=-1)# B, D/2, source 2
#         else:
#             v_blocks = torch.split(v, int(D/2), dim=1)
#             v_0 = v_blocks[0]# B, D/2, source 1
#             v_1 = v_blocks[1]# B, D/2, source 2
        
#         v_0 = v_0.unsqueeze(-1).unsqueeze(-1).broadcast_to(mix_x.shape)
#         v_1 = v_1.unsqueeze(-1).unsqueeze(-1).broadcast_to(mix_x.shape)
#         return torch.cat((v_0, v_1, mix_x), dim = 1), (v_0, v_1)# B, 3D/2, F, T

#     def a2v_att_forward(self, x, v):
#         # x: B, D, F, T
#         # v: B, D, H, W
#         # glb_x = torch.max(torch.max(x, dim=-2, keepdim=True), dim=-1, keepdim=True) # B, D
#         glb_x = self.max_pool(x) # B, D, 1, 1
#         if v is None:
#             v = glb_x
#             att = None
#         else:
#             B, D, H, W = v.shape
#             # Scale 的作用.
#             att = torch.sigmoid(torch.sum(glb_x * v, dim=1, keepdim=True)/torch.sqrt(torch.tensor(D))) # B, 1, H, W
#             v = self.avg_pool(att * v) + glb_x # B, D, 1, 1
#         v = v.broadcast_to(x.shape)
#         return torch.cat((v, x), dim=1), att
    
#     def share_forward(self, x, v):
#         # x:     B, D,   F, T      # Audio encoder ouput
#         # v_ls: [B, D/C, H, W*C] * 1 # C visual inputs.
#         assert len(v) == 1
#         B, D, F, T = x.shape
#         v = v[0] # B, D/C, H, W *C
#         C = 2
#         x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1) # [ B, D/C, 1, 1] *C
        
#         # 1. Compute Similarity.
#         x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
#         maps =  torch.nn.functional.cosine_similarity(x_cat, v.unsqueeze(1), dim=2) # B, C, H, W*C
        
#         # 2. Compute attention.
#         attended_v = self.max_pool(v.unsqueeze(1) * maps.unsqueeze(2)).squeeze(-1).squeeze(-1) # B, C, D/C
        
#         # 3. Get scores and matching loss.
#         scores = self.max_pool(maps).squeeze(-1).squeeze(-1) # B, C
#         # print(scores)
#         indices = scores < 0.5 # B, C
#         scores[scores>0] *= -1
#         match_loss = scores.sum(-1).mean().reshape(1) # 1
        
#         # 4. replace audio vectors, and random reorder.
#         attended_v[indices] = x_cat.squeeze(-1).squeeze(-1)[indices] # B, C, D/C
#         indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
#         attended_v = torch.gather(attended_v, 1, indices[:, :, None].broadcast_to(attended_v.shape)) # B, C, D/C
#         v_ls = [attended_v[:,i,:].unsqueeze(-1).unsqueeze(-1).broadcast_to((B, int(D/C), F, T)) for i in range(C)]
#         return torch.cat((*v_ls, x), dim=1), (match_loss, maps) # B, 2D, F, T
    
    
#     def forward(self, x, v = None):
#         if self.mtype == "con":
#             return self.concate_forward(x, v)
#         elif self.mtype == "con3":
#             return self.concate3_forward(x, v)
#         elif self.mtype == "vis2in":
#             return self.vis2in_forward(x, v)
#         elif self.mtype == "a2v_att":
#             return self.a2v_att_forward(x, v)
#         elif self.mtype == "hidsep":
#             return self.hidsep_forward(x, v)
#         elif self.mtype == "share":
#             return self.share_forward(x, v)