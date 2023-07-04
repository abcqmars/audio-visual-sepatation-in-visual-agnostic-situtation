
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_fusion_net(mtype):
    if mtype == "hidsep":
        # The baseline model.
        return CoLoc
    elif mtype == "CoLoc_Sel":
        # Baseline + selection.
        return CoLoc_Sel
    elif mtype == "MixVis":
        # The model taking visual mixture as input.
        return MixVis
    else:
        assert False

class CoLoc(nn.Module):
    def __init__(self, **kwargs):
        super(CoLoc, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = torch.nn.AdaptiveMaxPool3d((None, 1, 1))
        att_type = kwargs.get('att_type', 'cos')
        if att_type == "cos":
            def sig_att(x, v):
                # x: B, P, C, D/C, 1, 1
                return torch.nn.functional.cosine_similarity(x, v.unsqueeze(1), dim=3)
        elif att_type == "sig":
            def sig_att(x, v):
                return torch.sigmoid(torch.sum(x * v.unsqueeze(1) / (x.shape[3])**0.5, dim=3))
        self.att = sig_att
        
    def AV_default_forward(self, x, v_ls, C=2):
        B, D, F, T = x.shape
        # Extract global audio vector.
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
        # x_blocks = torch.split(self.max_pool(x), int(D/C), dim=1) # [ B, D/C, 1, 1] *C
        # Audio Visual separation.
        
        ### 1. Get indices: ###
        # Get all permutations of x: Below is a special case for C = 2
        x_p1 = torch.stack(x_blocks, dim=1)       # B, C, D/C, 1, 1
        x_p2 = torch.stack(x_blocks[::-1], dim=1) # B, C, D/C, 1, 1
        x_t = torch.stack((x_p1, x_p2), dim=1)    # B, P, C, D/C, 1, 1
        
        # Get attention maps with all permutation.
        v_cat = torch.stack(v_ls, dim=1) # B, C, D/C, H, W
        maps = self.att(x_t, v_cat)
        # maps = torch.nn.functional.cosine_similarity(x_t, v_cat.unsqueeze(1), dim=3) # B, P, C, H, W
        # maps = torch.sigmoid(torch.sum(x_t * v_cat.unsqueeze(1) / (x_t.shape[3])**0.5, dim=3))
        # Get scores of attention maps.
        scores = self.max_pool(maps).squeeze(-1).squeeze(-1).sum(-1) # B, P
        # Get indeices for reordering.
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # B, P
        
        ### 2. Compute Match loss ###
        # maximize the highest score, minimize the rest. 
        match_loss = -1 * sorted_scores[:, 0] + 1 * sorted_scores[:, 1:].sum(-1)
        match_loss = match_loss.mean(0)
        
        ### 3. Reorder ###
        maps = torch.gather(maps, 1, indices[:,:1][:,:,None,None,None].broadcast_to(maps.shape)) # B, P, C, H, W
        att_maps = maps[:, 0] # B, C, H, W
        # x_cat = torch.gather(x_t, 1, indices[:,0][:,None,None,None,None,None].broadcast_to(x_cat.shape)).squeeze(1) #, B, C, D/C, 1, 1
        
        ### 4. Attend on imgs ###
        v_cat = self.max_pool(v_cat * att_maps.unsqueeze(2)) # B, C, D/C, 1, 1
        feat_sources = torch.tensor_split(v_cat, C, dim=1)# [B, 1, D/C, 1, 1] * C
        feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
        return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

    def AV_forward(self, x, v_ls, option=None):
        """
        Args:
            x (torch.Tensor): B, C*D/C, F, T, the encoded audio features.
            v_ls (list[torch.Tensor]): [B, D/C, H, W]*C, the visual features.
            option (str, optional): the option of forward pass.
        """
        if option is None:
            return self.AV_default_forward(x, v_ls)
        elif option=='duet':
            # The input visual imgs are concatenated to form one mixture img.
            # The Training criterion should be PiT loss.
            # C = 2:
            assert len(v_ls) == 2
            cat_dim = 2 if torch.rand(1) > 0.5 else 3
            v_cat = torch.cat(v_ls, dim=cat_dim) # B, D/2, H(H*2), W*2(W)
            return self.AV_default_forward(x, [v_cat]*2)
        
        
    def AO_forward(self, x, C=2):
        B, D, F, T = x.shape
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
        indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
        x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
        x_cat = torch.gather(x_cat, 1, indices[:, :, None, None, None].broadcast_to(x_cat.shape)) # B, C, D/C, 1, 1
        x_blocks = torch.tensor_split(x_cat, C, dim=1)
        match_loss = None
        att_maps = None
        feat_sources = x_blocks
        feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
        return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

    def forward(self, x, v_ls, option=None):
        if v_ls is not None:
            return self.AV_forward(x, v_ls, option)
        else:
            return self.AO_forward(x)

class CoLoc_Sel(nn.Module):
    def __init__(self, **kwargs):
        super(CoLoc_Sel, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = torch.nn.AdaptiveMaxPool3d((None, 1, 1))
        att_type = kwargs.get('att_type', 'cos')
        if att_type == "cos":
            def sig_att(x, v):
                # x: B, P, C, D/C, 1, 1
                return torch.nn.functional.cosine_similarity(x, v.unsqueeze(1), dim=3)
        elif att_type == "sig":
            def sig_att(x, v):
                return torch.sigmoid(torch.sum(x * v.unsqueeze(1) / (x.shape[3])**0.5, dim=3))
        self.att = sig_att
    
    
    def AV_default_forward(self, x, v_ls, C=2):
        B, D, F, T = x.shape
        # Extract global audio vector.
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
        # x_blocks = torch.split(self.max_pool(x), int(D/C), dim=1) # [ B, D/C, 1, 1] *C
        # Audio Visual separation.
        
        ### 1. Get indices: ###
        # Get all permutations of x: Below is a special case for C = 2
        x_p1 = torch.stack(x_blocks, dim=1)       # B, C, D/C, 1, 1
        x_p2 = torch.stack(x_blocks[::-1], dim=1) # B, C, D/C, 1, 1
        x_t = torch.stack((x_p1, x_p2), dim=1)    # B, P, C, D/C, 1, 1
        
        # Get attention maps with all permutation.
        v_cat = torch.stack(v_ls, dim=1) # B, C, D/C, H, W
        maps = self.att(x_t, v_cat)
        # maps = torch.nn.functional.cosine_similarity(x_t, v_cat.unsqueeze(1), dim=3) # B, P, C, H, W
        # maps = torch.sigmoid(torch.sum(x_t * v_cat.unsqueeze(1) / (D)**0.5, dim=3))
        
        # Get scores of attention maps.
        scores_BPC = self.max_pool(maps).squeeze(-1).squeeze(-1) # B, P, C
        scores = scores_BPC.sum(-1) # B, P
        # Get indeices for reordering.
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # B, P
        
        ### 2. Compute Match loss ###
        # maximize the highest score, minimize the rest. 
        match_loss = -1 * sorted_scores[:, 0] + 1 * sorted_scores[:, 1:].sum(-1)
        match_loss = match_loss.mean(0)
        
        ### 3. Reorder ###
        maps = torch.gather(maps, 1, indices[:,:,None,None,None].broadcast_to(maps.shape)) # B, P, C, H, W # [:,:1]
        att_maps = maps[:, 0] # B, C, H, W
        # x_cat = torch.gather(x_t, 1, indices[:,0][:,None,None,None,None,None].broadcast_to(x_cat.shape)).squeeze(1) #, B, C, D/C, 1, 1
        
        ### 4. Selection on imgs ###
        # Selection method
        _, max_ind = att_maps.view(B, C, -1).max(-1) # B, C
        max_ind = max_ind.unsqueeze(-1).repeat(1, 1, int(D/C)).unsqueeze(-1) # B, C, D/C, 1
        v_cat_flatt = v_cat.view(B, C, int(D/C), -1) #, B, C, D/C, H*W
        # v_cat = torch.stack(v_ls, dim=1) # B, C, D/C, H, W
        v_cat = torch.gather(v_cat_flatt, dim=3, index=max_ind).unsqueeze(-1) # B, C, D/C, 1, 1
        
        
        ### 5. Replace Low-score visual features. ###
        # Select the permutation with the high score.
        # scores_BPC_selected = torch.gather(scores_BPC, 1, indices[:, :, None].broadcast_to(scores_BPC.shape))[:,0,:] # B, P, C => B, C
    
        # Reorder x_t
        # reordered_x = torch.gather(x_t, 1, indices[:,:,None,None,None, None].broadcast_to(x_t.shape))[:, 0, :] # B, P, C, D/C, 1, 1 => B, C, D/C, 1, 1
        # replace_indices = (scores_BPC_selected <= 0.2)[:,:,None,None,None].broadcast_to(reordered_x.shape)
        # v_cat[replace_indices] = reordered_x[replace_indices]
        
        # indices = scores < 0
        # v_cat[indices] = x_cat.squeeze(-1).squeeze(-1)[indices]
        # Attention method.
        # v_cat = self.max_pool(v_cat * att_maps.unsqueeze(2)) # B, C, D/C, 1, 1
        
        feat_sources = torch.tensor_split(v_cat, C, dim=1)# [B, 1, D/C, 1, 1] * C
        feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
        return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

    def AV_forward(self, x, v_ls, option=None):
        """
        Args:
            x (torch.Tensor): B, C*D/C, F, T, the encoded audio features.
            v_ls (list[torch.Tensor]): [B, D/C, H, W]*C, the visual features.
            option (str, optional): the option of forward pass.
        """
        if option is None:
            return self.AV_default_forward(x, v_ls)
        elif option=='duet':
            # The input visual imgs are concatenated to form one mixture img.
            # The Training criterion should be PiT loss.
            # C = 2:
            assert len(v_ls) == 2
            cat_dim = 2 if torch.rand(1) > 0.5 else 3
            v_cat = torch.cat(v_ls, dim=cat_dim) # B, D/2, H(H*2), W*2(W)
            return self.AV_default_forward(x, [v_cat]*2)
        
        
    def AO_forward(self, x, C=2):
        B, D, F, T = x.shape
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
        indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
        x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
        x_cat = torch.gather(x_cat, 1, indices[:, :, None, None, None].broadcast_to(x_cat.shape)) # B, C, D/C, 1, 1
        x_blocks = torch.tensor_split(x_cat, C, dim=1)
        match_loss = None
        att_maps = None
        feat_sources = x_blocks
        feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
        return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

    def forward(self, x, v_ls, option=None):
        if v_ls is not None:
            return self.AV_forward(x, v_ls, option)
        else:
            return self.AO_forward(x)


class MixVis(nn.Module):
    def __init__(self, **kwargs):
        super(MixVis, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = torch.nn.AdaptiveMaxPool3d((None, 1, 1))
        att_type = kwargs.get('att_type', 'cos')
        if att_type == "cos":
            def sig_att(x, v):
                # x: B, P, C, D/C, 1, 1
                return torch.nn.functional.cosine_similarity(x, v.unsqueeze(1), dim=2)
        elif att_type == "sig":
            def sig_att(x, v):
                return torch.sigmoid(torch.sum(x * v.unsqueeze(1) / (x.shape[2])**0.5, dim=2))
        self.att = sig_att
    
    
    def AV_default_forward(self, x, v, C=2):
        # x:     B, D,   F, T      # Audio encoder ouput
        # v_ls: [B, D/C, H, W*C] * 1 # C visual inputs.
        v = v[0]# B, D/C, H, W *C
        B, D, F, T = x.shape
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1) # [ B, D/C, 1, 1] * C
    
        # 1. Compute Similarity and attention maps.
        x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
        maps =  self.att(x_cat, v) # B, C, H, W*C
    
        # 2. Compute attention, => Use selection attention.
        # attended_v = self.max_pool(v.unsqueeze(1) * maps.unsqueeze(2)).squeeze(-1).squeeze(-1) # B, C, D/C
        viewed_maps = maps.view(B, C, -1) # B, C, H*W*C
        _,_,map_size = viewed_maps.shape
        _, max_ind = viewed_maps.max(-1) # B, C
        max_ind = max_ind.unsqueeze(-1).repeat(1, 1, int(D/C)).unsqueeze(-1) # B, C, D/C, 1
        v_cat_flatt = v.view(B, int(D/C), -1).unsqueeze(1).repeat(1, C, 1, 1) #, B, C, D/C, H*W*C
        
        # v_cat = torch.stack(v_ls, dim=1) # B, C, D/C, H, W
        selected_v = torch.gather(v_cat_flatt, dim=3, index=max_ind).unsqueeze(-1) # B, C, D/C, 1, 1

        # 3. Get scores and matching loss.
        # Optimize the one score on the map with the largest value.
        
        scores = self.max_pool(maps).squeeze(-1).squeeze(-1) * -1 # B, C
        # match_loss = scores.sum(-1).mean().reshape(1)
        match_loss = scores.sum(-1).mean().reshape(1) + maps.view(B, C, -1).sum(-1).sum(-1).mean(-1).reshape(1)/ map_size# 1
        # match_loss /= map_size
        penalty = torch.nn.functional.cosine_similarity(selected_v[:,0], selected_v[:,1], dim=1).mean().reshape(1)
        match_loss += penalty
        # 4. replace audio vectors, and random reorder.
        # attended_v[indices] = x_cat.squeeze(-1).squeeze(-1)[indices] # B, C, D/C
        # indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
        # attended_v = torch.gather(attended_v, 1, indices[:, :, None].broadcast_to(attended_v.shape)) # B, C, D/C
        v_ls = [selected_v[:,i,:].broadcast_to((B, int(D/C), F, T)) for i in range(C)]
        return torch.cat((*v_ls, x), dim=1), (match_loss, maps) # B, 2D, F, T

    def AV_forward(self, x, v_ls, option=None):
        """
        Args:
            x (torch.Tensor): B, C*D/C, F, T, the encoded audio features.
            v_ls (list[torch.Tensor]): [B, D/C, H, W]*C, the visual features.
            option (str, optional): the option of forward pass.
        """
        assert len(v_ls) == 1
        return self.AV_default_forward(x, v_ls)
        
        
    def AO_forward(self, x, C=2):
        B, D, F, T = x.shape
        x_blocks = torch.tensor_split(self.max_pool(x), C, dim=1)
        indices = torch.nn.functional.one_hot((torch.rand(B)>0.5).long()).cuda() # B, C
        x_cat = torch.stack(x_blocks, dim=1) # B, C, D/C, 1, 1
        x_cat = torch.gather(x_cat, 1, indices[:, :, None, None, None].broadcast_to(x_cat.shape)) # B, C, D/C, 1, 1
        x_blocks = torch.tensor_split(x_cat, C, dim=1)
        match_loss = None
        att_maps = None
        feat_sources = x_blocks
        feat_sources = [feat.squeeze(1).broadcast_to((B, int(D/C), F, T)) for feat in feat_sources]
        return torch.cat((*feat_sources, x), dim = 1), (match_loss, att_maps)

    def forward(self, x, v_ls, option=None):
        if v_ls is not None:
            return self.AV_forward(x, v_ls, option)
        else:
            return self.AO_forward(x)
