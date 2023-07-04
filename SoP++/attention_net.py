from curses import meta
from nis import maps
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_attmodule(args):
    if args.fusion_type == "Base":
        return AttModel
    elif args.fusion_type == "MatchAtt":
        return MatchAtt
    assert False


class AttModel(nn.Module):
    def __init__(self, **kwargs):
        super(AttModel, self).__init__()
        self.max_pool = nn.AdaptiveAvgPool3d((None, 1,1))
        att_type = kwargs.get('att_type', 'cos')
        if att_type == "cos":
            def sig_att(x, v):
                """Attention mechanism
                Args:
                    x (torch.Tensor): B x C x D x 1 x 1
                    v (torch.Tensor): B x D x H x W
                Returns:
                    maps (torch.Tensor): B x C x H x W
                """
                return torch.nn.functional.cosine_similarity(x, v.unsqueeze(1), dim=2)
        elif att_type == "sig":
            def sig_att(x, v):
                return torch.sigmoid(torch.sum(x * v.unsqueeze(1) / (x.shape[2])**0.5, dim=2))
        self.att = sig_att
        
    def av_infer_forward(self, aud_feats, mix_vis_feats):
        """Audio visual inference stage forward passing.
        Args:
            aud_feats ( list(torch.Tensor) ): [ B x D x F x T ] * C
            mix_vis_feats ( torch.Tensor ): B x D x H x W
        Returns:
            ctx_feats (torch.Tensor): B, C, D
            (match_loss, maps): (torch.Tensor, ): ( 1, B x C x H x W )
        """
        
        # 1. Get attention maps and scores.
        x_blocks = [self.max_pool(feat) for feat in aud_feats] # [ B x D x 1 x 1] * C
        x_cat = torch.stack(x_blocks, dim=1) # B x C x D x 1 x 1
        maps =  self.att(x_cat, mix_vis_feats) # B x C x H x W
        match_loss = self.max_pool(maps).squeeze(-1).squeeze(-1) # B x C
        maps = torch.clamp(maps, min=0, max=1) # B x C x H x W
        match_loss = -1 * match_loss.sum(-1).mean().reshape(1) # ranging -C to C.
    
        # 2. Compute attention, => Use selection attention.
        ctx_feats = self.max_pool(mix_vis_feats.unsqueeze(1) * maps.unsqueeze(2)).squeeze(-1).squeeze(-1) # B x C x D
        
        return ctx_feats, (match_loss, maps)
        
    def ao_forward(self, aud_feats):
        """Audio Only forward case.

        Args:
            aud_feats (list(torch.Tensor)): [ B x D x F x T ] * C
            # C: The number of sources. 
        return:
            ctx_feats (torch.Tensor): B, C, D
            match_loss: None
        """
        ctx_feats = [self.max_pool(feat).squeeze(-1).squeeze(-1) for feat in aud_feats] # [ B x D ] * C
        ctx_feats = torch.stack(ctx_feats, dim=1) # B x C x D
        match_loss = None
        return ctx_feats, match_loss
        
    
    def av_train_forward(self, aud_feats, mix_vis_feats, sep_vis_feats):
        """Audio visual training forward.
        Args:
            aud_feats (list(torch.Tensor)): [ B x D x F x T ] * C
            mix_vis_feats (torch.Tensor): B x D x H x W
            sep_vis_feats (torch.Tensor): [ B x D x H x W ] * C
        """
        
        # <1> Find ctx_feats and regularization loss.
        ctx_feats, meta = self.av_infer_forward(aud_feats, mix_vis_feats) # B x C x D
        reg_loss = meta[0]
        att_maps = meta[1] # B x C x H x W
        
        # <2> Get global visual features.
        glb_feats = [self.max_pool(feat).squeeze(-1).squeeze(-1) for feat in sep_vis_feats] # [B x D] * C
        glb_feats = torch.stack(glb_feats, dim=1) # B x C x D
        
        # <3> PiT match loss.
        # <3.1> Make permutations of ctx_feats, special case for C = 2.
        p1 = ctx_feats
        p2 = torch.stack((ctx_feats[:,1,:], ctx_feats[:,0,:]), dim=1)
        ctx_feats_p = torch.stack((p1, p2), dim=1) # B x P x C x D
        # <3.2> Get the sources of permutations, and match loss.
        scores = F.cosine_similarity(ctx_feats_p, glb_feats.unsqueeze(1), dim=3) # B x P x C
        scores = scores.sum(-1) # B x P # -2 to 2
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # B, P
        match_loss = -1 * sorted_scores[:, 0] + 1 * sorted_scores[:, 1:].sum(-1)
        match_loss = match_loss.mean(0).reshape(1) # ranging from -4 to 4
        
        # <4> Reorder ctx_feats
        reordered_ctx_feats_p = torch.gather(ctx_feats_p, 1, indices[:,:,None,None].broadcast_to(ctx_feats_p.shape))
        reordered_att_maps = torch.gather(att_maps, 1, indices[:,:,None,None].broadcast_to(att_maps.shape) )
        ctx_feats = reordered_ctx_feats_p[:, 0, :, :] # B x C x D
        return ctx_feats, ( match_loss, reg_loss, reordered_att_maps)
    
    
    def forward(self, aud_feats, mix_vis_feats, sep_vis_feats):
        assert aud_feats is not None
        if mix_vis_feats is None:
            return self.ao_forward(aud_feats)
        else:
            if sep_vis_feats is None:
                return self.av_infer_forward(aud_feats, mix_vis_feats)
            else:
                return self.av_train_forward(aud_feats, mix_vis_feats, sep_vis_feats)

class MatchAtt(nn.Module):
    def __init__(self, **kwargs):
        super(MatchAtt, self).__init__()
        self.max_pool = nn.AdaptiveAvgPool3d((None, 1,1))
        att_type = kwargs.get('att_type', 'cos')
        if att_type == "cos":
            def sig_att(x, v):
                """Attention mechanism
                Args:
                    x (torch.Tensor): B x C x D x 1 x 1
                    v (torch.Tensor): B x D x H x W
                Returns:
                    maps (torch.Tensor): B x C x H x W
                """
                return torch.nn.functional.cosine_similarity(x, v.unsqueeze(1), dim=2)
        elif att_type == "sig":
            def sig_att(x, v):
                return torch.sigmoid(torch.sum(x * v.unsqueeze(1) / (x.shape[2])**0.5, dim=2))
        self.att = sig_att
        
    def av_infer_forward(self, aud_feats, mix_vis_feats):
        """Audio visual inference stage forward passing.
        Args:
            aud_feats ( list(torch.Tensor) ): [ B x D x F x T ] * C
            mix_vis_feats ( torch.Tensor ): B x D x H x W
        Returns:
            ctx_feats (torch.Tensor): B, C, D
            (match_loss, maps): (torch.Tensor, ): ( 1, B x C x H x W )
        """
        
        # 1. Get attention maps and scores.
        x_blocks = [self.max_pool(feat) for feat in aud_feats] # [ B x D x 1 x 1] * C
        x_cat = torch.stack(x_blocks, dim=1) # B x C x D x 1 x 1
        maps =  self.att(x_cat, mix_vis_feats) # B x C x H x W
        match_loss = self.max_pool(maps).squeeze(-1).squeeze(-1) # B x C
        maps = torch.clamp(maps, min=0, max=1) # B x C x H x W
        match_loss = -1 * match_loss.sum(-1).mean().reshape(1) # ranging -C to C.
    
        # 2. Compute attention, => Use selection attention.
        ctx_feats = self.max_pool(mix_vis_feats.unsqueeze(1) * maps.unsqueeze(2)).squeeze(-1).squeeze(-1) # B x C x D
        
        return ctx_feats, (match_loss, maps)
        
    def ao_forward(self, aud_feats):
        """Audio Only forward case.

        Args:
            aud_feats (list(torch.Tensor)): [ B x D x F x T ] * C
            # C: The number of sources. 
        return:
            ctx_feats (torch.Tensor): B, C, D
            match_loss: None
        """
        ctx_feats = [self.max_pool(feat).squeeze(-1).squeeze(-1) for feat in aud_feats] # [ B x D ] * C
        ctx_feats = torch.stack(ctx_feats, dim=1) # B x C x D
        match_loss = None
        return ctx_feats, match_loss
        
    
    def av_train_forward(self, aud_feats, mix_vis_feats, sep_vis_feats):
        """Audio visual training forward.
        Args:
            aud_feats (list(torch.Tensor)): [ B x D x F x T ] * C
            mix_vis_feats (torch.Tensor): B x D x H x W
            sep_vis_feats (torch.Tensor): [ B x D x H x W ] * C
        """
        
        
        # <2> Get global visual features.
        glb_feats = [self.max_pool(feat).squeeze(-1).squeeze(-1) for feat in sep_vis_feats] # [B x D] * C
        glb_feats = torch.stack(glb_feats, dim=1) # B x C x D
        
        # <3> PiT match loss.
        # 1. B x C x D The one needs to be reordered.
        # 2. B x C x D The one with fixed order.
        
        # <3.1> Make permutations of ctx_feats, special case for C = 2.
        
        x_blocks = [self.max_pool(feat).squeeze(-1).squeeze(-1) for feat in aud_feats] # [ B x D ] * C
        x_cat = torch.stack(x_blocks, dim=1) # B x C x D 
        p1 = x_cat
        p2 = torch.stack((x_cat[:,1,:], x_cat[:,0,:]), dim=1)
        x_cat_p = torch.stack((p1, p2), dim=1) # B x P x C x D
        # <3.2> Get the sources of permutations, and match loss.
        scores = F.cosine_similarity(x_cat_p, glb_feats.unsqueeze(1), dim=3) # B x P x C
        scores = scores.sum(-1) # B x P # -2 to 2
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # B, P
        match_loss = -1 * sorted_scores[:, 0] + 1 * sorted_scores[:, 1:].sum(-1)
        match_loss = match_loss.mean(0).reshape(1) # ranging from -4 to 4
        
        # <4> Reorder ctx_feats
        reordered_x_cat_p = torch.gather(x_cat_p, 1, indices[:,:,None,None].broadcast_to(x_cat_p.shape))
        x_cat = reordered_x_cat_p[:, 0, :, :] # B x C x D
        
        # <1> Find ctx_feats and regularization loss.
        aud_feats = [x_cat[:,i,:].unsqueeze(-1).unsqueeze(-1) for i in range(2)]
        ctx_feats, meta = self.av_infer_forward(aud_feats, mix_vis_feats) # B x C x D
        # reg_loss = meta[0]
        att_maps = meta[1] # B x C x H x W

        return ctx_feats, ( match_loss, att_maps)
    
    
    def forward(self, aud_feats, mix_vis_feats, sep_vis_feats):
        assert aud_feats is not None
        if mix_vis_feats is None:
            return self.ao_forward(aud_feats)
        else:
            if sep_vis_feats is None:
                return self.av_infer_forward(aud_feats, mix_vis_feats)
            else:
                return self.av_train_forward(aud_feats, mix_vis_feats, sep_vis_feats)