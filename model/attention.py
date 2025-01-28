import torch
import torch.nn as nn
import math
from typing import Optional, List, Sequence
from model.primitives import Linear, RMSNorm
from data.rigid_utils import Rigid
from data.utils import create_ct_reg_mask
from model.index_embedder import RotaryEmbedding, SinusoidalEmbedding



def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


    



class GeometricAttention(nn.Module):
    def __init__(
        self,
        conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        
        super(GeometricAttention, self).__init__()
        self._conf = conf

        self.c_s = conf.c_s
        self.no_heads = conf.geom_no_heads
        self.no_qk_points = conf.no_qk_points
        self.no_v_points = conf.no_v_points
        self.inf = inf
        self.eps = eps
        self.no_anchors = conf.no_anchors

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)


        self.head_weights = nn.Parameter(torch.zeros((conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_v_points * 4

        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()


    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        r1: Rigid,
        r2: Rigid,
        mask1,
        mask2
    ) -> torch.Tensor:

        if s2.shape[-2] > self.no_anchors:
            selected_residues = torch.randperm(s2.shape[-2])[:self.no_anchors]
            s2 = s2[:,selected_residues]
            r2 = r2[:,selected_residues]
            mask2 = mask2[:,selected_residues]

        
        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s1)

        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r1[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s2)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r2[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        square_mask = mask1.unsqueeze(-1) * mask2.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = pt_att + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum( a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :], dim=-2)

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r1[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        o_feats = [*torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        # [*, N_res, C_s]
        s1 = self.linear_out(torch.cat(o_feats, dim=-1).to(dtype=s1.dtype))
        
        return s1
    


class Attention(nn.Module):
    def __init__(
        self,
        conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        
        super(Attention, self).__init__()
        self._conf = conf

        self.c_s = conf.c_s
        self.c_hidden = conf.c_hidden
        self.no_heads = conf.no_heads
        self.inf = inf
        self.eps = eps
        self.norm_qk = conf.norm_qk
        
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=False)

        self.sin_embed = SinusoidalEmbedding(dim = self.c_s)

        self.linear_out = Linear(self.no_heads * self.c_hidden, self.c_s)

        if self.norm_qk:
            self.norm_q = RMSNorm(self.c_hidden)
            self.norm_k = RMSNorm(self.c_hidden)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()


    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        ridx1,
        ct1,
        mask1,
        mask2
    ) -> torch.Tensor:
    

        ct_idx1 = ridx1[torch.arange(ridx1.size(0)), ct1.int()]
        s1 = s1 + self.sin_embed.embed_pos(ridx1 - ct_idx1.unsqueeze(-1))
        
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s1)
        kv = self.linear_kv(s2)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        if self.norm_qk: 
            q = self.norm_q(q)

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        if self.norm_qk: 
            k = self.norm_k(k)

        square_mask = mask1.unsqueeze(-1) * mask2.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        
        # [*, H, N_res, N_res]
        a = torch.matmul(permute_final_dims(q, (1, 0, 2)), permute_final_dims(k, (1, 2, 0)))
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a = a + square_mask.unsqueeze(-3)

        a = self.softmax(a)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, C_s]
        s1 = self.linear_out(o)
        
        return s1
    



class SelfAttention(nn.Module):
    def __init__(
        self,
        conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(SelfAttention, self).__init__()
        self._conf = conf

        self.c_s = conf.c_s
        self.c_hidden = conf.c_hidden
        self.no_heads = conf.no_heads
        self.inf = inf
        self.eps = eps
        self.norm_qk = conf.norm_qk


        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=False)

        self.rotary_emb = RotaryEmbedding(dim = self.c_hidden)

        concat_out_dim = self.c_hidden

        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s)

        if self.norm_qk:
            self.norm_q = RMSNorm(self.c_hidden)
            self.norm_k = RMSNorm(self.c_hidden)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()


    def forward(self, s, ridx, mask, ct=None) -> torch.Tensor:
        
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        if self.norm_qk:
            q = self.norm_q(q)

        q = self.rotary_emb.rotate_queries_or_keys(q, ridx)
        
        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        if self.norm_qk:
            k = self.norm_k(k)

        k = self.rotary_emb.rotate_queries_or_keys(k, ridx)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        if not ct is None:
            n = mask.shape[-1]
            ct_reg_mask = create_ct_reg_mask(ct, n)
            square_mask *= ct_reg_mask

        square_mask = self.inf * (square_mask - 1)


        # [*, H, N_res, N_res]
        a = torch.matmul(permute_final_dims(q, (1, 0, 2)), permute_final_dims(k, (1, 2, 0)))
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a = a + square_mask.unsqueeze(-3)

        a = self.softmax(a)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, C_s]
        s = self.linear_out(o)
        
        return s



