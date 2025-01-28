import torch
import torch.nn as nn
from model.primitives import Linear, LayerNorm, RMSNorm
from model.attention import Attention, GeometricAttention, SelfAttention



class EncoderAA(nn.Module):

    def __init__(self, model_conf):
        super(EncoderAA, self).__init__()

        self.c_s = model_conf.c_s
        self.n = model_conf.transition_n

        self.self_mha = SelfAttention(model_conf)

        self.geom_mha = GeometricAttention(model_conf)

        self.feed_forward = nn.Sequential(
            Linear(self.c_s, self.n * self.c_s, init='relu'),
            nn.GELU(),
            Linear(self.n * self.c_s, self.c_s, init='glorot')
        )

        self.layer_norm_1 = RMSNorm(self.c_s)
        self.layer_norm_2 = RMSNorm(self.c_s)
        self.layer_norm_3 = RMSNorm(self.c_s)


    def forward(self, s, ridx, T, pad):
        
        s = s + self.self_mha(s, ridx, pad)

        s = self.layer_norm_1(s)

        s = s + self.geom_mha(s, s, T, T, pad, pad)

        s = self.layer_norm_2(s)

        s = s + self.feed_forward(s)

        s = self.layer_norm_3(s)

        return s
    

class EncoderNA(nn.Module):

    def __init__(self, model_conf):
        super(EncoderNA, self).__init__()

        self.c_s = model_conf.c_s
        self.n = model_conf.transition_n

        self.self_mha = SelfAttention(model_conf)

        self.mha = Attention(model_conf)

        self.feed_forward = nn.Sequential(
            Linear(self.c_s, self.n * self.c_s, init='relu'),
            nn.GELU(),
            Linear(self.n * self.c_s, self.c_s, init='glorot')
        )

        self.layer_norm_1 = RMSNorm(self.c_s)
        self.layer_norm_2 = RMSNorm(self.c_s)
        self.layer_norm_3 = RMSNorm(self.c_s)


    def forward(self, s1, s2, ridx1, pad1, pad2, ct1):
        
        s1 = s1 + self.self_mha(s1, ridx1, pad1, ct1)

        s1 = self.layer_norm_1(s1)

        s1 = s1 + self.mha(s1, s2, ridx1, ct1, pad1, pad2)

        s1 = self.layer_norm_2(s1)

        s1 = s1 + self.feed_forward(s1)

        s1 = self.layer_norm_3(s1)

        return s1
    

