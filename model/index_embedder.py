import torch
import math
from torch.nn import Module
from torch import nn, einsum

from model.primitives import Linear, RMSNorm

from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')



class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        theta = 1000,
    ):
        super().__init__()

        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad = False)


    def rotate_queries_or_keys(self, x, seq_pos):

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', seq_pos.type(freqs.dtype), freqs)

        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        
        freqs = freqs.unsqueeze(-2)

        return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())



class SinusoidalEmbedding(Module):
    def __init__(
        self,
        dim,
        max_len=512,
    ):
        super().__init__()

        K = torch.arange(dim//2)
        freqs = math.pi / max_len**(2*K[None]/dim)
        self.freqs = nn.Parameter(freqs, requires_grad = False)

        self.proj = Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def embed_pos(self, indices):

        pos_embedding_sin = torch.sin(indices[..., None] * self.freqs)

        pos_embedding_cos = torch.cos(indices[..., None] * self.freqs)

        pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)

        pos_embedding = self.norm(self.proj(pos_embedding))
        
        return pos_embedding