import torch
import torch.nn as nn
import numpy as np
import math
from scipy.stats import truncnorm
from typing import Optional, Tuple, Callable, List


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

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2.0
    b = 2.0
    std = math.sqrt(scale)
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=a, b=b)
    '''std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))'''
    

def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)

def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)

def bietti_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0, fan="fan_out")

def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)

def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x): 
        d = x.dtype
        if(d is torch.bfloat16):
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x, 
                    self.c_in, 
                    self.weight.to(dtype=d), 
                    self.bias.to(dtype=d), 
                    self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out
    

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):

        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d
        
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    


class Linear(torch.nn.Linear):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            elif init == "bietti":
                bietti_normal_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
            

class Embedding(torch.nn.Embedding):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        padding_idx: int = None, 
        init: str = "default",
    ):
        super(Embedding, self).__init__(in_dim, out_dim, padding_idx)


        if init == "default":
            lecun_normal_init_(self.weight)
        elif init == "relu":
            he_normal_init_(self.weight)
        elif init == "glorot":
            glorot_uniform_init_(self.weight)
        elif init == "gating":
            gating_init_(self.weight)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        elif init == "bietti":
            bietti_normal_init_(self.weight)
        else:
            raise ValueError("Invalid init string.")