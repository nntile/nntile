import torch
from torch import nn


class MixerMlp(nn.Module):
    def __init__(self, side: str, dim: int):

        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if dim <= 0:
            raise ValueError("ndim must be positive integer")
        super().__init__()
        self.side = side
        self.dim = dim
        self.fn = nn.Sequential(nn.Linear(self.dim, 4 * self.dim, bias = False), nn.GELU(), 
                                nn.Linear(4 * self.dim, self.dim, bias = False))


    def set_weight(self, w1, w2):
        if self.side == 'L':
            if self.dim != w1.shape[0]:
                raise ValueError("Initialized and loaded weight sizes do not match")
            self.fn[0].weight.data = torch.from_numpy(w1.T)
            self.fn[2].weight.data = torch.from_numpy(w2.T)
        if self.side == 'R':
            if self.dim != w1.shape[1]:
                raise ValueError("Initialized and loaded weight sizes do not match")
            self.fn[0].weight.data = torch.from_numpy(w1)
            self.fn[2].weight.data = torch.from_numpy(w2)


    def forward(self, x: torch.Tensor):
        if self.side == 'L':
            return self.fn(x)
        if self.side == 'R':
            x = torch.transpose(x, 0, 2)
            output = self.fn(x)
            return torch.transpose(output, 0, 2)
        

class Mixer(nn.Module):
    def __init__(self, patch_dim :int, channel_dim: int):       
        super().__init__()
        self.norm_1 = nn.LayerNorm(channel_dim)
        self.norm_2 = nn.LayerNorm(channel_dim)
        self.mlp_1 = MixerMlp('R', patch_dim)
        self.mlp_2 = MixerMlp('L', channel_dim)


    def set_normalization_parameters(self, norm_1_gamma, norm_1_beta, norm_2_gamma, norm_2_beta):
        self.norm_1.weight.data = torch.from_numpy(norm_1_gamma)
        self.norm_1.bias.data = torch.from_numpy(norm_1_beta)
        self.norm_2.weight.data = torch.from_numpy(norm_2_gamma)
        self.norm_2.bias.data = torch.from_numpy(norm_2_beta)


    def set_weight_parameters(self, mlp1_w1, mlp1_w2, mlp2_w1, mlp2_w2):
        self.mlp_1.set_weight(mlp1_w1, mlp1_w2)
        self.mlp_2.set_weight(mlp2_w1, mlp2_w2)


    def forward(self, x: torch.Tensor):       
        y_tmp = self.mlp_1.forward(self.norm_1(x)) + x
        return self.mlp_2.forward(self.norm_2(y_tmp)) + y_tmp
    