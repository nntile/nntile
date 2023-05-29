import torch
from torch import nn


class MixerMlp(nn.Module):
    def __init__(self, side: str, w1, w2):

        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        
        super().__init__()
        self.side = side
        self.dim = w1.shape[1]
        self.fn = nn.Sequential(nn.Linear(self.dim, 4 * self.dim, bias = False), nn.GELU(), 
                                nn.Linear(4 * self.dim, self.dim, bias = False))
        if self.side == 'L':
            self.fn[0].weight.data = torch.from_numpy(w1.T)
            self.fn[2].weight.data = torch.from_numpy(w2.T)
        if self.side == 'R':
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
    def __init__(self, mlp1_w1, mlp1_w2, mlp2_w1, mlp2_w2):       
        super().__init__()
        dim_1 = mlp1_w1.shape[1]
        dim_2 = mlp2_w1.shape[1]
        self.norm_1 = nn.LayerNorm(dim_1)
        self.norm_2 = nn.LayerNorm(dim_2)
        self.mlp_1 = MixerMlp('R', mlp1_w1, mlp1_w2)
        self.mlp_2 = MixerMlp('L', mlp2_w1, mlp2_w2)


    def forward(self, x: torch.Tensor):
        
        y_tmp = self.mlp_1.forward(x) + x
        return self.mlp_2.forward(y_tmp) + y_tmp
    