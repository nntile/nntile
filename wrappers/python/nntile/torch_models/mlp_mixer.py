import torch
from torch import nn


class MlpMixer(nn.Module):
    def __init__(self, side: str, dim: int, w1, w2):

        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        
        super().__init__()
        self.side = side
        self.fn = nn.Sequential(nn.Linear(dim, 4 * dim, bias = False), nn.GELU(), nn.Linear(4 * dim, dim, bias = False))
        with torch.no_grad():
            self.fn[0].weight.copy_(torch.from_numpy(w1).float())
            self.fn[2].weight.copy_(torch.from_numpy(w2).float())

    def forward(self, x: torch.Tensor):
        if self.side == 'L':
            return self.fn(x)
        if self.side == 'R':
            x = torch.transpose(x, 0, 2)
            output = self.fn(x)
            return torch.transpose(output, 0, 2)
