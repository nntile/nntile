import torch
from torch import nn
from nntile.torch_models.mlp_mixer import MlpMixer


class Mixer(nn.Module):
    def __init__(self, w1, w2, w3, w4):       
        super().__init__()
        dim_1 = w1.shape[1]
        dim_2 = w3.shape[1]
        self.norm = nn.LayerNorm(dim_1)
        self.block_mlp_1 = MlpMixer('R', dim_1, w1, w2)
        self.block_mlp_2 = MlpMixer('L', dim_2, w3, w4)


    def forward(self, x: torch.Tensor):
        y_tmp = self.block_mlp_1.forward(x) + x
        return self.block_mlp_2.forward(y_tmp) + y_tmp