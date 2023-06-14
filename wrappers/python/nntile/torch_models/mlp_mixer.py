import torch
from torch import nn
from nntile.torch_models.mixer import Mixer


def image_patching(image_batch, patch_size):
    h, b, w = image_batch.shape
    if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("patch size should be divisor of image size")
    n_patches = int(h * w / (patch_size ** 2))
    patched_batch = torch.empty((n_patches, b, patch_size ** 2), dtype=image_batch.dtype)

    n_x = int(h / patch_size)
    n_y = int(w / patch_size)
    for i in range(n_patches):
        x = i // n_y
        y = i % n_y
        patched_batch[i, :, :] = image_batch[x * patch_size: (x+1) * patch_size,: , y * patch_size: (y+1) * patch_size].reshape(1, b, patch_size ** 2)
    return patched_batch


class MlpMixer(nn.Module):
    def __init__(self, channel_dim: int, init_patch_dim: int, patch_dim: int, num_mixer_layers: int, n_classes: int):
        super().__init__()
        self.channel_dim = channel_dim
        self.init_patch_size = init_patch_dim
        self.patch_dim = patch_dim
        self.num_mixer_layers = num_mixer_layers

        mixer_layer_blocks = [Mixer(self.channel_dim,self.patch_dim) for _ in range(self.num_mixer_layers)]
        self.mixer_sequence =  nn.Sequential(nn.Linear(self.init_patch_size, self.patch_dim, bias = False), *mixer_layer_blocks)
        self.classification = nn.Linear(self.patch_dim, n_classes, bias = False)

    def forward(self, x: torch.Tensor):
        mixer_output = self.mixer_sequence(x)
        return self.classification(mixer_output.mean(dim=(0)))