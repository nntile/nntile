# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/torch_models/mlp_mixer.py
# PyTorch-based implementation of MLP-Mixer architecture
#
# @version 1.1.0

import torch
from torch import nn


def image_patching_rgb(image_batch, patch_size):
    b, h, w, c = image_batch.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "patch size should be divisor of both image height and width"
        )
    n_patches = int(h * w / (patch_size**2))
    n_channels = c * (patch_size**2)
    patched_batch = torch.empty(
        (n_patches, b, n_channels), dtype=image_batch.dtype
    )

    n_y = int(w / patch_size)
    for batch_iter in range(b):
        for i in range(n_patches):
            x = i // n_y
            y = i % n_y

            for clr in range(c):
                vect_patch = image_batch[
                    batch_iter,
                    x * patch_size : (x + 1) * patch_size,
                    y * patch_size : (y + 1) * patch_size,
                    clr,
                ].flatten()
                patched_batch[
                    i,
                    batch_iter,
                    clr * (patch_size**2) : (clr + 1) * (patch_size**2),
                ] = vect_patch
    return patched_batch, n_patches, n_channels


def image_patching(image_batch, patch_size):
    h, b, w = image_batch.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "patch size should be divisor of image height and width"
        )
    n_patches = int(h * w / (patch_size**2))
    patched_batch = torch.empty(
        (n_patches, b, patch_size**2), dtype=image_batch.dtype
    )

    n_y = int(w / patch_size)
    for i in range(n_patches):
        x = i // n_y
        y = i % n_y
        for j in range(b):
            patched_batch[i, j, :] = image_batch[
                x * patch_size : (x + 1) * patch_size,
                j,
                y * patch_size : (y + 1) * patch_size,
            ].reshape(1, 1, patch_size**2)
    return patched_batch


class MixerMlp(nn.Module):
    def __init__(self, side: str, dim: int):
        if side != "L" and side != "R":
            raise ValueError("side must be either 'L' or 'R'")
        # Check parameter ndim
        if dim <= 0:
            raise ValueError("ndim must be positive integer")
        super().__init__()
        self.side = side
        self.dim = dim
        self.fn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim, bias=False),
        )

    def set_weight(self, w1, w2):
        if self.side == "L":
            if self.dim != w1.shape[0]:
                raise ValueError(
                    "Initialized and loaded weight sizes do not match"
                )
            self.fn[0].weight.data = torch.from_numpy(w1.T)
            self.fn[2].weight.data = torch.from_numpy(w2.T)
        if self.side == "R":
            if self.dim != w1.shape[1]:
                raise ValueError(
                    "Initialized and loaded weight sizes do not match"
                )
            self.fn[0].weight.data = torch.from_numpy(w1)
            self.fn[2].weight.data = torch.from_numpy(w2)

    def forward(self, x: torch.Tensor):
        if self.side == "L":
            return self.fn(x)
        if self.side == "R":
            x = torch.transpose(x, 0, 2)
            output = self.fn(x)
            return torch.transpose(output, 0, 2)


class Mixer(nn.Module):
    # channel dim - dimensionality of each channel - number of rows in a
    # matrix [n_patches x n_channels]
    # patch dim - dimensionality of each patch vector - number of columns in a
    # matrix [n_patches x n_channels]
    def __init__(self, channel_dim: int, patch_dim: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(patch_dim)
        self.mlp_1 = MixerMlp("R", channel_dim)
        self.norm_2 = nn.LayerNorm(patch_dim)
        self.mlp_2 = MixerMlp("L", patch_dim)

    def set_normalization_parameters(
        self, norm_1_gamma, norm_1_beta, norm_2_gamma, norm_2_beta
    ):
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


class MlpMixer(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        init_patch_dim: int,
        patch_dim: int,
        num_mixer_layers: int,
        n_classes: int,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.init_patch_dim = init_patch_dim
        self.projected_patch_dim = patch_dim
        self.num_mixer_layers = num_mixer_layers

        mixer_layer_blocks = [
            Mixer(self.channel_dim, self.projected_patch_dim)
            for _ in range(self.num_mixer_layers)
        ]
        self.mixer_sequence = nn.Sequential(
            nn.Linear(
                self.init_patch_dim, self.projected_patch_dim, bias=False
            ),
            *mixer_layer_blocks,
        )
        self.classification = nn.Linear(
            self.projected_patch_dim, n_classes, bias=False
        )

    def forward(self, x: torch.Tensor):
        mixer_output = self.mixer_sequence(x)
        return self.classification(mixer_output.mean(dim=(0)))

    def evaluate(self, test_data_tensor, test_label_tensor, device):
        num_batch_test, num_minibatch_test = (
            test_data_tensor.shape[0],
            test_data_tensor.shape[1],
        )
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            for test_batch_iter in range(num_batch_test):
                for test_minibatch_iter in range(num_minibatch_test):
                    patched_test_sample = test_data_tensor[
                        test_batch_iter, test_minibatch_iter, :, :, :
                    ]
                    patched_test_sample = patched_test_sample.to(device)
                    true_test_labels = test_label_tensor[
                        test_batch_iter, test_minibatch_iter, :
                    ].to(device)

                    torch_output = self.forward(patched_test_sample)

                    _, predictions = torch.max(torch_output, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(
                        true_test_labels, predictions
                    ):
                        if label == prediction:
                            correct_pred += 1
                        total_pred += 1
        # print accuracy
        print("Total accuracy = {}".format(float(correct_pred / total_pred)))
