# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/deep_linear.py
# Deep Linear model of NNTile Python package
#
# @version 1.1.0

import numpy as np
import torch

from nntile.layer.linear import Linear
from nntile.layer.mixer import GAP, Mixer
from nntile.model.base_model import BaseModel
from nntile.tensor import (
    Tensor_fp32, TensorMoments, TensorTraits, clear_async, notrans)


class MlpMixer(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(
        self,
        x: TensorMoments,
        embedding_dim: int,
        n_layers: int,
        n_classes: int,
        next_tag: int,
    ):
        # Check number of layers
        if n_layers < 1:
            raise ValueError("nlayers must be at least 1")

        self.channel_dim = x.value.shape[0]
        self.init_patch_dim = x.value.shape[2]
        self.projected_patch_dim = embedding_dim
        self.num_mixer_layers = n_layers

        # Init activations and list of layers
        activations = [x]
        layers = []

        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple(
            x,
            "L",
            notrans,
            1,
            [embedding_dim],
            [embedding_dim],
            next_tag,
            bias=False,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        for _ in range(1, n_layers + 1):
            new_layer, next_tag = Mixer.generate_simple(
                activations[-1], next_tag
            )
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)

        # Global Average Pooling Layer
        new_layer, next_tag = GAP.generate_simple(activations[-1], next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        # Final classification fully connected layer
        new_layer, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            1,
            [n_classes],
            [n_classes],
            next_tag,
            bias=False,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Unregister all tensors related to this model
    def unregister(self):
        for layer in self.layers:
            layer.unregister()
        for x in self.activations:
            x.unregister()

    # Clear gradients of inter-layer activations
    def clear_activations_grads(self):
        for t in self.activations:
            if t.grad is not None and t.grad_required:
                clear_async(t.grad)
        for layer in self.layers:
            for t in layer.temporaries:
                if t is not None and t.grad is not None and t.grad_required:
                    clear_async(t.grad)

    # Clear gradients of inter-layer activations
    def clear_tmp_grads(self):
        for layer in self.layers:
            for t in layer.temporaries:
                if t is not None and t.grad is not None and t.grad_required:
                    clear_async(t.grad)

    # Clear all gradients (parameters and inter-layer activations)
    def clear_gradients(self):
        self.clear_parameters_grads()
        self.clear_activations_grads()
        self.clear_tmp_grads()

    @staticmethod
    def from_torch(
        torch_mlp_mixer, batch_size: int, n_classes: int, next_tag: int
    ):
        n_mixer_layers = int((len(list(torch_mlp_mixer.parameters())) - 2) / 8)
        for i, p in enumerate(torch_mlp_mixer.parameters()):
            if i == 0:
                hidden_layer_dim = p.shape[0]
                n_pixels = p.shape[1]
            if i == 3:
                n_patches = p.shape[1]
            # if i == n_layers - 1 and p.shape[0] != n_classes:
            #     raise ValueError("Last layer of PyTorch model does not " \
            #             "correspond to the target number of classes")
        x_traits = TensorTraits(
            [n_patches, batch_size, n_pixels],
            [n_patches, batch_size, n_pixels],
        )
        x_distr = [0]
        x = Tensor_fp32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)

        mlp_mixer_nntile = MlpMixer(
            x_moments, hidden_layer_dim, n_mixer_layers, n_classes, next_tag
        )
        for p, p_torch in zip(
            mlp_mixer_nntile.parameters, torch_mlp_mixer.parameters()
        ):
            if p.value.shape[0] != p_torch.shape[0]:
                p.value.from_array(p_torch.detach().numpy().T)
            else:
                p.value.from_array(p_torch.detach().numpy())
        return mlp_mixer_nntile, mlp_mixer_nntile.next_tag

    def to_torch(self, torch_mlp_mixer):
        with torch.no_grad():
            for p, p_torch in zip(
                self.parameters, torch_mlp_mixer.parameters()
            ):
                if p.value.shape[0] != p_torch.shape[0]:
                    nntile_np = np.zeros(
                        p.value.shape, dtype=np.float32, order="F"
                    )
                    p.value.to_array(nntile_np)
                    p_torch.copy_(torch.from_numpy(nntile_np.T))
                else:
                    nntile_np = np.zeros(
                        p.value.shape, dtype=np.float32, order="F"
                    )
                    p.value.to_array(nntile_np)
                    p_torch.copy_(torch.from_numpy(nntile_np))
