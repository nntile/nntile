# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_mlp.py
# LlamaMLP submodule of NNTile Python package
#
# @version 1.0.0

import torch
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP as GPT2MLP_torch, GPT2Config as GPT2Config_torch)

from nntile.tensor import TensorMoments, notrans, to_numpy

from ..layer.act import Act
from ..layer.linear import Linear
from .base_model import BaseModel
from .gpt2_config import GPT2ConfigNNTile


class GPT2MLP(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: GPT2ConfigNNTile, next_tag: int):
        # Init activations and list of layers
        activations = [x]
        layers = []
        self.embed_dim = config["embed_dim"]
        embed_dim_tile = config["embed_dim_tile"]
        self.inner_dim = config["inner_dim"]
        inner_dim_tile = config["inner_dim_tile"]
        activation_function = config["activation_function"]
        redux = config["redux"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.inner_dim],
            [inner_dim_tile],
            next_tag,
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Act.generate_simple(
            activations[-1], activation_function, next_tag
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.embed_dim],
            [embed_dim_tile],
            next_tag,
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for layer in self.layers:
            if type(layer) is Linear:
                layer.init_randn_async()

    @staticmethod
    def from_torch(
        torch_mlp, x: TensorMoments, config: GPT2ConfigNNTile, next_tag: int
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        gpt2mlp_nntile = GPT2MLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(gpt2mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy().T)
        return gpt2mlp_nntile, gpt2mlp_nntile.next_tag

    def to_torch(self):
        config_torch = GPT2Config_torch()
        config_torch.hidden_size = self.embed_dim
        gpt2_mlp_torch = GPT2MLP_torch(self.inner_dim, config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     gpt2_mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value).T,
                                        requires_grad=True)
        return gpt2_mlp_torch

    def to_torch_with_grads(self):
        config_torch = GPT2Config_torch()
        config_torch.hidden_size = self.embed_dim
        gpt2_mlp_torch = GPT2MLP_torch(self.inner_dim, config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     gpt2_mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value).T,
                                        requires_grad=True)
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad).T)
        return gpt2_mlp_torch