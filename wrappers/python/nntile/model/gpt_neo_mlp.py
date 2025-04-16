# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gptneo_mlp.py
# GPTNeoMLP submodule of NNTile Python package
#
# @version 1.1.0

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoConfig as GPTNeoConfigTorch, GPTNeoMLP as GPTNeoMlpTorch)

from nntile.tensor import TensorMoments, notrans, to_numpy

from ..layer.act import Act
from ..layer.linear import Linear
from .base_model import BaseModel
from .gpt_neo_config import GPTNeoConfig


class GPTNeoMLP(BaseModel):

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments,
                 config: GPTNeoConfig):
        # Init activations and list of layers
        activations = [x]
        layers = []
        self.hidden_size = config.hidden_size
        hidden_size_tile = config.hidden_size_tile
        self.intermediate_size = config.intermediate_size
        intermediate_size_tile = config.intermediate_size_tile
        activation_function = config.activation_function
        redux = config.redux
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        new_layer = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.intermediate_size],
            [intermediate_size_tile],
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Act.generate_simple(
            activations[-1], activation_function
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.hidden_size],
            [hidden_size_tile],
            redux=redux,
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(
        mlp_torch, x: TensorMoments, config: GPTNeoConfig
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        mlp_nntile = GPTNeoMLP(x, config)
        torch_params = list(mlp_torch.parameters())
        for i, p in enumerate(mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return mlp_nntile

    def to_torch(self):
        config_torch = GPTNeoConfigTorch()
        config_torch.hidden_size = self.hidden_size
        mlp_torch = GPTNeoMlpTorch(self.intermediate_size, config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return mlp_torch

    def to_torch_with_grads(self):
        config_torch = GPTNeoConfigTorch()
        config_torch.hidden_size = self.hidden_size
        mlp_torch = GPTNeoMlpTorch(self.intermediate_size, config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return mlp_torch
