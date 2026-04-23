# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama_mlp.py
# LlamaMLP submodule of NNTile Python package
#
# @version 1.1.0

import torch
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLP_torch

from nntile.tensor import TensorMoments, notrans, to_numpy

from ..layer.act import Act
from ..layer.linear import Linear
from ..layer.multiply import Multiply
from .base_model import BaseModel
from .llama_config import LlamaConfigNNTile


class LlamaMLP(BaseModel):

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: LlamaConfigNNTile):
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
        self.bias = config.mlp_bias
        # Initial linear layer that converts input to internal shape
        gate_proj = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.intermediate_size],
            [intermediate_size_tile],
            redux=redux,
            bias=self.bias
        )
        layers.append(gate_proj)
        activations.extend(gate_proj.activations_output)

        new_layer = Act.generate_simple(
            activations[-1], activation_function
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        up_proj = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.intermediate_size],
            [intermediate_size_tile],
            redux=redux,
            bias=self.bias
        )
        layers.append(up_proj)
        activations.extend(up_proj.activations_output)

        multiply_layer = Multiply.generate_simple(
            activations[-2], activations[-1]
        )
        layers.append(multiply_layer)
        activations.extend(multiply_layer.activations_output)

        down_proj = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.hidden_size],
            [hidden_size_tile],
            redux=redux,
            bias=self.bias
        )
        layers.append(down_proj)
        activations.extend(down_proj.activations_output)
        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    # Randomly init all linear layers
    def init_randn_async(self):
        for layer in self.layers:
            if type(layer) is Linear:
                layer.init_randn_async()

    def forward_dynamic(self, x: TensorMoments):
        gate_proj, gate_proj_act, up_proj, multiply, down_proj = self.layers
        gate_outs = gate_proj.forward_dynamic(x)

        gate_act_outs = gate_proj_act.forward_dynamic(gate_outs)
        up_proj_outs = up_proj.forward_dynamic(x)

        multiply_outs = multiply.forward_dynamic(gate_act_outs, up_proj_outs)
        down_proj_outs = down_proj.forward_dynamic(multiply_outs)

        return down_proj_outs

    @staticmethod
    def from_torch(
        torch_mlp, x: TensorMoments, config: LlamaConfigNNTile
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        llama_mlp_nntile = LlamaMLP(x, config)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(llama_mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return llama_mlp_nntile

    def to_torch(self):
        config_torch = LlamaConfig_torch(hidden_size=self.hidden_size,
                                         intermediate_size=self.intermediate_size,
                                         mlp_bias=self.bias)
        llama_mlp_torch = LlamaMLP_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     llama_mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return llama_mlp_torch

    def to_torch_with_grads(self):
        config_torch = LlamaConfig_torch(hidden_size=self.hidden_size,
                                         intermediate_size=self.intermediate_size)
        llama_mlp_torch = LlamaMLP_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                     llama_mlp_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return llama_mlp_torch
