# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/llama_mlp.py
# LlamaMLP layer of NNTile Python package
#
# @version 1.0.0

import torch
from transformers import LlamaConfig as LlamaConfig_torch
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLP_torch

from nntile.layer import Act, Linear, Prod
from nntile.model.base_model import BaseModel
from nntile.model.llama import LlamaConfig as LlamaConfig_nntile
from nntile.tensor import TensorMoments, notrans, to_numpy


class LlamaMLP(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, config: LlamaConfig_nntile,
                 next_tag: int):
        # Init activations and list of layers
        activations = [x]
        layers = []
        self.hidden_size = config["hidden_size"]
        hidden_size_tile = config["hidden_size_tile"]
        self.intermediate_size = config["intermediate_size"]
        intermediate_size_tile = config["intermediate_size_tile"]
        activation_function = config["activation_function"]
        redux = config["redux"]
        gemm_ndim = 1
        # Initial linear layer that converts input to internal shape
        gate_proj, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.intermediate_size],
            [intermediate_size_tile],
            next_tag,
            redux=redux,
            bias=False
        )
        layers.append(gate_proj)
        activations.extend(gate_proj.activations_output)

        new_layer, next_tag = Act.generate_simple(
            activations[-1], activation_function, next_tag
        )
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        up_proj, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.intermediate_size],
            [intermediate_size_tile],
            next_tag,
            redux=redux,
            bias=False
        )
        layers.append(up_proj)
        activations.extend(up_proj.activations_output)
        self.next_tag = next_tag

        prod_layer, next_tag = Prod.generate_simple(
            activations[-2], activations[-1], next_tag
        )
        layers.append(prod_layer)
        activations.extend(prod_layer.activations_output)

        down_proj, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.hidden_size],
            [hidden_size_tile],
            next_tag,
            redux=redux,
            bias=False
        )
        layers.append(down_proj)
        activations.extend(down_proj.activations_output)
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
        torch_mlp, x: TensorMoments, config: LlamaConfig_nntile, next_tag: int
    ):
        """
        torch_mlp is PyTorch MLP where no biases in linear layers
        """
        llama_mlp_nntile = LlamaMLP(x, config, next_tag)
        torch_params = list(torch_mlp.parameters())
        for i, p in enumerate(llama_mlp_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return llama_mlp_nntile, llama_mlp_nntile.next_tag

    def to_torch(self):
        config_torch = LlamaConfig_torch(hidden_size=self.hidden_size,
                                         intermediate_size=self.intermediate_size)
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
